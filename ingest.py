"""
Enhanced RAG ingestion system for ADO Wiki content.
Implements robust content processing, intelligent chunking, and efficient vector storage.
"""

import os
import json
import hashlib
import signal
import sys
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import openai

from utils import (
    ConfigManager, WikiContentFetcher, SmartContentChunker, 
    EmbeddingGenerator, ProcessingStats, ContentChunk, logger
)
from azure_search_client import AzureSearchVectorStore

@dataclass
class IngestionState:
    """Tracks the state of ingestion process."""
    last_updated: str
    pages_processed: int
    total_chunks: int
    pages: Dict[str, Dict]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IngestionState':
        return cls(**data)

class RAGIngestor:
    """Main RAG ingestion orchestrator."""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.stats = ProcessingStats()
        self.state_file = Path("ingestion_state.json")
        self.interrupted = False
        
        # Initialize components
        self.wiki_fetcher = WikiContentFetcher(
            config["ADO_ORG"], config["ADO_PROJECT"], 
            config["ADO_WIKI"], config["ADO_PAT"]
        )
        self.chunker = SmartContentChunker(max_tokens=800, overlap_tokens=200)
        self.embedder = EmbeddingGenerator(
            config["AZURE_OPENAI_ENDPOINT"],
            config["AZURE_OPENAI_KEY"],
            config["AZURE_OPENAI_EMBED_MODEL"]
        )
        
        # Setup interrupt handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # OpenAI client for image/mermaid descriptions (used by ingest_images.py subclass)
        self.ai_client = openai.AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
            api_key=config["AZURE_OPENAI_KEY"],
        )
        
        # Flag to enable/disable image processing (overridden in MultimodalRAGIngestor)
        self.process_images = False
        
        # Initialize Azure AI Search
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Initialize and configure Azure AI Search."""
        logger.info("Setting up Azure AI Search vector store...")
        
        try:
            # Check if we have Azure Search configuration
            search_service_name = self.config.get("SEARCH_SERVICE_NAME")
            search_service_key = self.config.get("SEARCH_SERVICE_KEY")
            
            if search_service_name and search_service_key:
                # Use Azure AI Search
                logger.info(f"Using Azure AI Search: {search_service_name}")
                self.vector_store = AzureSearchVectorStore(
                    service_name=search_service_name,
                    api_key=search_service_key,
                    index_name=self.config.get("INDEX_NAME", "ado_wiki")
                )
                self.use_azure_search = True
            else:
                # Fallback to local ChromaDB for development
                logger.info("Azure Search not configured, using local ChromaDB...")
                import chromadb
                from chromadb.config import Settings
                
                os.makedirs(self.config["CHROMA_DB_PATH"], exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(
                    path=self.config["CHROMA_DB_PATH"],
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                try:
                    existing_collection = self.chroma_client.get_collection("ado_wiki")
                    self.chroma_client.delete_collection("ado_wiki")
                    logger.info("Deleted existing ChromaDB collection for fresh start")
                except Exception as e:
                    logger.info(f"No existing ChromaDB collection found: {str(e)}")
                
                self.collection = self.chroma_client.create_collection(
                    name="ado_wiki",
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 200,
                        "hnsw:M": 16
                    }
                )
                self.use_azure_search = False
            
            logger.info("Vector store setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise
    
    def load_state(self) -> IngestionState:
        """Load previous ingestion state or create new one."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                logger.info(f"Loaded previous state: {state_data['pages_processed']} pages processed")
                return IngestionState.from_dict(state_data)
            except Exception as e:
                logger.warning(f"Error loading state: {str(e)}. Starting fresh.")
        
        return IngestionState(
            last_updated=datetime.now().isoformat(),
            pages_processed=0,
            total_chunks=0,
            pages={}
        )
    
    def save_state(self, state: IngestionState):
        """Save current ingestion state."""
        try:
            state.last_updated = datetime.now().isoformat()
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
    
    def _handle_interrupt(self, signum, frame):
        """Handle graceful shutdown on interrupt."""
        logger.info("\nReceived interrupt signal. Initiating graceful shutdown...")
        self.interrupted = True
    
    def describe_image(self, image_url: str, page_path: str, surrounding_text: str = "") -> str:
        """Use GPT-4.1 vision to describe an image from the wiki."""
        try:
            # Build full URL for ADO wiki attachments
            if image_url.startswith("/.attachments/") or image_url.startswith(".attachments/"):
                # ADO wiki attachments: use Git Items API with $format=octetStream
                attachment_name = image_url.split("/")[-1]
                git_url = (
                    f"https://dev.azure.com/{self.config['ADO_ORG']}/{self.config['ADO_PROJECT']}"
                    f"/_apis/git/repositories/{self.config['ADO_WIKI']}/items"
                    f"?path=/.attachments/{attachment_name}"
                    f"&$format=octetStream&api-version=7.1-preview.1"
                )
                res = self.wiki_fetcher.session.get(git_url, headers=self.wiki_fetcher.headers)
                if res.status_code != 200:
                    logger.warning(f"Failed to download image {image_url}: {res.status_code}")
                    return f"[Image: {image_url}]"
                # Determine content type from response or file extension
                ct = res.headers.get('content-type', '')
                if 'png' in ct or image_url.endswith('.png'):
                    mime = 'image/png'
                elif 'jpeg' in ct or 'jpg' in ct or image_url.endswith('.jpg') or image_url.endswith('.jpeg'):
                    mime = 'image/jpeg'
                elif 'gif' in ct or image_url.endswith('.gif'):
                    mime = 'image/gif'
                elif 'svg' in ct or image_url.endswith('.svg'):
                    mime = 'image/svg+xml'
                else:
                    mime = 'image/png'
                img_base64 = base64.b64encode(res.content).decode('utf-8')
                img_data_url = f"data:{mime};base64,{img_base64}"
            elif image_url.startswith("http"):
                img_data_url = image_url
            else:
                return f"[Image: {image_url}]"

            # Build context from page path and surrounding text
            context_hint = f"Wiki page: {page_path}"
            if surrounding_text:
                ctx_text = surrounding_text[:500].strip()
                context_hint += f"\nSurrounding text from the page:\n{ctx_text}"

            resp = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=[
                    {"role": "system", "content": (
                        "You are a technical documentation image describer for an engineering team's wiki. "
                        "Describe images directly and factually. Never start with filler phrases like "
                        "'Certainly', 'Sure', 'Here is', 'This image shows'. "
                        "Start directly with the content description. "
                        "Example: 'Architecture diagram depicting the request flow from client through the load balancer to the backend service.' "
                        "Include all visible text, labels, arrows, flow directions, color coding, and relationships. "
                        "Use the surrounding page context to interpret ambiguous elements in the image."
                    )},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Describe this technical image/diagram.\n\n{context_hint}"},
                        {"type": "image_url", "image_url": {"url": img_data_url}}
                    ]}
                ],
                temperature=0.0,
                max_tokens=500,
            )
            description = resp.choices[0].message.content.strip()
            logger.info(f"Described image {image_url}: {description[:100]}...")
            return description
        except Exception as e:
            logger.warning(f"Image description failed for {image_url}: {str(e)}")
            return f"[Image: {image_url}]"

    def describe_mermaid(self, mermaid_code: str, page_path: str, surrounding_text: str = "") -> str:
        """Use GPT-4.1 to describe a mermaid diagram for embedding."""
        try:
            context_hint = f"Wiki page: {page_path}"
            if surrounding_text:
                ctx_text = surrounding_text[:500].strip()
                context_hint += f"\nSurrounding text from the page:\n{ctx_text}"

            resp = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=[
                    {"role": "system", "content": (
                        "You are a technical documentation diagram describer for an engineering team's wiki. "
                        "Describe diagrams directly and factually. Never start with filler phrases like "
                        "'Certainly', 'Sure', 'Here is', 'This diagram shows'. "
                        "Start directly with the content description. "
                        "Example: 'Flowchart of the feature flag rollout process: starts with the on-call engineer creating a flag in the deployment system.' "
                        "Include all nodes, relationships, flow directions, decision points, and what the diagram represents. "
                        "Use the surrounding page context to interpret node labels and abbreviations."
                    )},
                    {"role": "user", "content": (
                        f"Describe this Mermaid diagram.\n\n{context_hint}\n\n"
                        f"```mermaid\n{mermaid_code}\n```"
                    )}
                ],
                temperature=0.0,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Mermaid description failed: {str(e)}")
            return mermaid_code

    def generate_content_hash(self, page: Dict) -> str:
        """Generate a hash for page content to detect changes."""
        content_for_hash = {
            "content": page.get("content", ""),
            "path": page.get("path", ""),
            "lastModified": page.get("lastModified", "")
        }
        content_str = json.dumps(content_for_hash, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def should_process_page(self, page: Dict, state: IngestionState) -> bool:
        """Determine if a page needs processing based on content changes."""
        page_id = str(page.get("id", page.get("path", "unknown")))
        current_hash = self.generate_content_hash(page)
        
        if page_id not in state.pages:
            return True  # New page
        
        return state.pages[page_id].get("content_hash") != current_hash
    
    def process_page(self, page: Dict, state: IngestionState) -> Tuple[int, bool]:
        """Process a single page and return (chunks_created, success)."""
        page_id = str(page.get("id", page.get("path", "unknown")))
        page_path = page.get("path", "unknown")
        content = page.get("content", "")
        
        logger.info(f"Processing page: {page_path}")
        
        if not content or not content.strip():
            logger.warning(f"Empty content for page: {page_path}")
            return 0, True
        
        try:
            # Chunk the content
            chunks = self.chunker.chunk_content(content, page_path)

            # Delta detection via per-chunk hash (skip unchanged)
            new_chunk_hashes = [hashlib.md5((chunk.text or "").encode("utf-8")).hexdigest() for chunk in chunks]
            old_info = state.pages.get(page_id, {})
            old_chunk_hashes = old_info.get("chunk_hashes", [])
            old_chunk_count = old_info.get("chunk_count", 0)
            
            if not chunks:
                logger.warning(f"No chunks created for page: {page_path}")
                return 0, True

            logger.info(f"Created {len(chunks)} chunks for page: {page_path}")

            # Determine changed chunk indices only (delta ingest)
            changed_indices = [i for i in range(len(chunks)) if i >= len(old_chunk_hashes) or new_chunk_hashes[i] != old_chunk_hashes[i]]

            if not changed_indices and len(chunks) == old_chunk_count:
                logger.info(f"No content changes detected for page: {page_path}")
                return 0, True

            # Generate embeddings only for changed chunks
            changed_chunks = [chunks[i] for i in changed_indices]

            # Process mermaid and image chunks — generate descriptions for better embedding
            # Only when process_images is True (MultimodalRAGIngestor)
            if self.process_images:
                for ci, chunk in enumerate(changed_chunks):
                    # Gather surrounding text from adjacent chunks for context
                    surrounding_parts = []
                    original_idx = changed_indices[ci] if ci < len(changed_indices) else ci
                    if original_idx > 0 and original_idx - 1 < len(chunks):
                        surrounding_parts.append(chunks[original_idx - 1].text[:300])
                    if original_idx + 1 < len(chunks):
                        surrounding_parts.append(chunks[original_idx + 1].text[:300])
                    surrounding_text = "\n---\n".join(surrounding_parts)

                    if chunk.chunk_type == "mermaid" and chunk.mermaid_code:
                        logger.info(f"Describing mermaid diagram in {page_path}")
                        description = self.describe_mermaid(chunk.mermaid_code, page_path, surrounding_text)
                        chunk.text = f"[Mermaid Diagram]\n{description}"
                        chunk.tokens = len(self.chunker.encoder.encode(chunk.text))
                    
                    if chunk.image_url and chunk.image_type not in ("none", "mermaid"):
                        logger.info(f"Describing image {chunk.image_url} in {page_path}")
                        description = self.describe_image(chunk.image_url, page_path, surrounding_text)
                        chunk.text = chunk.text + f"\n\n[Image description: {description}]"
                        chunk.tokens = len(self.chunker.encoder.encode(chunk.text))

            # Contextual chunk enrichment: prepend page path + section title
            # This gives the embedding model richer context about each chunk
            enriched_for_embedding = []
            for chunk in changed_chunks:
                lines = (chunk.text or "").splitlines()
                titles = [ln.lstrip("#").strip() for ln in lines if ln.lstrip().startswith("#")]
                prefix = f"Page: {page_path}"
                if titles:
                    prefix += f" > Section: {titles[0]}"
                enriched_text = f"{prefix}\n\n{chunk.text}"
                enriched_for_embedding.append(ContentChunk(
                    text=enriched_text,
                    tokens=len(self.chunker.encoder.encode(enriched_text)),
                    chunk_type=chunk.chunk_type,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    metadata=chunk.metadata
                ))
            embeddings = self.embedder.embed_chunks(enriched_for_embedding)

            # Filter out empty embeddings and align back to indices
            ids_to_upsert = []
            docs_to_upsert = []
            metas_to_upsert = []
            embs_to_upsert = []

            for idx, (chunk, embedding) in enumerate(zip(changed_chunks, embeddings)):
                if embedding:
                    original_i = changed_indices[idx]
                    ids_to_upsert.append(f"{page_id}_{original_i}")
                    docs_to_upsert.append(chunk.text)
                    # Derive simple header metadata
                    lines = (chunk.text or "").splitlines()
                    titles = []
                    for ln in lines:
                        ls = ln.lstrip()
                        if ls.startswith("#"):
                            titles.append(ls.lstrip("#").strip())
                    section_title = titles[0] if titles else ""
                    headers_path = " > ".join(titles) if titles else ""
                    metas_to_upsert.append({
                        "page_id": page_id,
                        "page_path": page_path,
                        "parent_path": page.get("parent_path", ""),
                        "chunk_index": original_i,
                        "chunk_type": chunk.chunk_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "token_count": chunk.tokens,
                        "section_title": section_title,
                        "headers_path": headers_path,
                        "last_modified": page.get("lastModified") or None,
                        "tags": [],
                        "image_url": chunk.image_url,
                        "image_type": chunk.image_type,
                        "mermaid_code": chunk.mermaid_code
                    })
                    embs_to_upsert.append(embedding)
                else:
                    logger.warning(f"Empty embedding for changed chunk at index {changed_indices[idx]} in page {page_path}")

            # Upsert changed/added chunks first
            if ids_to_upsert:
                if self.use_azure_search:
                    self.vector_store.add_documents(
                        ids=ids_to_upsert,
                        documents=docs_to_upsert,
                        metadatas=metas_to_upsert,
                        embeddings=embs_to_upsert
                    )
                else:
                    # Chroma upsert
                    try:
                        self.collection.add(
                            ids=ids_to_upsert,
                            documents=docs_to_upsert,
                            metadatas=metas_to_upsert,
                            embeddings=embs_to_upsert
                        )
                    except Exception as e:
                        logger.warning(f"Chroma upsert failed, retrying as delete+add: {str(e)}")
                        try:
                            self.collection.delete(ids=ids_to_upsert)
                        except Exception:
                            pass
                        self.collection.add(
                            ids=ids_to_upsert,
                            documents=docs_to_upsert,
                            metadatas=metas_to_upsert,
                            embeddings=embs_to_upsert
                        )

            # Delete stale chunks if new count is smaller
            if old_chunk_count > len(chunks):
                stale_ids = [f"{page_id}_{i}" for i in range(len(chunks), old_chunk_count)]
                try:
                    if self.use_azure_search:
                        self.vector_store.delete_by_ids(stale_ids)
                    else:
                        self.collection.delete(ids=stale_ids)
                    logger.info(f"Deleted {len(stale_ids)} stale chunks for {page_path}")
                except Exception as e:
                    logger.warning(f"Error deleting stale chunks for {page_path}: {str(e)}")

            # Stats
            uploaded = len(ids_to_upsert)
            logger.info(f"Delta ingest complete for {page_path}: upserted {uploaded}, total_chunks_now={len(chunks)}")
            
            logger.info(f"Successfully upserted {uploaded} chunks for {page_path}")
            
            # Update state
            state.pages[page_id] = {
                "content_hash": self.generate_content_hash(page),
                "path": page_path,
                "chunk_count": len(chunks),
                "chunk_hashes": new_chunk_hashes,
                "last_processed": datetime.now().isoformat()
            }
            
            self.stats.chunks_created += uploaded
            self.stats.texts_embedded += len(embs_to_upsert)
            
            return uploaded, True
            
        except Exception as e:
            logger.error(f"Error processing page {page_path}: {str(e)}")
            self.stats.errors_encountered += 1
            return 0, False
    
    def run(self):
        """Main ingestion process."""
        logger.info("Starting RAG ingestion process...")
        
        try:
            # Load previous state
            state = self.load_state()
            self.state = state
            processed_pages = set(state.pages.keys())
            current_batch_pages = set()
            
            logger.info("Fetching wiki pages...")
            
            # Process pages in batches
            batch_count = 0
            total_pages_in_batch = 0
            
            for pages_batch in self.wiki_fetcher.get_wiki_pages(batch_size=25):
                if self.interrupted:
                    logger.info("Interrupted during processing")
                    break
                
                batch_count += 1
                total_pages_in_batch = len(pages_batch)
                
                logger.info(f"\n=== Processing Batch {batch_count} ({total_pages_in_batch} pages) ===")
                
                batch_chunks = 0
                batch_processed = 0
                
                for page in pages_batch:
                    if self.interrupted:
                        break
                    
                    page_id = str(page.get("id", page.get("path", "unknown")))
                    current_batch_pages.add(page_id)
                    
                    # Check if page needs processing
                    if not self.should_process_page(page, state):
                        logger.info(f"Skipping unchanged page: {page.get('path')}")
                        continue
                    
                    # Process the page
                    chunks_created, success = self.process_page(page, state)
                    
                    if success:
                        batch_chunks += chunks_created
                        batch_processed += 1
                        self.stats.pages_processed += 1
                        self.stats.total_content_size += len(page.get("content", ""))
                
                # Save state after each batch
                self.save_state(state)
                
                logger.info(f"Batch {batch_count} complete:")
                logger.info(f"  - Pages processed: {batch_processed}/{total_pages_in_batch}")
                logger.info(f"  - Chunks created: {batch_chunks}")
                logger.info(f"  - Total pages processed: {self.stats.pages_processed}")
                logger.info(f"  - Total chunks: {self.stats.chunks_created}")
            
            # Handle page deletions (pages that were in previous state but not in current batch)
            if not self.interrupted:
                self.cleanup_deleted_pages(state, current_batch_pages)
            
            # Final save
            state.pages_processed = self.stats.pages_processed
            state.total_chunks = self.stats.chunks_created
            self.save_state(state)
            
            # Print final summary
            self.print_summary()
            
        except Exception as e:
            logger.error(f"Fatal error during ingestion: {str(e)}")
            raise
        finally:
            if hasattr(self, 'state'):
                self.save_state(self.state)
    
    def cleanup_deleted_pages(self, state: IngestionState, current_pages: set):
        """Remove chunks for pages that no longer exist."""
        deleted_pages = set(state.pages.keys()) - current_pages
        
        if deleted_pages:
            logger.info(f"Cleaning up {len(deleted_pages)} deleted pages...")
            
            for page_id in deleted_pages:
                try:
                    page_info = state.pages[page_id]
                    chunk_count = page_info.get("chunk_count", 0)
                    
                    if chunk_count > 0:
                        chunk_ids = [f"{page_id}_{i}" for i in range(chunk_count)]
                        if self.use_azure_search:
                            self.vector_store.delete_by_ids(chunk_ids)
                        else:
                            self.collection.delete(ids=chunk_ids)
                        logger.info(f"Deleted {chunk_count} chunks for removed page: {page_info.get('path')}")
                    
                    del state.pages[page_id]
                    
                except Exception as e:
                    logger.error(f"Error cleaning up page {page_id}: {str(e)}")
    
    def print_summary(self):
        """Print final processing summary."""
        logger.info("\n" + "="*60)
        logger.info("INGESTION COMPLETE - SUMMARY")
        logger.info("="*60)
        logger.info(f"Pages processed: {self.stats.pages_processed}")
        logger.info(f"Total chunks created: {self.stats.chunks_created}")
        logger.info(f"Total texts embedded: {self.stats.texts_embedded}")
        logger.info(f"Total content size: {self.stats.total_content_size:,} characters")
        logger.info(f"Errors encountered: {self.stats.errors_encountered}")
        
        if self.stats.pages_processed > 0:
            avg_chunks = self.stats.chunks_created / self.stats.pages_processed
            avg_content = self.stats.total_content_size / self.stats.pages_processed
            logger.info(f"Average chunks per page: {avg_chunks:.1f}")
            logger.info(f"Average content size: {avg_content:,.0f} characters")
        
        logger.info(f"Vector store location: {self.config['CHROMA_DB_PATH']}")
        logger.info(f"State file: {self.state_file}")
        
        if self.interrupted:
            logger.info("Process was interrupted but state has been saved.")
            logger.info("You can resume by running the ingestion again.")
        
        logger.info("="*60)

def main():
    """Main entry point for the ingestion process."""
    try:
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Load configuration
        config = ConfigManager.load_config()
        
        # Create and run ingestor
        ingestor = RAGIngestor(config)
        ingestor.run()
        
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
