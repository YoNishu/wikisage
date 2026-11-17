"""
Shared helpers for WikiSage: config loading, wiki fetching, chunking and embeddings.
"""

import os
import re
import json
import requests
import time
import sys
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import tiktoken
import openai

# Simple console-only logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# Azure SDK imports for Key Vault integration
try:
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure SDK not available. Key Vault integration disabled.")

@dataclass
class ContentChunk:
    """Represents a chunk of content with metadata."""
    text: str
    tokens: int
    chunk_type: str  # 'header', 'paragraph', 'list', 'code', 'mermaid', 'image', 'table'
    start_line: int
    end_line: int
    metadata: Dict[str, Any]
    image_url: str = ""      # URL of embedded image (for image chunks)
    image_type: str = "none"  # 'none', 'mermaid', 'png', 'jpg', 'gif', 'svg'
    mermaid_code: str = ""   # Raw mermaid source code (for mermaid chunks)

@dataclass
class ProcessingStats:
    """Tracks processing statistics."""
    pages_processed: int = 0
    chunks_created: int = 0
    texts_embedded: int = 0
    errors_encountered: int = 0
    total_content_size: int = 0

class ConfigManager:
    """Manages configuration loading and validation."""
    
    @staticmethod
    def load_config() -> Dict[str, str]:
        """Load and validate configuration from environment variables or Key Vault."""
        # Check if running in production (Azure App Service)
        keyvault_url = os.getenv("KEYVAULT_URL")
        environment = os.getenv("ENVIRONMENT", "local")
        
        if keyvault_url and environment == "production" and AZURE_AVAILABLE:
            logger.info("Loading configuration from Azure Key Vault...")
            config = ConfigManager._load_from_keyvault(keyvault_url)
        else:
            logger.info("Loading configuration from environment variables...")
            config = ConfigManager._load_from_env()
        
        # Add Azure Search configuration
        config["SEARCH_SERVICE_NAME"] = os.getenv("SEARCH_SERVICE_NAME", "")
        config["SEARCH_SERVICE_ENDPOINT"] = os.getenv("SEARCH_SERVICE_ENDPOINT", "")
        config["SEARCH_SERVICE_KEY"] = config.get("SEARCH_SERVICE_KEY", "")
        config["INDEX_MODE"] = os.getenv("INDEX_MODE", "text")
        
        # Validate required configuration
        required_keys = [
            "ADO_ORG", "ADO_PROJECT", "ADO_WIKI", "ADO_PAT",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", 
            "AZURE_OPENAI_EMBED_MODEL", "AZURE_OPENAI_CHAT_MODEL"
        ]
        
        # For cloud deployment, also require search service config
        if environment == "production":
            required_keys.extend(["SEARCH_SERVICE_NAME", "SEARCH_SERVICE_KEY"])
        
        missing_keys = [key for key in required_keys if not config.get(key)]
        if missing_keys:
            raise ValueError(f"Missing required configuration: {missing_keys}")
        
        # Debug: Print loaded values to verify they're correct
        logger.info("Configuration loaded successfully:")
        for key, value in config.items():
            if "PAT" in key or "KEY" in key:
                logger.info(f"  {key}: {'*' * 10 if value else 'None'}")
            else:
                logger.info(f"  {key}: {value}")
        
        return config
    
    @staticmethod
    def _load_from_env() -> Dict[str, str]:
        """Load configuration from .env file and environment variables."""
        # Load .env — look in same directory as common.py first, then CWD
        env_path = Path(__file__).parent / ".env"
        if not env_path.exists():
            env_path = Path(".env")
        
        load_dotenv(dotenv_path=env_path, override=True)
        
        return {
            "ADO_ORG": os.getenv("ADO_ORG"),
            "ADO_PROJECT": os.getenv("ADO_PROJECT"),
            "ADO_WIKI": os.getenv("ADO_WIKI"),
            "ADO_PAT": os.getenv("ADO_PAT"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY"),
            "AZURE_OPENAI_EMBED_MODEL": os.getenv("AZURE_OPENAI_EMBED_MODEL"),
            "AZURE_OPENAI_CHAT_MODEL": os.getenv("AZURE_OPENAI_CHAT_MODEL"),
            "CHROMA_DB_PATH": os.getenv("CHROMA_DB_PATH", "./chromadb_storage"),
            "SEARCH_SERVICE_KEY": os.getenv("SEARCH_SERVICE_KEY", "")
        }
    
    @staticmethod
    def _load_from_keyvault(keyvault_url: str) -> Dict[str, str]:
        """Load configuration from Azure Key Vault."""
        try:
            # Use managed identity in production
            credential = ManagedIdentityCredential()
            client = SecretClient(vault_url=keyvault_url, credential=credential)
            
            # Map of environment variable names to Key Vault secret names
            secret_mappings = {
                "ADO_ORG": "ADO-ORG",
                "ADO_PROJECT": "ADO-PROJECT", 
                "ADO_WIKI": "ADO-WIKI",
                "ADO_PAT": "ADO-PAT",
                "AZURE_OPENAI_ENDPOINT": "AZURE-OPENAI-ENDPOINT",
                "AZURE_OPENAI_KEY": "AZURE-OPENAI-KEY",
                "AZURE_OPENAI_EMBED_MODEL": "AZURE-OPENAI-EMBED-MODEL",
                "AZURE_OPENAI_CHAT_MODEL": "AZURE-OPENAI-CHAT-MODEL",
                "SEARCH_SERVICE_KEY": "SEARCH-SERVICE-KEY"
            }
            
            config = {}
            for env_key, secret_name in secret_mappings.items():
                try:
                    secret = client.get_secret(secret_name)
                    config[env_key] = secret.value
                    logger.info(f"Retrieved secret: {secret_name}")
                except Exception as e:
                    logger.warning(f"Could not retrieve secret {secret_name}: {str(e)}")
                    config[env_key] = ""
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading from Key Vault: {str(e)}")
            logger.info("Falling back to environment variables...")
            return ConfigManager._load_from_env()

class WikiContentFetcher:
    """Handles fetching content from Azure DevOps Wiki API."""
    
    def __init__(self, org: str, project: str, wiki: str, pat: str):
        self.org = org
        self.project = project
        self.wiki = wiki
        self.pat = pat
        self.headers = self._setup_headers()
        self.session = self._setup_session()
        
    def _setup_headers(self) -> Dict[str, str]:
        """Setup authentication headers."""
        import base64
        authorization = str(base64.b64encode(f":{self.pat}".encode('utf-8')), 'utf-8')
        return {
            'Accept': 'application/json',
            'Authorization': f'Basic {authorization}'
        }
    
    def _setup_session(self) -> requests.Session:
        """Setup HTTP session with retry strategy."""
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        return session
    
    def get_wiki_pages(self, batch_size: int = 50) -> Generator[List[Dict], None, None]:
        """Fetch all wiki pages recursively in batches."""
        try:
            # Get wiki details
            wiki_url = f"https://dev.azure.com/{self.org}/{self.project}/_apis/wiki/wikis/{self.wiki}?api-version=7.1-preview.1"
            logger.info(f"Fetching wiki details from: {wiki_url}")
            
            wiki_res = self.session.get(wiki_url, headers=self.headers)
            wiki_res.raise_for_status()
            wiki_data = wiki_res.json()
            
            # Get the wiki ID
            if 'value' in wiki_data and len(wiki_data['value']) > 0:
                wiki_details = wiki_data['value'][0]
            else:
                wiki_details = wiki_data
            
            wiki_id = wiki_details.get('id') or wiki_details.get('wikiIdentifier') or self.wiki
            logger.info(f"Using wiki ID: {wiki_id}")
            
            # Get pages structure
            base_url = f"https://dev.azure.com/{self.org}/{self.project}/_apis/wiki/wikis/{wiki_id}/pages"
            params = {
                "path": "/",
                "recursionLevel": "Full",
                "includeContent": "False",
                "api-version": "7.1-preview.1"
            }
            
            structure_res = self.session.get(base_url, headers=self.headers, params=params)
            structure_res.raise_for_status()
            root_data = structure_res.json()
            
            # Process pages in batches
            yield from self._process_pages_batch(base_url, root_data, batch_size)
            
        except Exception as e:
            logger.error(f"Error fetching wiki pages: {str(e)}")
            raise
    
    def _process_pages_batch(self, base_url: str, root_data: Dict, batch_size: int) -> Generator[List[Dict], None, None]:
        """Process pages recursively in batches."""
        current_batch = []
        
        def process_recursive(page: Dict):
            nonlocal current_batch
            
            try:
                # Fetch content for this page
                path = page["path"]
                logger.info(f"Fetching content for: {path}")
                
                page_with_content = self._fetch_page_content(base_url, path)
                page["content"] = page_with_content.get("content", "")
                
                # Compute parent path
                parts = path.strip('/').split('/')
                page["parent_path"] = '/' + '/'.join(parts[:-1]) if len(parts) > 1 else ""
                
                current_batch.append(page)
                logger.info(f"Added page to batch. Content length: {len(page['content'])}")
                
                # Yield batch when full
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
                
                # Process subpages
                for sub_page in page.get("subPages", []):
                    yield from process_recursive(sub_page)
                    
            except Exception as e:
                logger.error(f"Error processing page {page.get('path', 'unknown')}: {str(e)}")
                # Continue processing other pages
        
        # Start recursive processing
        yield from process_recursive(root_data)
        
        # Yield final batch
        if current_batch:
            yield current_batch
    
    def _fetch_page_content(self, base_url: str, path: str) -> Dict:
        """Fetch content for a single page."""
        params = {
            "path": path,
            "includeContent": "true",
            "api-version": "7.1-preview.1"
        }
        
        try:
            res = self.session.get(base_url, headers=self.headers, params=params)
            res.raise_for_status()
            time.sleep(0.1)  # Rate limiting
            return res.json()
        except Exception as e:
            logger.error(f"Error fetching content for {path}: {str(e)}")
            return {"content": f"Error fetching content: {str(e)}"}

class SmartContentChunker:
    """Advanced content chunking with semantic awareness."""
    
    def __init__(self, max_tokens: int = 800, overlap_tokens: int = 200):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def extract_image_refs(self, text: str) -> List[Dict[str, str]]:
        """Extract markdown image references from text."""
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        images = []
        for match in re.finditer(pattern, text):
            alt_text = match.group(1)
            url = match.group(2)
            # Skip base64 data URIs
            if url.startswith("data:"):
                continue
            ext = url.rsplit('.', 1)[-1].lower().split('?')[0] if '.' in url else ''
            img_type = ext if ext in ('png', 'jpg', 'jpeg', 'gif', 'svg') else 'png'
            if img_type == 'jpeg':
                img_type = 'jpg'
            images.append({"alt": alt_text, "url": url, "type": img_type})
        return images

    def chunk_content(self, text: str, page_path: str) -> List[ContentChunk]:
        """Smart chunking that preserves semantic meaning."""
        if not text or not text.strip():
            logger.warning(f"Empty content for page: {page_path}")
            return []
        
        logger.info(f"Chunking content for: {page_path} (length: {len(text)})")
        
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                logger.warning(f"No content after cleaning for: {page_path}")
                return []
            
            # Parse document structure
            sections = self._parse_document_structure(cleaned_text)
            
            # Create chunks with semantic boundaries
            chunks = self._create_semantic_chunks(sections, page_path)
            
            logger.info(f"Created {len(chunks)} chunks for: {page_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking content for {page_path}: {str(e)}")
            # Fallback to simple chunking
            return self._fallback_chunking(cleaned_text, page_path)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content. Preserve image references as markers."""
        # Remove base64 image data
        if text.startswith("data:image"):
            return ""
        
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            # Strip raw HTML img tags but keep markdown image refs ![alt](url)
            if line.startswith("<img"):
                continue
            # Skip empty lines in succession
            if not line and lines and not lines[-1]:
                continue
            lines.append(line)
        
        return "\n".join(lines)
    
    def _parse_document_structure(self, text: str) -> List[Dict]:
        """Parse document into structured sections."""
        lines = text.split('\n')
        sections = []
        current_section = {"type": "content", "lines": [], "level": 0}
        
        for i, line in enumerate(lines):
            # Detect headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save current section
                if current_section["lines"]:
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                current_section = {
                    "type": "header",
                    "level": level,
                    "title": header_match.group(2),
                    "lines": [line],
                    "start_line": i
                }
            # Detect code blocks (and mermaid diagrams)
            elif line.startswith("```"):
                if current_section["type"] not in ("code", "mermaid"):
                    if current_section["lines"]:
                        sections.append(current_section)
                    # Detect mermaid blocks specifically
                    lang = line[3:].strip().lower()
                    if lang == "mermaid":
                        current_section = {"type": "mermaid", "lines": [line], "level": 0, "start_line": i}
                    else:
                        current_section = {"type": "code", "lines": [line], "level": 0, "start_line": i}
                else:
                    current_section["lines"].append(line)
                    sections.append(current_section)
                    current_section = {"type": "content", "lines": [], "level": 0, "start_line": i + 1}
            # Detect lists
            elif re.match(r'^[\s]*[-*+]\s+', line) or re.match(r'^[\s]*\d+\.\s+', line):
                if current_section["type"] != "list":
                    if current_section["lines"]:
                        sections.append(current_section)
                    current_section = {"type": "list", "lines": [line], "level": 0, "start_line": i}
                else:
                    current_section["lines"].append(line)
            else:
                current_section["lines"].append(line)
        
        # Add final section
        if current_section["lines"]:
            sections.append(current_section)
        
        return sections
    
    def _create_semantic_chunks(self, sections: List[Dict], page_path: str) -> List[ContentChunk]:
        """Create chunks that respect semantic boundaries."""
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        chunk_start_line = 0
        
        for section in sections:
            section_text = "\n".join(section["lines"])
            section_tokens = len(self.encoder.encode(section_text))
            
            # If section alone exceeds max tokens, split it
            if section_tokens > self.max_tokens:
                # Save current chunk if exists
                if current_chunk_lines:
                    chunks.append(self._create_chunk(
                        current_chunk_lines, chunk_start_line, 
                        len(current_chunk_lines) + chunk_start_line - 1, page_path
                    ))
                    current_chunk_lines = []
                    current_tokens = 0
                
                # Split large section
                chunks.extend(self._split_large_section(section, page_path))
                chunk_start_line = section.get("start_line", 0) + len(section["lines"])
                
            # If adding section would exceed limit, finalize current chunk
            elif current_tokens + section_tokens > self.max_tokens:
                if current_chunk_lines:
                    chunks.append(self._create_chunk(
                        current_chunk_lines, chunk_start_line,
                        len(current_chunk_lines) + chunk_start_line - 1, page_path
                    ))
                
                # Start new chunk with overlap if applicable
                overlap_lines = self._get_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines + section["lines"]
                current_tokens = len(self.encoder.encode("\n".join(current_chunk_lines)))
                chunk_start_line = section.get("start_line", 0) - len(overlap_lines)
                
            else:
                # Add section to current chunk
                current_chunk_lines.extend(section["lines"])
                current_tokens += section_tokens
        
        # Add final chunk
        if current_chunk_lines:
            chunks.append(self._create_chunk(
                current_chunk_lines, chunk_start_line,
                len(current_chunk_lines) + chunk_start_line - 1, page_path
            ))
        
        return chunks
    
    def _create_chunk(self, lines: List[str], start_line: int, end_line: int, page_path: str) -> ContentChunk:
        """Create a content chunk from lines."""
        text = "\n".join(lines).strip()
        tokens = len(self.encoder.encode(text))
        
        # Determine chunk type
        chunk_type = "paragraph"
        has_mermaid = any(line.strip().lower().startswith("```mermaid") for line in lines)
        if has_mermaid:
            chunk_type = "mermaid"
        elif any(line.startswith("#") for line in lines):
            chunk_type = "header"
        elif any(line.strip().startswith("```") for line in lines):
            chunk_type = "code"
        elif any(re.match(r'^[\s]*[-*+]\s+', line) for line in lines):
            chunk_type = "list"
        
        # Detect image references
        image_refs = self.extract_image_refs(text)
        image_url = image_refs[0]["url"] if image_refs else ""
        image_type_val = image_refs[0]["type"] if image_refs else "none"
        
        # For mermaid chunks, extract raw code
        mermaid_code = ""
        if chunk_type == "mermaid":
            mermaid_lines = [l for l in lines if not l.strip().startswith("```")]
            mermaid_code = "\n".join(mermaid_lines).strip()
            image_type_val = "mermaid"
        
        return ContentChunk(
            text=text,
            tokens=tokens,
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
            metadata={"page_path": page_path, "chunk_type": chunk_type},
            image_url=image_url,
            image_type=image_type_val,
            mermaid_code=mermaid_code
        )
    
    def _split_large_section(self, section: Dict, page_path: str) -> List[ContentChunk]:
        """Split a large section into smaller chunks."""
        lines = section["lines"]
        chunks = []
        current_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = len(self.encoder.encode(line))
            
            if current_tokens + line_tokens > self.max_tokens:
                if current_lines:
                    chunks.append(self._create_chunk(
                        current_lines, section.get("start_line", 0),
                        section.get("start_line", 0) + len(current_lines) - 1, page_path
                    ))
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens
        
        if current_lines:
            chunks.append(self._create_chunk(
                current_lines, section.get("start_line", 0),
                section.get("start_line", 0) + len(current_lines) - 1, page_path
            ))
        
        return chunks
    
    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get overlap lines for context continuity."""
        if not lines:
            return []
        
        overlap_text = ""
        overlap_lines = []
        
        # Take lines from the end until we reach overlap token limit
        for line in reversed(lines):
            test_text = line + "\n" + overlap_text
            if len(self.encoder.encode(test_text)) <= self.overlap_tokens:
                overlap_text = test_text
                overlap_lines.insert(0, line)
            else:
                break
        
        return overlap_lines
    
    def _fallback_chunking(self, text: str, page_path: str) -> List[ContentChunk]:
        """Simple fallback chunking when semantic chunking fails."""
        logger.warning(f"Using fallback chunking for: {page_path}")
        
        words = text.split()
        chunks = []
        current_words = []
        current_tokens = 0
        
        for word in words:
            word_tokens = len(self.encoder.encode(word))
            
            if current_tokens + word_tokens > self.max_tokens:
                if current_words:
                    chunk_text = " ".join(current_words)
                    chunks.append(ContentChunk(
                        text=chunk_text,
                        tokens=len(self.encoder.encode(chunk_text)),
                        chunk_type="fallback",
                        start_line=0,
                        end_line=0,
                        metadata={"page_path": page_path, "chunk_type": "fallback"}
                    ))
                current_words = [word]
                current_tokens = word_tokens
            else:
                current_words.append(word)
                current_tokens += word_tokens
        
        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append(ContentChunk(
                text=chunk_text,
                tokens=len(self.encoder.encode(chunk_text)),
                chunk_type="fallback",
                start_line=0,
                end_line=0,
                metadata={"page_path": page_path, "chunk_type": "fallback"}
            ))
        
        return chunks

class EmbeddingGenerator:
    """Efficient embedding generation with batching and caching."""
    
    def __init__(self, endpoint: str, api_key: str, model_name: str):
        self.client = openai.AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        self.model_name = model_name
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8000  # Safe limit for embedding models
        
    def embed_chunks(self, chunks: List[ContentChunk]) -> List[List[float]]:
        """Generate embeddings for a list of chunks."""
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Validate chunk sizes
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.tokens <= self.max_tokens:
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Chunk {i} has {chunk.tokens} tokens, exceeding limit. Skipping.")
        
        if not valid_chunks:
            logger.error("No valid chunks for embedding")
            return []
        
        # Generate embeddings
        embeddings = []
        batch_size = 10  # Process in small batches to avoid rate limits
        
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                logger.info(f"Embedding batch {i//batch_size + 1}/{(len(valid_chunks) + batch_size - 1)//batch_size}")
                
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        logger.info(f"Successfully generated {len([e for e in embeddings if e])} embeddings")
        return embeddings

# Convenience functions for backward compatibility
def load_config():
    """Load configuration (backward compatibility)."""
    return ConfigManager.load_config()

def get_wiki_pages(org: str, project: str, wiki: str, pat: str):
    """Get wiki pages (backward compatibility)."""
    fetcher = WikiContentFetcher(org, project, wiki, pat)
    return fetcher.get_wiki_pages()

def chunk_markdown(text: str, max_tokens: int = 1500):
    """Chunk markdown content (backward compatibility)."""
    chunker = SmartContentChunker(max_tokens=max_tokens)
    chunks = chunker.chunk_content(text, "unknown")
    # Return simple objects for compatibility
    return [type("Chunk", (), {"text": chunk.text})() for chunk in chunks]

def embed_texts(texts: List[str], model_name: str):
    """Embed texts (backward compatibility)."""
    config = load_config()
    generator = EmbeddingGenerator(
        config["AZURE_OPENAI_ENDPOINT"],
        config["AZURE_OPENAI_KEY"],
        model_name
    )
    
    # Convert to chunks for processing
    chunks = [ContentChunk(
        text=text,
        tokens=len(tiktoken.get_encoding("cl100k_base").encode(text)),
        chunk_type="text",
        start_line=0,
        end_line=0,
        metadata={}
    ) for text in texts]
    
    return generator.embed_chunks(chunks)
