"""
Azure AI Search client wrapper to replace ChromaDB functionality.
Provides a simple interface for vector storage and retrieval using Azure AI Search.
"""

import json
import time
import base64
import os
from typing import List, Dict, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
from common import logger

def encode_document_id(doc_id: str) -> str:
    """Encode document ID to be Azure Search compatible using URL-safe base64."""
    try:
        # Convert to URL-safe base64
        encoded = base64.urlsafe_b64encode(doc_id.encode('utf-8')).decode('ascii')
        # Remove padding to make it cleaner
        return encoded.rstrip('=')
    except Exception as e:
        logger.error(f"Error encoding document ID '{doc_id}': {str(e)}")
        # Fallback: replace invalid characters with underscores
        return doc_id.replace('/', '_').replace(' ', '_').replace('[', '_').replace(']', '_')

def decode_document_id(encoded_id: str) -> str:
    """Decode document ID back to original from URL-safe base64."""
    try:
        # Add padding back if needed (length must be multiple of 4)
        pad_len = (-len(encoded_id)) % 4
        return base64.urlsafe_b64decode(encoded_id + ("=" * pad_len)).decode('utf-8')
    except Exception as e:
        logger.error(f"Error decoding document ID '{encoded_id}': {str(e)}")
        # Return as-is if decoding fails
        return encoded_id

class AzureSearchVectorStore:
    """Azure AI Search vector store implementation."""
    
    def __init__(self, service_name: str, api_key: str, index_name: str = "ado_wiki"):
        self.service_name = service_name
        self.api_key = api_key
        self.index_name = index_name or os.getenv("INDEX_NAME", "ado_wiki")
        self.endpoint = f"https://{service_name}.search.windows.net"
        
        # Initialize clients
        self.credential = AzureKeyCredential(api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
    
    def _create_index_if_not_exists(self):
        """Create the search index if it doesn't exist."""
        try:
            # Check if index exists
            try:
                existing = self.index_client.get_index(self.index_name)
                # Ensure required fields/config exist; update in-place if missing
                existing_field_names = {f.name for f in existing.fields}
                updated = False

                def ensure_field(field):
                    nonlocal updated
                    if field.name not in existing_field_names:
                        existing.fields.append(field)
                        updated = True

                # New/required fields
                ensure_field(SearchableField(name="section_title", type=SearchFieldDataType.String, filterable=True, facetable=True))
                ensure_field(SearchableField(name="headers_path", type=SearchFieldDataType.String, filterable=True))
                ensure_field(SimpleField(name="last_modified", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True))
                ensure_field(SearchField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True))
                # Image/mermaid fields
                ensure_field(SimpleField(name="image_url", type=SearchFieldDataType.String, filterable=True))
                ensure_field(SimpleField(name="image_type", type=SearchFieldDataType.String, filterable=True, facetable=True))
                ensure_field(SearchableField(name="mermaid_code", type=SearchFieldDataType.String))

                # Ensure semantic configuration exists
                if existing.semantic_search is None or not any(c.name == "my-semantic-config" for c in existing.semantic_search.configurations):
                    semantic_config = SemanticConfiguration(
                        name="my-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            content_fields=[SemanticField(field_name="content")]
                        )
                    )
                    if existing.semantic_search is None:
                        existing.semantic_search = SemanticSearch(configurations=[semantic_config])
                    else:
                        existing.semantic_search.configurations.append(semantic_config)
                    updated = True

                if updated:
                    self.index_client.create_or_update_index(existing)
                    logger.info(f"Updated index '{self.index_name}' with new fields/semantic config")
                else:
                    logger.info(f"Index '{self.index_name}' already exists and is up to date")
                return
            except Exception:
                # Index doesn't exist, create it
                pass
            
            logger.info(f"Creating search index: {self.index_name}")
            
            # Define the index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="page_id", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="page_path", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="parent_path", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
                SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="start_line", type=SearchFieldDataType.Int32),
                SimpleField(name="end_line", type=SearchFieldDataType.Int32),
                SimpleField(name="token_count", type=SearchFieldDataType.Int32, filterable=True),
                SearchableField(name="section_title", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="headers_path", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="last_modified", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SearchField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True),
                SimpleField(name="image_url", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="image_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="mermaid_code", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,  # text-embedding-3-large dimensions
                    vector_search_profile_name="my-vector-config"
                )
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="my-hnsw-algorithm",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="my-vector-config",
                        algorithm_configuration_name="my-hnsw-algorithm"
                    )
                ]
            )
            
            # Configure semantic search
            semantic_config = SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")]
                )
            )
            
            semantic_search = SemanticSearch(configurations=[semantic_config])
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            self.index_client.create_index(index)
            logger.info(f"Successfully created index: {self.index_name}")
            
            # Wait for index to be ready
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error creating search index: {str(e)}")
            raise
    
    def add_documents(self, ids: List[str], documents: List[str], metadatas: List[Dict], embeddings: List[List[float]]):
        """Add documents to the search index (equivalent to ChromaDB add)."""
        try:
            search_documents = []
            
            for i, (doc_id, content, metadata, embedding) in enumerate(zip(ids, documents, metadatas, embeddings)):
                if not embedding:  # Skip empty embeddings
                    logger.warning(f"Skipping document {doc_id} due to empty embedding")
                    continue
                
                # Encode the document ID to make it Azure Search compatible
                encoded_id = encode_document_id(doc_id)
                
                document = {
                    "id": encoded_id,  # Use encoded ID for Azure Search
                    "content": content,
                    "page_id": metadata.get("page_id", ""),
                    "page_path": metadata.get("page_path", ""),
                    "parent_path": metadata.get("parent_path", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "chunk_type": metadata.get("chunk_type", ""),
                    "start_line": metadata.get("start_line", 0),
                    "end_line": metadata.get("end_line", 0),
                    "token_count": metadata.get("token_count", 0),
                    "section_title": metadata.get("section_title", ""),
                    "headers_path": metadata.get("headers_path", ""),
                    "last_modified": metadata.get("last_modified") or None,
                    "tags": metadata.get("tags") or [],
                    "image_url": metadata.get("image_url", ""),
                    "image_type": metadata.get("image_type", "none"),
                    "mermaid_code": metadata.get("mermaid_code", ""),
                    "content_vector": embedding
                }
                search_documents.append(document)
            
            if not search_documents:
                logger.warning("No valid documents to add")
                return
            
            # Upload documents in batches
            batch_size = 100
            for i in range(0, len(search_documents), batch_size):
                batch = search_documents[i:i + batch_size]
                result = self.search_client.upload_documents(documents=batch)
                
                # Check for any failures
                for item in result:
                    if not item.succeeded:
                        logger.error(f"Failed to upload document {item.key}: {item.error_message}")
                
                logger.info(f"Uploaded batch {i//batch_size + 1}: {len(batch)} documents")
                time.sleep(1)  # Rate limiting
            
            logger.info(f"Successfully added {len(search_documents)} documents to index")
            
        except Exception as e:
            logger.error(f"Error adding documents to search index: {str(e)}")
            raise
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 10, include: List[str] = None) -> Dict:
        """Query the search index (equivalent to ChromaDB query)."""
        try:
            if not query_embeddings or not query_embeddings[0]:
                logger.error("Empty query embedding provided")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            query_vector = query_embeddings[0]  # Take first embedding
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=n_results,
                fields="content_vector"
            )
            
            # Perform search
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=n_results,
                select=["id", "content", "page_id", "page_path", "parent_path", 
                       "chunk_index", "chunk_type", "start_line", "end_line", "token_count",
                       "image_url", "image_type", "mermaid_code"]
            )
            
            # Format results to match ChromaDB format
            documents = []
            metadatas = []
            distances = []
            
            for result in results:
                documents.append(result["content"])
                
                metadata = {
                    "page_id": result.get("page_id", ""),
                    "page_path": result.get("page_path", ""),
                    "parent_path": result.get("parent_path", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "token_count": result.get("token_count", 0),
                    "image_url": result.get("image_url", ""),
                    "image_type": result.get("image_type", "none"),
                    "mermaid_code": result.get("mermaid_code", "")
                }
                metadatas.append(metadata)
                
                # Convert search score to distance (inverse relationship)
                # Azure Search scores are typically 0-1, higher is better
                # We convert to distance where lower is better
                score = result.get("@search.score", 0.0)
                distance = 1.0 - min(score, 1.0)  # Ensure distance is between 0-1
                distances.append(distance)
            
            return {
                "documents": [documents],
                "metadatas": [metadatas], 
                "distances": [distances]
            }
            
        except Exception as e:
            logger.error(f"Error querying search index: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def hybrid_search(self, query: str, query_embeddings: List[float], keyword_count: int = 10, semantic_count: int = 20) -> Dict:
        """
        Perform hybrid search combining keyword (BM25) and semantic (vector) search.
        
        Args:
            query: The search query text
            query_embeddings: Vector embeddings for the query
            keyword_count: Number of documents to retrieve from keyword search (default: 10)
            semantic_count: Number of documents to retrieve from semantic search (default: 20)
            
        Returns:
            Dict with same format as query() method, containing interleaved results
        """
        try:
            if not query.strip():
                logger.error("Empty query text provided")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            if not query_embeddings:
                logger.error("Empty query embeddings provided")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            # Perform parallel searches
            keyword_results = self._keyword_search(query, keyword_count)
            semantic_results = self._semantic_search(query_embeddings, semantic_count)
            
            # Combine and interleave results based on relevance scores
            combined_results = self._interleave_results(keyword_results, semantic_results)
            
            # Format results to match ChromaDB format
            documents = []
            metadatas = []
            distances = []
            
            for result in combined_results:
                documents.append(result["content"])
                
                metadata = {
                    "page_id": result.get("page_id", ""),
                    "page_path": result.get("page_path", ""),
                    "parent_path": result.get("parent_path", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "token_count": result.get("token_count", 0),
                    "search_type": result.get("search_type", "unknown")  # Track search source
                }
                metadatas.append(metadata)
                distances.append(result.get("distance", 1.0))
            
            logger.info(f"Hybrid search returned {len(documents)} documents")
            return {
                "documents": [documents],
                "metadatas": [metadatas], 
                "distances": [distances]
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def hybrid_semantic_search(self, query: str, query_vector: List[float], top_k: int = 30, filter_expression: Optional[str] = None) -> Dict:
        """Hybrid search with semantic reranker, extractive captions and answers."""
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=min(top_k, 50),
                fields="content_vector"
            )

            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k,
                query_type="semantic",
                semantic_configuration_name="my-semantic-config",
                select=["id", "content", "page_id", "page_path", "parent_path", 
                        "chunk_index", "chunk_type", "start_line", "end_line", "token_count",
                        "image_url", "image_type", "mermaid_code"],
                filter=filter_expression
            )

            documents = []
            metadatas = []
            distances = []

            for result in results:
                documents.append(result["content"])
                meta = {
                    "page_id": result.get("page_id", ""),
                    "page_path": result.get("page_path", ""),
                    "parent_path": result.get("parent_path", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "token_count": result.get("token_count", 0),
                    "image_url": result.get("image_url", ""),
                    "image_type": result.get("image_type", "none"),
                    "mermaid_code": result.get("mermaid_code", ""),
                    "search_type": "hybrid_semantic"
                }
                # Attach first extractive caption if available
                captions = result.get("@search.captions")
                if captions and isinstance(captions, list) and len(captions) > 0:
                    meta["caption"] = captions[0].get("text", "")

                metadatas.append(meta)

                # Prefer reranker score when present (reranker range is 0-4, normalize to 0-1)
                reranker_score = result.get("@search.reranker_score")
                search_score = result.get("@search.score", 0.0)
                try:
                    if reranker_score is not None:
                        s = min(float(reranker_score) / 4.0, 1.0)
                    else:
                        s = min(float(search_score), 1.0)
                except Exception:
                    s = 0.0
                distance = 1.0 - max(s, 0.0)
                distances.append(distance)

            return {
                "documents": [documents],
                "metadatas": [metadatas], 
                "distances": [distances]
            }
        except Exception as e:
            logger.error(f"Error in hybrid_semantic_search: {str(e)}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def fetch_neighbor_chunks(self, page_path: str, chunk_index: int, window: int = 1) -> List[Dict]:
        """Fetch adjacent chunks for a given page and chunk index (±window)."""
        try:
            neighbors = []
            escaped_path = page_path.replace("'", "''")
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                target_index = chunk_index + offset
                if target_index < 0:
                    continue
                filter_expr = f"page_path eq '{escaped_path}' and chunk_index eq {target_index}"
                results = self.search_client.search(
                    search_text="*",
                    filter=filter_expr,
                    top=1,
                    select=["id", "content", "page_id", "page_path", "parent_path",
                            "chunk_index", "chunk_type", "start_line", "end_line", "token_count",
                            "image_url", "image_type", "mermaid_code"]
                )
                for result in results:
                    neighbors.append({
                        "content": result.get("content", ""),
                        "page_path": result.get("page_path", ""),
                        "chunk_index": result.get("chunk_index", 0),
                        "chunk_type": result.get("chunk_type", ""),
                        "token_count": result.get("token_count", 0),
                        "start_line": result.get("start_line", 0),
                        "end_line": result.get("end_line", 0),
                        "image_url": result.get("image_url", ""),
                        "image_type": result.get("image_type", "none"),
                        "mermaid_code": result.get("mermaid_code", ""),
                    })
            return neighbors
        except Exception as e:
            logger.error(f"Error fetching neighbor chunks: {str(e)}")
            return []

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform keyword search using BM25 on content field only."""
        try:
            # Search only the content field (not page_path due to searchability issues)
            results = self.search_client.search(
                search_text=query,
                top=top_k,
                search_fields=["content"],  # Content only - more reliable
                select=["id", "content", "page_id", "page_path", "parent_path", 
                       "chunk_index", "chunk_type", "start_line", "end_line", "token_count"]
            )
            
            keyword_docs = []
            for result in results:
                # Normalize BM25 score to 0-1 range
                bm25_score = result.get("@search.score", 0.0)
                normalized_score = min(bm25_score / 10.0, 1.0)  # BM25 scores can be > 1
                distance = 1.0 - normalized_score
                
                doc = {
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "page_id": result.get("page_id", ""),
                    "page_path": result.get("page_path", ""),
                    "parent_path": result.get("parent_path", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "token_count": result.get("token_count", 0),
                    "search_type": "keyword",
                    "raw_score": bm25_score,
                    "normalized_score": normalized_score,
                    "distance": distance
                }
                keyword_docs.append(doc)
            
            logger.info(f"Keyword search returned {len(keyword_docs)} documents")
            return keyword_docs
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def _semantic_search(self, query_embeddings: List[float], top_k: int) -> List[Dict]:
        """Perform semantic search using vector embeddings."""
        try:
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embeddings,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # Perform search
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                top=top_k,
                select=["id", "content", "page_id", "page_path", "parent_path", 
                       "chunk_index", "chunk_type", "start_line", "end_line", "token_count"]
            )
            
            semantic_docs = []
            for result in results:
                # Vector search scores are already normalized 0-1
                vector_score = result.get("@search.score", 0.0)
                distance = 1.0 - min(vector_score, 1.0)
                
                doc = {
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "page_id": result.get("page_id", ""),
                    "page_path": result.get("page_path", ""),
                    "parent_path": result.get("parent_path", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "chunk_type": result.get("chunk_type", ""),
                    "start_line": result.get("start_line", 0),
                    "end_line": result.get("end_line", 0),
                    "token_count": result.get("token_count", 0),
                    "search_type": "semantic",
                    "raw_score": vector_score,
                    "normalized_score": vector_score,
                    "distance": distance
                }
                semantic_docs.append(doc)
            
            logger.info(f"Semantic search returned {len(semantic_docs)} documents")
            return semantic_docs
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _interleave_results(self, keyword_docs: List[Dict], semantic_docs: List[Dict]) -> List[Dict]:
        """Interleave keyword and semantic results based on relevance scores."""
        try:
            # Remove duplicates - if same document appears in both, keep the one with better score
            all_docs = {}
            
            # Add keyword docs
            for doc in keyword_docs:
                doc_id = doc["id"]
                all_docs[doc_id] = doc
            
            # Add semantic docs, but keep better scoring version if duplicate
            for doc in semantic_docs:
                doc_id = doc["id"]
                if doc_id in all_docs:
                    # Keep the version with better normalized score
                    if doc["normalized_score"] > all_docs[doc_id]["normalized_score"]:
                        all_docs[doc_id] = doc
                    # Keep track that this was found in both searches
                    all_docs[doc_id]["search_type"] = "both"
                else:
                    all_docs[doc_id] = doc
            
            # Convert back to list and sort by normalized score (descending)
            combined_docs = list(all_docs.values())
            combined_docs.sort(key=lambda x: x["normalized_score"], reverse=True)
            
            # Limit to 30 documents maximum
            final_docs = combined_docs[:30]
            
            logger.info(f"Interleaved {len(keyword_docs)} keyword + {len(semantic_docs)} semantic = {len(final_docs)} final documents")
            
            # Log search type distribution
            search_types = {}
            for doc in final_docs:
                search_type = doc["search_type"]
                search_types[search_type] = search_types.get(search_type, 0) + 1
            
            logger.info(f"Search type distribution: {search_types}")
            
            return final_docs
            
        except Exception as e:
            logger.error(f"Error interleaving results: {str(e)}")
            # Fallback: just combine without deduplication
            return (keyword_docs + semantic_docs)[:30]
    
    def delete_by_ids(self, ids: List[str]):
        """Delete documents by IDs."""
        try:
            # Encode the IDs to match what's stored in Azure Search
            encoded_ids = [encode_document_id(doc_id) for doc_id in ids]
            documents_to_delete = [{"id": encoded_id} for encoded_id in encoded_ids]
            result = self.search_client.delete_documents(documents=documents_to_delete)
            
            for item in result:
                if not item.succeeded:
                    logger.error(f"Failed to delete document {item.key}: {item.error_message}")
            
            logger.info(f"Deleted {len(ids)} documents from index")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def count(self) -> int:
        """Get the total number of documents in the index."""
        try:
            result = self.search_client.search(
                search_text="*",
                include_total_count=True,
                top=0
            )
            return result.get_count()
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    def reset(self):
        """Delete all documents from the index."""
        try:
            # Get all document IDs
            results = self.search_client.search(
                search_text="*",
                select=["id"],
                top=10000  # Adjust based on your needs
            )
            
            ids_to_delete = [result["id"] for result in results]
            
            if ids_to_delete:
                self.delete_by_ids(ids_to_delete)
                logger.info(f"Reset index: deleted {len(ids_to_delete)} documents")
            else:
                logger.info("Index is already empty")
                
        except Exception as e:
            logger.error(f"Error resetting index: {str(e)}")
            raise
