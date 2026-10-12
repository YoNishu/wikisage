"""
Enhanced query interface for the ADO Wiki RAG system.
Provides intelligent question answering with context-aware retrieval and response generation.
"""

import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import openai
import tiktoken
import json
from utils import ConfigManager, logger
from azure_search_client import AzureSearchVectorStore

@dataclass
class QueryResult:
    """Represents a search result with metadata."""
    content: str
    page_path: str
    chunk_type: str
    distance: float
    chunk_index: int
    token_count: int
    # Optional extras for richer UI/diagnostics
    search_type: str = "hybrid"
    caption: str = ""
    # Image/mermaid fields
    image_url: str = ""
    image_type: str = "none"
    mermaid_code: str = ""

@dataclass
class QueryResponse:
    """Represents the complete response to a user query."""
    answer: str
    contexts_used: List[QueryResult]
    enhanced_query: str
    total_tokens_used: int

class EnhancedQueryProcessor:
    """Handles query processing, enhancement, and response generation."""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.ai_client = openai.AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=config["AZURE_OPENAI_ENDPOINT"],
            api_key=config["AZURE_OPENAI_KEY"],
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # Model-aware token budgeting (fallback to 128k if unknown)
        self.model_name = config["AZURE_OPENAI_CHAT_MODEL"]
        self.model_context_window = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4.1": 128000,
            "gpt-4-32k": 32768,
            "gpt-4": 8192
        }.get(self.model_name, 128000)
        self.answer_tokens_budget = 4096
        self.system_overhead_tokens = 1000
        self.max_context_tokens = max(
            2000, self.model_context_window - (self.answer_tokens_budget + self.system_overhead_tokens)
        )
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Initialize connection to vector store (Azure Search or ChromaDB)."""
        try:
            logger.info("Connecting to vector store...")
            
            # Check if we have Azure Search configuration
            search_service_name = self.config.get("SEARCH_SERVICE_NAME")
            search_service_key = self.config.get("SEARCH_SERVICE_KEY")
            
            if search_service_name and search_service_key:
                # Determine index name based on INDEX_MODE
                index_mode = self.config.get("INDEX_MODE", "text").strip().lower()
                if index_mode == "multimodal":
                    index_name = "ado_wiki_images"
                else:
                    index_name = self.config.get("INDEX_NAME", "ado_wiki")
                
                logger.info(f"Using Azure AI Search: {search_service_name}, index: {index_name}, mode: {index_mode}")
                self.vector_store = AzureSearchVectorStore(
                    service_name=search_service_name,
                    api_key=search_service_key,
                    index_name=index_name
                )
                self.use_azure_search = True
                
                # Get document count
                count = self.vector_store.count()
                logger.info(f"Azure Search index contains {count} chunks")
                
            else:
                # Fallback to local ChromaDB
                logger.info("Azure Search not configured, using local ChromaDB...")
                import chromadb
                
                self.chroma_client = chromadb.PersistentClient(path=self.config["CHROMA_DB_PATH"])
                
                try:
                    self.collection = self.chroma_client.get_collection("ado_wiki")
                    logger.info("Connected to existing ADO Wiki collection")
                    
                    # Get collection info
                    count = self.collection.count()
                    logger.info(f"ChromaDB collection contains {count} chunks")
                    
                except Exception as e:
                    logger.error(f"Error accessing ChromaDB collection: {str(e)}")
                    logger.error("Please run 'python ingest.py' first to create the knowledge base")
                    sys.exit(1)
                
                self.use_azure_search = False
                
        except Exception as e:
            logger.error(f"Error connecting to vector store: {str(e)}")
            sys.exit(1)
    
    def enhance_query(self, query: str) -> str:
        """Return the original query without enhancement (kept for UI display)."""
        logger.info(f"Using original query without enhancement: {query}")
        return query

    def expand_queries(self, query: str) -> List[str]:
        """
        Generate a small set of paraphrases to improve recall (multi-query).
        Returns [original, p1, p2] best-effort; falls back to [original] on error.
        """
        try:
            prompt = (
                "Rewrite the user query into 4 alternative phrasings that preserve meaning and key entities. "
                "Include at least one phrasing that uses different terminology or synonyms. "
                "Return a JSON array of 4 strings, no extra text."
            )
            resp = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=[
                    {"role": "system", "content": "You generate concise paraphrases for search recall. The queries are about an engineering team's internal documentation: architecture, operations, deployments, monitoring, and related DevOps topics."},
                    {"role": "user", "content": f"{prompt}\n\nUser query: {query}"}
                ],
                temperature=0.2,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            variants = []
            try:
                variants = json.loads(text)
                if not isinstance(variants, list):
                    variants = []
            except Exception:
                # If not valid JSON, attempt simple split
                variants = []
            expanded = [query]
            for v in variants:
                if isinstance(v, str):
                    v_clean = v.strip()
                    if v_clean and v_clean.lower() != query.lower() and v_clean not in expanded:
                        expanded.append(v_clean)
                if len(expanded) >= 5:
                    break
            logger.info(f"Multi-query expansions: {expanded}")
            return expanded or [query]
        except Exception as e:
            logger.warning(f"Query expansion failed, using original only: {str(e)}")
            return [query]
    
    def generate_hypothetical_answer(self, query: str) -> Optional[str]:
        """Generate a hypothetical answer (HyDE) to improve embedding-based retrieval.
        
        The embedding of a plausible answer is semantically closer to actual
        documentation content than the embedding of a question.
        """
        try:
            resp = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=[
                    {"role": "system", "content": (
                        "You are a technical documentation writer for an engineering team's wiki covering "
                        "architecture, networking, operations, testing, and DevOps practices. Given a question, write "
                        "a short paragraph (3-5 sentences) that would be a plausible answer found in the team's "
                        "internal wiki documentation. Do not add caveats, disclaimers, or preamble. "
                        "Write as if this text exists in the wiki. Use precise technical terminology such as "
                        "service and component names, configuration settings, runbook steps, alerts, "
                        "and feature flags where relevant."
                    )},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=300,
            )
            hyde_text = resp.choices[0].message.content.strip()
            logger.info(f"HyDE generated ({len(hyde_text)} chars): {hyde_text[:100]}...")
            return hyde_text
        except Exception as e:
            logger.warning(f"HyDE generation failed: {str(e)}")
            return None

    def llm_rerank(self, query: str, results: List[QueryResult], top_n: int = 15) -> List[QueryResult]:
        """Use LLM to rerank search results by relevance to the query."""
        if not results or len(results) <= 1:
            return results

        # Take top candidates for reranking (limit prompt size)
        candidates = results[:min(len(results), 25)]

        # Build numbered list of chunk previews
        chunks_text = ""
        for i, r in enumerate(candidates):
            preview = r.content[:400].replace('\n', ' ').strip()
            chunks_text += f"[{i}] (page: {r.page_path})\n{preview}\n\n"

        try:
            resp = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=[
                    {"role": "system", "content": (
                        "You are a search relevance expert for an engineering team's wiki. "
                        "Given a user query and numbered document chunks from the wiki, "
                        "score each chunk's relevance to answering the query on a scale of 1-10 "
                        "(10 = perfectly relevant). Return a JSON array of objects: "
                        '[{"index": <int>, "score": <float>}, ...]. '
                        "Include ALL chunks. Return ONLY the JSON array, no other text."
                    )},
                    {"role": "user", "content": f"Query: {query}\n\nDocument chunks:\n{chunks_text}"}
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content.strip()
            # Handle possible markdown code-fence wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            scores = json.loads(text)

            if not isinstance(scores, list):
                logger.warning("LLM rerank did not return a list, falling back")
                return candidates[:top_n]

            # Build reranked list sorted by LLM score descending
            scored_results = []
            for item in scores:
                idx = item.get("index", -1)
                score = float(item.get("score", 0))
                if 0 <= idx < len(candidates) and score >= 3:
                    r = candidates[idx]
                    r.distance = 1.0 - (score / 10.0)  # normalize to distance
                    scored_results.append(r)

            scored_results.sort(key=lambda r: r.distance)
            logger.info(f"LLM reranking: kept {len(scored_results)} (score >= 3) from {len(candidates)} candidates")
            return scored_results[:top_n]

        except Exception as e:
            logger.warning(f"LLM reranking failed, using original order: {str(e)}")
            return candidates[:top_n]

    def stitch_neighbor_chunks(self, contexts: List[QueryResult]) -> List[QueryResult]:
        """Fetch and stitch adjacent chunks (+-1) for richer context around each selected chunk."""
        if not self.use_azure_search:
            return contexts

        stitched = []
        seen_keys = set()

        for ctx in contexts:
            key = f"{ctx.page_path}|{ctx.chunk_index}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Fetch neighbor chunks from the index
            neighbors = self.vector_store.fetch_neighbor_chunks(
                page_path=ctx.page_path,
                chunk_index=ctx.chunk_index,
                window=1
            )

            # Assemble: previous + current + next
            before_parts = []
            after_parts = []
            for n in neighbors:
                nkey = f"{n['page_path']}|{n['chunk_index']}"
                seen_keys.add(nkey)  # avoid duplicating if neighbor was also selected
                if n["chunk_index"] < ctx.chunk_index:
                    before_parts.append((n["chunk_index"], n["content"]))
                elif n["chunk_index"] > ctx.chunk_index:
                    after_parts.append((n["chunk_index"], n["content"]))

            before_parts.sort(key=lambda x: x[0])
            after_parts.sort(key=lambda x: x[0])

            parts = [t for _, t in before_parts] + [ctx.content] + [t for _, t in after_parts]
            stitched_content = "\n\n".join(parts)
            stitched_tokens = len(self.tokenizer.encode(stitched_content))

            stitched.append(QueryResult(
                content=stitched_content,
                page_path=ctx.page_path,
                chunk_type=ctx.chunk_type,
                distance=ctx.distance,
                chunk_index=ctx.chunk_index,
                token_count=stitched_tokens,
                search_type=ctx.search_type,
                caption=ctx.caption,
                image_url=ctx.image_url,
                image_type=ctx.image_type,
                mermaid_code=ctx.mermaid_code
            ))

        logger.info(f"Stitched {len(stitched)} contexts with neighbor chunks")
        return stitched

    def search_knowledge_base(self, query: str, enhanced_query: str, top_k: int = 30, keywords: Optional[List[str]] = None) -> List[QueryResult]:
        """Search KB using multi-query + keyword boost + Azure hybrid semantic reranker + RRF fusion."""
        try:
            # Multi-query expansions
            expansions = self.expand_queries(enhanced_query)

            # HyDE: generate hypothetical answer and use its embedding for retrieval
            hyde_answer = self.generate_hypothetical_answer(query)
            if hyde_answer:
                expansions.append(hyde_answer)

            # Add a keyword-focused query if keywords were extracted
            if keywords:
                keyword_query = " ".join(keywords)
                if keyword_query not in expansions:
                    expansions.append(keyword_query)

            if self.use_azure_search:
                # Embed each expansion and run Azure hybrid semantic search
                per_query_results = []
                for q in expansions:
                    emb = self.ai_client.embeddings.create(
                        model=self.config["AZURE_OPENAI_EMBED_MODEL"],
                        input=q
                    )
                    q_vec = emb.data[0].embedding
                    res = self.vector_store.hybrid_semantic_search(
                        query=q,
                        query_vector=q_vec,
                        top_k=top_k
                    )
                    per_query_results.append(res)

                # RRF fuse across expansions
                k_rrf = 60.0
                score_map: Dict[str, float] = {}
                best_item: Dict[str, Dict[str, Any]] = {}

                for res in per_query_results:
                    docs = res.get("documents", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]
                    dists = res.get("distances", [[]])[0]
                    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
                        key = f"{meta.get('page_path','unknown')}|{meta.get('chunk_index',0)}"
                        score_map[key] = score_map.get(key, 0.0) + 1.0 / (k_rrf + rank)
                        # Keep the best (lowest) distance and associated fields
                        if (key not in best_item) or (dist < best_item[key]["distance"]):
                            best_item[key] = {
                                "content": doc,
                                "metadata": meta,
                                "distance": dist,
                            }

                # Sort by fused score desc
                ranked_keys = sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)
                fused: List[QueryResult] = []
                for key in ranked_keys:
                    item = best_item[key]
                    meta = item["metadata"]
                    content = item["content"]
                    dist = item["distance"]
                    fused.append(
                        QueryResult(
                            content=content.strip(),
                            page_path=meta.get("page_path", "unknown"),
                            chunk_type=meta.get("chunk_type", "unknown"),
                            distance=dist,
                            chunk_index=meta.get("chunk_index", 0),
                            token_count=meta.get("token_count", len(self.tokenizer.encode(content))),
                            search_type=meta.get("search_type", "hybrid_semantic"),
                            caption=meta.get("caption", ""),
                            image_url=meta.get("image_url", ""),
                            image_type=meta.get("image_type", "none"),
                            mermaid_code=meta.get("mermaid_code", "")
                        )
                    )

                # Filter by distance threshold (relaxed — LLM reranking refines later)
                filtered = [r for r in fused if r.distance < 0.8]
                logger.info(f"Azure hybrid+RRF results: {len(filtered)} kept from {len(fused)} fused")
                return filtered

            else:
                # Chroma fallback (single semantic search)
                emb = self.ai_client.embeddings.create(
                    model=self.config["AZURE_OPENAI_EMBED_MODEL"],
                    input=enhanced_query
                )
                q_vec = emb.data[0].embedding
                res = self.collection.query(
                    query_embeddings=[q_vec],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]

                out: List[QueryResult] = []
                for doc, meta, dist in zip(docs, metas, dists):
                    out.append(
                        QueryResult(
                            content=doc.strip(),
                            page_path=meta.get("page_path", "unknown"),
                            chunk_type=meta.get("chunk_type", "unknown"),
                            distance=dist,
                            chunk_index=meta.get("chunk_index", 0),
                            token_count=meta.get("token_count", len(self.tokenizer.encode(doc))),
                            search_type="semantic",
                            image_url=meta.get("image_url", ""),
                            image_type=meta.get("image_type", "none"),
                            mermaid_code=meta.get("mermaid_code", "")
                        )
                    )
                filtered = [r for r in out if r.distance < 0.8]
                logger.info(f"Chroma semantic results: {len(filtered)} kept from {len(out)}")
                return filtered

        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def select_best_contexts(self, results: List[QueryResult]) -> List[QueryResult]:
        """Select contexts with diversity and within adaptive token budget."""
        if not results:
            return []

        # Sort by relevance (lower distance = more relevant)
        results_sorted = sorted(results, key=lambda r: r.distance)

        # Per-page cap to encourage diversity
        per_page_cap = 4
        per_page_counts: Dict[str, int] = {}

        selected: List[QueryResult] = []
        total_tokens = 0

        for r in results_sorted:
            # Enforce per-page limit
            cnt = per_page_counts.get(r.page_path, 0)
            if cnt >= per_page_cap:
                continue

            # Check token budget
            if total_tokens + r.token_count > self.max_context_tokens:
                continue

            selected.append(r)
            per_page_counts[r.page_path] = cnt + 1
            total_tokens += r.token_count

        logger.info(f"Selected {len(selected)} contexts using {total_tokens} tokens (budget {self.max_context_tokens})")
        return selected
    
    def generate_response(self, query: str, enhanced_query: str, contexts: List[QueryResult], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a response based on the query, contexts, and optional chat history."""
        if not contexts:
            return "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or check if the content has been ingested properly."
        
        # Build context string with source attribution
        context_parts = []
        for i, context in enumerate(contexts, 1):
            context_header = f"[Source {i}: {context.page_path} - {context.chunk_type} chunk]"
            context_parts.append(f"{context_header}\n{context.content}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Create system message
        system_message = {
            "role": "system",
            "content": """You are the ADO Wiki Agent — an expert assistant that answers questions about an engineering team's systems, operations, and DevOps practices using the team's Azure DevOps wiki.

Your task is to provide the most accurate, comprehensive, and actionable answer possible based SOLELY on the provided context documents from the team's wiki.

**Response Protocol:**
1. THINK step-by-step: First internally reason about which sources are most relevant and how they connect.
2. DIRECT ANSWER: Lead with a clear, concise answer to the question.
3. DETAILED EXPLANATION: Provide supporting details, procedures, configurations, or examples drawn from the sources.
4. STEP-BY-STEP INSTRUCTIONS: If the question involves a process, provide numbered steps.
5. INLINE CITATIONS: Reference sources inline like [Source 1], [Source 3] to support each claim.
6. GAPS: If the provided context does not fully answer the question, explicitly state what information is missing.

**Formatting:**
- Use markdown headers, bullet points, code blocks, and tables where appropriate.
- For commands or configurations, always use fenced code blocks with the language specified.
- Bold key terms and important warnings.

**Constraints:**
- NEVER fabricate information not present in the sources.
- If sources conflict, acknowledge the discrepancy and present both views.
- Prefer specificity over generality — include exact values, paths, commands, and names from the sources.
- Use the exact terminology, component names, and configuration names that appear in the sources.
- If the user asks a follow-up question, use the conversation history to understand context and provide continuity."""
        }
        
        # Build messages list with chat history
        messages = [system_message]
        
        if chat_history:
            # Include prior conversation turns (limit to last 10 to manage tokens)
            history_tokens = 0
            trimmed_history = []
            for msg in reversed(chat_history[-10:]):
                msg_tokens = self.count_tokens(msg.get("content", ""))
                if history_tokens + msg_tokens > 8000:  # cap history at 8k tokens
                    break
                trimmed_history.insert(0, msg)
                history_tokens += msg_tokens
            messages.extend(trimmed_history)
        
        # Add current turn with context
        messages.append({
            "role": "user",
            "content": f"""Context Information (retrieved from wiki):
{context_text}

Question: {query}

Using the context above, provide a thorough, well-structured answer with inline source citations [Source N]."""
        })
        
        try:
            response = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=messages,
                temperature=0.0,
                max_tokens=4096
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Sorry, I encountered an error while generating the response: {str(e)}"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def process_query(self, query: str) -> QueryResponse:
        """Process a single query (no chat history). Kept for backward compatibility."""
        return self.process_chat_query(query, chat_history=None)

    def contextualize_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Rewrite a follow-up question into a standalone query using chat history.
        
        Handles cases like:
          User: 'what is the deployment gate?' -> 'what is the deployment gate'
          User: 'how is it configured?'      -> 'how is the deployment gate configured'
        """
        if not chat_history:
            return query
        try:
            # Build recent history summary (last 4 turns = 8 messages)
            recent = chat_history[-8:]
            history_text = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:500]}"
                for m in recent
            )
            resp = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=[
                    {"role": "system", "content": (
                        "You are a search query rewriter for an internal engineering wiki search engine.\n"
                        "Given the conversation history and a follow-up question, rewrite the follow-up into a "
                        "STANDALONE search query that:\n"
                        "1. Replaces ALL pronouns (it, they, this, that, these, those) with the actual entities from conversation history.\n"
                        "2. Includes the key topic/entity being discussed (e.g. a service, component, tool, or process name).\n"
                        "3. Adds relevant context terms from the conversation when the query is vague.\n"
                        "4. Is phrased as a good search query (not a conversational sentence).\n"
                        "5. Preserves the user's INTENT (asking how, what, why, troubleshooting, etc.).\n\n"
                        "Examples:\n"
                        "  History: User asked 'what is the deployment gate?' -> Follow-up: 'how is it configured?' -> Rewrite: 'how is the deployment gate configured'\n"
                        "  History: User asked about the ingestion service rollout -> Follow-up: 'what are the steps?' -> Rewrite: 'ingestion service rollout steps and process'\n"
                        "  History: User asked about compliance exceptions -> Follow-up: 'how to create one?' -> Rewrite: 'how to create a compliance exception'\n\n"
                        "Return ONLY the rewritten query, nothing else."
                    )},
                    {"role": "user", "content": f"Conversation:\n{history_text}\n\nFollow-up question: {query}"}
                ],
                temperature=0.0,
                max_tokens=200,
            )
            standalone = resp.choices[0].message.content.strip().strip('"').strip("'")
            logger.info(f"Contextualized query: '{query}' -> '{standalone}'")
            return standalone
        except Exception as e:
            logger.warning(f"Query contextualization failed: {str(e)}")
            return query

    def extract_search_keywords(self, query: str) -> List[str]:
        """Extract important keywords/entities from a query for boosted keyword (BM25) search.
        
        Returns a list of 3-6 key terms that should be used for keyword-only search.
        """
        try:
            resp = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=[
                    {"role": "system", "content": (
                        "Extract the 3-6 most important search keywords from the query. "
                        "Focus on: proper nouns, technical terms, product names, acronyms, and action words. "
                        "Include exact service, component, and tool names, plus acronyms, as written in the query. "
                        "Return a JSON array of strings. No explanation."
                    )},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=100,
            )
            text = resp.choices[0].message.content.strip()
            keywords = json.loads(text)
            if isinstance(keywords, list):
                keywords = [k for k in keywords if isinstance(k, str) and k.strip()]
                logger.info(f"Extracted keywords: {keywords}")
                return keywords[:6]
        except Exception:
            pass
        # Fallback: simple word extraction
        stop_words = {'how', 'what', 'is', 'are', 'the', 'a', 'an', 'in', 'to', 'for', 'of', 'and', 'or', 'it', 'do', 'does', 'can', 'this', 'that', 'with'}
        return [w for w in query.split() if w.lower() not in stop_words and len(w) > 2][:6]

    def generate_refined_query(self, original_query: str, first_pass_chunks: List['QueryResult']) -> str:
        """After the first retrieval pass, analyze retrieved chunks and generate a refined
        follow-up query to capture information that may have been missed."""
        if not first_pass_chunks:
            return original_query
        try:
            # Summarize what was found in first pass
            chunk_summaries = []
            for i, c in enumerate(first_pass_chunks[:8]):
                preview = c.content[:250].replace('\n', ' ').strip()
                chunk_summaries.append(f"[{i+1}] ({c.page_path}): {preview}")
            summaries_text = "\n".join(chunk_summaries)

            resp = self.ai_client.chat.completions.create(
                model=self.config["AZURE_OPENAI_CHAT_MODEL"],
                messages=[
                    {"role": "system", "content": (
                        "You are a search query refinement expert for an internal engineering wiki.\n"
                        "Given the user's original query and the document chunks retrieved in the first search pass, "
                        "generate a DIFFERENT follow-up search query that will retrieve ADDITIONAL relevant information "
                        "that was NOT covered by the first retrieval.\n\n"
                        "Strategy:\n"
                        "- Look at what aspects of the original question are NOT yet answered by the retrieved chunks.\n"
                        "- Use different terminology, synonyms, or related concepts.\n"
                        "- If the first pass found high-level docs, the refined query should target specific details, procedures, or architecture.\n"
                        "- If the first pass found implementation details, the refined query should target overview or context docs.\n"
                        "- Include specific terms discovered in the chunks (e.g., component names, config names).\n\n"
                        "Return ONLY the refined search query, nothing else."
                    )},
                    {"role": "user", "content": f"Original query: {original_query}\n\nFirst-pass retrieved chunks:\n{summaries_text}"}
                ],
                temperature=0.3,
                max_tokens=200,
            )
            refined = resp.choices[0].message.content.strip().strip('"').strip("'")
            logger.info(f"Refined second-pass query: '{original_query}' -> '{refined}'")
            return refined
        except Exception as e:
            logger.warning(f"Query refinement failed: {str(e)}")
            return original_query

    def _merge_results(self, first: List[QueryResult], second: List[QueryResult]) -> List[QueryResult]:
        """Merge two result lists, deduplicating by page_path|chunk_index key."""
        seen = set()
        merged = []
        for r in first + second:
            key = f"{r.page_path}|{r.chunk_index}"
            if key not in seen:
                seen.add(key)
                merged.append(r)
        return merged

    def process_chat_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> QueryResponse:
        """Process a query with full chat history for multi-turn conversations.
        
        Uses a TWO-PASS retrieval strategy:
          Pass 1: Search with the user's query (contextualized if chat history exists)
          Pass 2: LLM reviews first-pass chunks, generates a refined query, retrieves more
          Final: Merge, rerank, and select at least 10 chunks for the answer.
        """
        logger.info(f"Processing chat query: {query}")
        
        # ── Step 1: Contextualize query using chat history ──
        search_query = query
        if chat_history and len(chat_history) >= 2:
            search_query = self.contextualize_query(query, chat_history)
        
        enhanced_query = self.enhance_query(search_query)
        
        # ── Step 2: Extract keywords for boosted keyword search ──
        keywords = self.extract_search_keywords(search_query)
        
        # ── Step 3: FIRST PASS retrieval ──
        logger.info("=== PASS 1: Initial retrieval ===")
        first_pass_results = self.search_knowledge_base(search_query, enhanced_query, keywords=keywords)
        
        if not first_pass_results:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question. Please try rephrasing it or check if the content has been ingested.",
                contexts_used=[],
                enhanced_query=enhanced_query,
                total_tokens_used=0
            )
        
        # LLM rerank first-pass results
        first_pass_reranked = self.llm_rerank(search_query, first_pass_results, top_n=15)
        
        # ── Step 4: SECOND PASS - Refined retrieval ──
        logger.info("=== PASS 2: Refined retrieval ===")
        refined_query = self.generate_refined_query(search_query, first_pass_reranked)
        refined_keywords = self.extract_search_keywords(refined_query)
        
        second_pass_results = self.search_knowledge_base(
            refined_query, refined_query, top_k=20, keywords=refined_keywords
        )
        
        # LLM rerank second-pass results
        second_pass_reranked = self.llm_rerank(refined_query, second_pass_results, top_n=10) if second_pass_results else []
        
        # ── Step 5: Merge and deduplicate both passes ──
        merged_results = self._merge_results(first_pass_reranked, second_pass_reranked)
        logger.info(f"Merged results: {len(first_pass_reranked)} (pass1) + {len(second_pass_reranked)} (pass2) = {len(merged_results)} unique")
        
        # ── Step 6: Final rerank on merged set ──
        final_reranked = self.llm_rerank(search_query, merged_results, top_n=20)
        
        # ── Step 7: Select contexts ensuring minimum 10 chunks ──
        selected_contexts = self.select_best_contexts(final_reranked)
        
        # Ensure at least 10 chunks if available
        if len(selected_contexts) < 10 and len(final_reranked) > len(selected_contexts):
            # Add more chunks from reranked results that weren't selected
            selected_keys = {f"{c.page_path}|{c.chunk_index}" for c in selected_contexts}
            for r in final_reranked:
                if len(selected_contexts) >= 10:
                    break
                key = f"{r.page_path}|{r.chunk_index}"
                if key not in selected_keys:
                    selected_contexts.append(r)
                    selected_keys.add(key)
            logger.info(f"Padded to {len(selected_contexts)} contexts (minimum 10 target)")
        
        # ── Step 8: Stitch neighbor chunks for fuller context ──
        stitched_contexts = self.stitch_neighbor_chunks(selected_contexts)
        
        # ── Step 9: Generate response with chat history for continuity ──
        answer = self.generate_response(query, enhanced_query, stitched_contexts, chat_history=chat_history)
        
        # Calculate total tokens used
        total_tokens = sum(context.token_count for context in stitched_contexts)
        total_tokens += self.count_tokens(answer)
        
        return QueryResponse(
            answer=answer,
            contexts_used=stitched_contexts,
            enhanced_query=enhanced_query,
            total_tokens_used=total_tokens
        )

class InteractiveQueryInterface:
    """Interactive command-line interface for querying the knowledge base."""
    
    def __init__(self):
        self.processor = None
        self.setup()
    
    def setup(self):
        """Initialize the query processor."""
        try:
            config = ConfigManager.load_config()
            self.processor = EnhancedQueryProcessor(config)
            logger.info("Query interface ready!")
        except Exception as e:
            logger.error(f"Failed to initialize query interface: {str(e)}")
            sys.exit(1)
    
    def display_contexts(self, contexts: List[QueryResult]):
        """Display the contexts used for the response."""
        if not contexts:
            print("\nNo relevant contexts found.")
            return
        
        print(f"\n{'='*60}")
        print("RELEVANT CONTEXTS")
        print('='*60)
        
        for i, context in enumerate(contexts, 1):
            print(f"\n=== Context {i} ===")
            print(f"Source: {context.page_path}")
            print(f"Type: {context.chunk_type} | Relevance: {1-context.distance:.3f} | Tokens: {context.token_count}")
            print("-" * 50)
            print(context.content[:500] + ("..." if len(context.content) > 500 else ""))
            print("-" * 50)
    
    def run(self):
        """Run the interactive query interface."""
        print("=" * 60)
        print("ADO WIKI KNOWLEDGE BASE - INTERACTIVE QUERY")
        print("=" * 60)
        print("Ask questions about your Azure DevOps wiki content!")
        print("Type 'quit' or 'exit' to stop, 'help' for guidance.\n")
        
        while True:
            try:
                query = input("\n🔍 Enter your question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break
                
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                print("\n⏳ Processing your question...")
                
                # Process the query
                response = self.processor.process_query(query)
                
                # Display results - skip enhanced query display since enhancement is disabled
                # if response.enhanced_query != query:
                #     print(f"\n📝 Enhanced query: {response.enhanced_query}")
                
                self.display_contexts(response.contexts_used)
                
                print(f"\n{'='*60}")
                print("ANSWER")
                print('='*60)
                print(f"\n{response.answer}")
                
                print(f"\n{'='*60}")
                print(f"📊 Used {len(response.contexts_used)} contexts | {response.total_tokens_used:,} tokens")
                print('='*60)
                
            except KeyboardInterrupt:
                print("\n\nProcess interrupted. Goodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"\n❌ Sorry, an error occurred: {str(e)}")
    
    def show_help(self):
        """Show help information."""
        help_text = """
🆘 HELP - How to use the ADO Wiki Query System:

📝 QUERY EXAMPLES:
• "How do I set up performance testing with K6?"
• "What are the service configuration steps?"
• "Where can I find troubleshooting guides for latency issues?"
• "What is the compliance exception process?"
• "How do I configure Prometheus monitoring?"

💡 TIPS:
• Be specific in your questions
• Use full terms or the abbreviations your team uses
• Ask about procedures, configuration, or troubleshooting
• Reference specific tools or components when relevant

⚙️ COMMANDS:
• Type your question and press Enter
• 'help' - Show this help message
• 'quit', 'exit', or 'q' - Exit the program
• Ctrl+C - Force exit

🔍 SEARCH FEATURES:
• Semantic search finds contextually relevant content
• Uses your exact query terms for precise matching
• Multiple contexts provide comprehensive answers
• Source attribution shows where information comes from
        """
        print(help_text)

def main():
    """Main entry point for the query interface."""
    interface = InteractiveQueryInterface()
    interface.run()

if __name__ == "__main__":
    main()
