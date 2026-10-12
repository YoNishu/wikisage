# Architecture

## System Overview

Two-plane design: **offline ingestion** (local) and **online query** (Azure App Service).

```
        ┌──────────────────────────┐
        │   Azure DevOps Wiki      │
        └─────────────┬────────────┘
                      │ REST API + PAT
                ┌─────▼──────────┐
                │  Ingestion     │  Local machine
                │  ingest.py     │
                ├────────────────┤
                │ Fetch pages    │
                │ Chunk (800 tok)│
                │ Enrich context │
                │ Embed (3072-d) │
                │ Delta upsert   │
                └──────┬─────────┘
                       │
        ┌──────────────▼─────────────────────────┐
        │ Azure AI Search (Index: ado_wiki)       │
        │  - HNSW vector (3072-d, cosine)         │
        │  - BM25 keyword search                  │
        │  - Semantic reranker                    │
        └──────────────┬─────────────────────────┘
                       │
            ┌──────────▼───────────┐
            │ Query Plane          │  Azure App Service
            │ streamlit_app.py     │
            ├──────────────────────┤
            │ Chat history         │
            │ Query contextualize  │
            │ Multi-query expand   │
            │ HyDE generation      │
            │ Hybrid search × N    │
            │ RRF fusion           │
            │ LLM reranking        │
            │ Neighbor stitching   │
            │ Answer generation    │
            └──────────┬───────────┘
                       │
                ┌──────▼──────┐
                │  Chat UI    │
                └─────────────┘
```

## Components

| Component | File | Role |
|-----------|------|------|
| Config | `utils.py` (`ConfigManager`) | Load env vars or Key Vault secrets |
| Wiki Fetcher | `utils.py` (`WikiContentFetcher`) | Recursive ADO Wiki page retrieval |
| Chunker | `utils.py` (`SmartContentChunker`) | Structure-aware markdown chunking (800 tok, 200 overlap) |
| Embedder | `utils.py` (`EmbeddingGenerator`) | Batched Azure OpenAI embedding (text-embedding-3-large) |
| Search Client | `azure_search_client.py` | Index CRUD, hybrid + semantic search, neighbor fetch |
| Ingestor | `ingest.py` (`RAGIngestor`) | Delta detection, contextual enrichment, upsert orchestration |
| Query Processor | `query.py` (`EnhancedQueryProcessor`) | Full query pipeline with chat history |
| Chat UI | `streamlit_app.py` | Multi-turn Streamlit chat with session metrics |

## Ingestion Pipeline

1. Load config from env or Key Vault
2. Fetch full wiki tree recursively via ADO REST API
3. For each page (batched, 25/batch):
   - Clean markdown, parse structure (headers, code, lists)
   - Chunk at 800 tokens with 200-token overlap
   - Compute per-chunk MD5 hash; compare vs saved state → skip unchanged
   - **Contextual enrichment**: prepend `Page: /path > Section: title` to each chunk before embedding
   - Embed changed chunks with `text-embedding-3-large` (3072-dim, batch size 10)
   - Upsert to Azure AI Search; delete stale chunks if page shrank
4. Save `ingestion_state.json` after each batch

## Query Pipeline

1. **Chat contextualization**: if follow-up, rewrite query as standalone using LLM + conversation history
2. **Multi-query expansion**: generate 4 paraphrases (total 5 variants)
3. **HyDE**: generate a hypothetical answer paragraph; add its embedding as 6th search vector
4. **Hybrid semantic search** (per expansion): BM25 + vector + semantic reranker via Azure AI Search
5. **RRF fusion**: Reciprocal Rank Fusion across all 6 result sets (k=60)
6. **Distance filter**: keep results with distance < 0.8
7. **LLM reranking**: GPT-4.1 scores top 25 candidates 1-10; keep score ≥ 3
8. **Context selection**: sort by distance, per-page cap (4), token budget (dynamic based on model window)
9. **Neighbor stitching**: fetch ±1 adjacent chunks from the index for each selected chunk
10. **Answer generation**: GPT-4.1 with system prompt (chain-of-thought + inline citations), chat history (up to 8K tokens), temperature 0, max 4096 tokens

## Index Schema

```
Index: ado_wiki
  id              (key, base64-encoded)
  content         (searchable)
  content_vector  (3072-dim, HNSW, cosine)
  page_id, page_path, parent_path
  chunk_index, chunk_type
  start_line, end_line, token_count
  section_title, headers_path
  last_modified, tags
  Semantic config: my-semantic-config (content field)
```

## Security

- Secrets: loaded from env vars (local) or Key Vault (production)
- PAT: Basic auth over HTTPS only
- Keys: masked in logs
- IDs: base64-safe encoding prevents path injection

## Performance & Cost

| Mechanism | Benefit |
|-----------|---------|
| Delta ingestion | Skip unchanged pages/chunks |
| Contextual enrichment | Better retrieval without extra API calls |
| Batch embedding (10) | Throughput vs rate limits |
| Multi-query (≤6 with HyDE) | Recall improvement |
| LLM rerank | Precision filter before expensive generation |
| Per-page cap (4) | Context diversity |
| Token budgeting | Prevents overflow |
| Distance threshold (0.8) | Filters weak matches |

## Local vs Production

| Aspect | Local | Production |
|--------|-------|------------|
| Vector store | ChromaDB fallback | Azure AI Search |
| Config source | `.env` file | App Settings (set `ENVIRONMENT=local`) or Key Vault (`ENVIRONMENT=production`) |
| Python | 3.12+ | App Service Python 3.12 |
| Deploy | `streamlit run` | `startup.sh` via App Service |
