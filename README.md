# ADO Wiki RAG Agent — Technical Deep Dive

A production-grade Retrieval-Augmented Generation (RAG) system that ingests Azure DevOps wiki pages and provides an intelligent conversational interface for querying organizational knowledge. Built with a **multi-stage advanced query pipeline**, **hybrid search**, and **multi-strategy retrieval** to maximize answer quality.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Query Pipeline — All Stages](#query-pipeline--all-stages)
- [RAG Optimizations & Techniques](#rag-optimizations--techniques)
- [Conversational Memory](#conversational-memory)
- [UI & Observability](#ui--observability)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User (Streamlit Chat UI)                     │
│   "What does GW_BE_Request latency mean?"                          │
│   → "How is it different from the backend metric?"  (follow-up)    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Multi-Stage Query Pipeline                     │
│                                                                     │
│  Contextualize → Enhance → Multi-Query Expand (5x) → HyDE          │
│       → Parallel Hybrid Search (6 runs) → RRF Fusion                │
│       → LLM Reranking → Neighbor Stitching → Chat-Aware LLM Answer │
└─────────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐ ┌────────────┐ ┌─────────────┐
     │ Azure AI    │ │ Azure      │ │ Azure       │
     │ Search      │ │ OpenAI     │ │ DevOps      │
     │ (Hybrid     │ │ (GPT-4.1 + │ │ (Wiki API)  │
     │  Index)     │ │  Embed-3L) │ │             │
     └─────────────┘ └────────────┘ └─────────────┘
```

**Key Numbers:**

| Metric | Value |
|--------|-------|
| Embedding dimensions | 3072 (`text-embedding-3-large`) |
| Chunk size | 800 tokens with 200 token overlap |
| Search queries per user question | 6 (1 original + 4 expansions + 1 HyDE) |
| Search type | Hybrid (BM25 + vector + semantic reranker) |
| LLM reranking | Top 25 candidates scored 1–10 |
| Chat history budget | ~8,000 tokens of prior conversation |
| RRF fusion constant (k) | 60 |
| Answer generation | GPT-4.1, temperature 0, max 4096 tokens |

---

## Architecture

### Component Map

| Component | File | Technology | Purpose |
|-----------|------|-----------|---------|
| **Chat UI** | `streamlit_app.py` | Streamlit `st.chat_message` | Multi-turn conversational interface with session metrics |
| **Query Engine** | `query.py` | Python + Azure OpenAI | Full pipeline: contextualize → expand → search → rerank → generate |
| **Search Client** | `azure_search_client.py` | `azure-search-documents` SDK | Index management, hybrid queries, neighbor chunk fetch |
| **Ingestor** | `ingest.py` | Python | Crawl wiki, chunk, enrich, embed, delta upsert |
| **Utilities** | `utils.py` | Config, chunker, embedder | Config loader (env/Key Vault), wiki fetcher, smart chunker, batch embedder |
| **Startup** | `startup.sh` | Bash | Azure App Service bootstrap (pip install + Streamlit launch) |
| **Deploy** | `deploy.ps1` | PowerShell | One-command: fix LF → zip → deploy → monitor |

### Repo Layout

```
wikiagent/
├── streamlit_app.py          # Chat UI (entry point on Azure)
├── query.py                  # Query pipeline (EnhancedQueryProcessor)
├── azure_search_client.py    # Azure AI Search wrapper
├── ingest.py                 # Ingestion orchestrator (RAGIngestor)
├── utils.py                  # Config, fetcher, chunker, embedder
├── startup.sh                # Azure App Service startup
├── deploy.ps1                # Windows deploy script
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python 3.12
├── .env.example              # Environment variable template
├── .gitignore
├── ARCHITECTURE.md           # Architecture reference
├── DEPLOYMENT.md             # Deployment guide
└── README.md                 # This file
```

---

## Ingestion Pipeline

### How Ingestion Works

```
ADO Wiki REST API
    │
    ▼
┌─────────────────────────────────┐
│ 1. FETCH: Recursive page crawl  │  GET /wikis/{id}/pages?recursionLevel=Full
│    (PAT auth, retry + backoff)  │
├─────────────────────────────────┤
│ 2. CLEAN: Strip images, HTML    │  Preserve headers, code blocks, lists
│    Normalize whitespace          │
├─────────────────────────────────┤
│ 3. CHUNK: Structure-aware split  │  800 tokens, 200 overlap
│    Respect headers/code/list     │  Semantic boundaries preserved
│    boundaries                    │
├─────────────────────────────────┤
│ 4. DELTA DETECT: Per-chunk MD5   │  Compare vs ingestion_state.json
│    Skip unchanged chunks         │  Only re-embed what changed
├─────────────────────────────────┤
│ 5. ENRICH: Prepend context       │  "Page: /path > Section: title"
│    to each chunk before embed    │  Gives embeddings richer context
├─────────────────────────────────┤
│ 6. EMBED: text-embedding-3-large │  3072 dimensions, batch size 10
│    via Azure OpenAI              │  Rate-limited with 0.5s sleep
├─────────────────────────────────┤
│ 7. UPSERT: Azure AI Search      │  Base64-safe document IDs
│    Delete stale chunks           │  Upload in batches of 100
├─────────────────────────────────┤
│ 8. SAVE STATE: ingestion_state   │  Per-page hash + chunk hashes
│    .json after each batch        │  Enables resumable ingestion
└─────────────────────────────────┘
```

### Intelligent Chunking (SmartContentChunker)

The chunker parses markdown structure before splitting:

```
Document structure detected:
  ┌── Header sections (# ## ###)
  ├── Code blocks (```)
  ├── Lists (- * 1.)
  └── Paragraphs (everything else)

Splitting strategy:
  1. Parse into typed sections (header, code, list, paragraph)
  2. Accumulate sections into chunks until 800 tokens
  3. When limit exceeded, finalize chunk + start new one with overlap
  4. Large sections (>800 tok) get split at line boundaries
  5. Overlap: take last ~200 tokens of previous chunk as prefix
```

**Why 800 tokens / 200 overlap?**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_tokens` | 800 | Small enough for precise retrieval (less noise per chunk), large enough to contain a complete concept. Previously 1500 — we halved it for precision. |
| `overlap_tokens` | 200 | 25% overlap ensures sentences split at boundaries are complete in at least one chunk. Increased from 150 to catch more boundary cases. |

**Visual:**
```
┌──────────────────────────────┐
│ Chunk 1 (800 tokens)          │
│                      ┌────────┼──────────────────┐
│               overlap│200 tok │ Chunk 2 (800 tok) │
└──────────────────────┘────────┘──────────────────┘
```

### Contextual Chunk Enrichment

**Problem:** A chunk containing "The timeout is 30 seconds" is meaningless without knowing WHAT has a 30-second timeout.

**Solution:** Before embedding, prepend each chunk with its page path and section title:

```
Before enrichment (what gets stored in the index):
  "The timeout is 30 seconds. If exceeded, the request will be retried..."

After enrichment (what gets embedded):
  "Page: /Engineering/Gateway/Configuration > Section: Request Timeouts

   The timeout is 30 seconds. If exceeded, the request will be retried..."
```

The enriched text is used ONLY for embedding generation. The index stores the original (unenriched) text — this way the user sees clean content in search results.

**Why this matters:** The embedding of the enriched text captures "gateway request timeout configuration" as a topic, not just "timeout 30 seconds" which could be about anything.

### Delta Detection

Each page and chunk gets an MD5 hash. On re-ingestion:

```python
# Per-page: hash of content + path + lastModified
page_hash = md5(json.dumps({content, path, lastModified}))

# Per-chunk: hash of chunk text
chunk_hashes = [md5(chunk.text) for chunk in chunks]

# Decision logic:
if page_hash unchanged:     → skip entire page
if chunk_hashes changed:    → re-embed only changed chunks
if chunk_count decreased:   → delete stale trailing chunks
```

**Result:** Re-ingesting a 77-page wiki where 2 pages changed takes ~10 seconds instead of ~15 minutes.

---

## Query Pipeline — All Stages

Every user question passes through a multi-stage pipeline. Each stage independently improves answer quality.

### Stage 1: Conversation Contextualization

**Problem:** Follow-up questions like *"Tell me more about that"* are meaningless without conversation context.

**Solution:** Rewrite follow-ups into standalone queries using chat history.

```
Chat History:
  User: "What does GW_BE_Request latency mean?"
  Assistant: "GW_BE_Request latency measures the time from..."

New Input: "How is the backend version different?"

→ LLM Rewrite → "How is the backend version of GW_BE_Request latency
                  different from the frontend version?"
```

**How it works:** We send the last 3 Q/A turns (truncated to 300 chars each) plus the follow-up to the LLM and ask it to produce a standalone search query. If the question is already standalone (first message in chat), this step is skipped.

**Token cost:** ~200-400 tokens. Only triggered when `chat_history` has ≥2 messages.

---

### Stage 2: Multi-Query Expansion

**Problem:** A single query phrasing may miss documents that use different terminology. *"deployment process"* won't match *"release pipeline"*.

**Solution:** Generate 4 alternative phrasings, then search with all 5 independently.

```
Original:  "What does GW_BE_Request latency metric mean?"

Expansion 1: "What is the meaning of the GW_BE_Request latency metric?"
Expansion 2: "Can you explain the GW_BE_Request latency metric?"
Expansion 3: "What does the GW_BE_Request latency measurement represent?"
Expansion 4: "How is the GW_BE_Request latency metric defined?"
```

**Why 4 alternatives (5 total)?**
- Fewer than 3: insufficient lexical diversity
- More than 5: diminishing returns + added latency
- 5 queries × hybrid search = excellent recall coverage

**Token cost:** ~400-600 tokens.

---

### Stage 3: HyDE — Hypothetical Document Embeddings

**Problem:** There's a fundamental asymmetry between queries and documents. A query *"What is DRI?"* is 4 words. The relevant document is 500 words. Their embeddings may not be close in vector space.

**Solution:** Generate a **hypothetical answer** first, then search with that answer's embedding. The embedding of a plausible answer is semantically closer to actual documents than the embedding of a question.

```
Query:  "What does GW_BE_Request latency metric mean?"

HyDE Generation (LLM writes a plausible wiki paragraph):
  "The GW_BE_Request latency metric measures the time taken from when
   the gateway receives a request from the client to when it sends
   the last response byte. This includes TCP connection time, TLS
   handshake, and backend processing time..."

→ Embed this hypothetical answer (3072-dim vector)
→ Search with it — it's semantically closer to actual wiki content
  than the original 10-word question
```

**Visual — why HyDE works:**
```
    Vector Space

    "What is DRI?"                           ← Query embedding
         ●                                      (far from documents)



    "DRI stands for Directly Responsible
     Individual. It is a role..."            ← HyDE embedding
                          ●                     (close to actual docs!)

    "The DRI (Directly Responsible
     Individual) is assigned to..."          ← Actual document
                           ●
```

We search with BOTH the original query embedding AND the HyDE embedding. Total: 6 search calls (5 multi-query + 1 HyDE).

**Token cost:** ~500-800 tokens for HyDE generation.

---

### Stage 4: Hybrid Semantic Search (×6 parallel)

**Problem:** Vector search alone misses exact keyword matches. Keyword search alone misses semantic matches.

**Solution:** Each of the 6 query variants runs as a **hybrid search** — combining BM25 keyword scoring + vector cosine similarity + semantic reranker in a single Azure AI Search call.

```
For each of the 6 query variants:
┌───────────────────────────────────────────────┐
│              Azure AI Search Call               │
│                                                 │
│  BM25 Keyword Search    Vector Search (HNSW)   │
│  on "content" field     on "content_vector"     │
│  (tf-idf scoring)       (cosine, 3072-dim)      │
│         │                       │               │
│         └───────┬───────────────┘               │
│                 ▼                                │
│    Semantic Reranker (transformer model)         │
│    Reads full text + query → relevance score     │
│                 │                                │
│                 ▼                                │
│         Top 30 results per query                 │
└───────────────────────────────────────────────┘

Total: 6 queries × up to 30 results = up to 180 candidates
(with significant overlap — good documents appear in multiple queries)
```

**When each signal wins:**

| Signal | Best For | Example |
|--------|----------|---------|
| BM25 | Exact term matches, acronyms | "SLA exception" → exact match |
| Vector | Paraphrases, conceptual similarity | "How fast is CDN?" → finds latency docs |
| Semantic reranker | Understanding intent | "What does X mean?" vs "When was X created?" |

**HNSW Index Parameters:**

| Parameter | Value | What it does |
|-----------|-------|-------------|
| `m` | 4 | Bi-directional links per node. Optimal for <100K docs. |
| `efConstruction` | 400 | Build-time search width. Higher = better quality index. |
| `efSearch` | 500 | Query-time search width. Higher = better recall. |
| `metric` | cosine | Standard for text embeddings. |

---

### Stage 5: Reciprocal Rank Fusion (RRF)

**Problem:** We have up to 180 results from 6 queries. Many duplicates. Need to merge into a single ranked list.

**Solution:** RRF — a proven rank aggregation method that combines multiple ranked lists using only rank positions (not scores).

**Formula:**
```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                 i∈queries

k = 60 (smoothing constant)
```

**Example:**
```
Query 1 results: [Doc_A (rank 1), Doc_B (rank 2), Doc_C (rank 3)]
Query 2 results: [Doc_B (rank 1), Doc_A (rank 2), Doc_D (rank 3)]
Query 3 results: [Doc_A (rank 1), Doc_D (rank 2), Doc_E (rank 3)]

RRF scores:
  Doc_A: 1/61 + 1/62 + 1/61 = 0.0489  ← Highest (appeared in all 3, mostly rank 1)
  Doc_B: 1/62 + 1/61 + 0    = 0.0325
  Doc_D: 0    + 1/63 + 1/62 = 0.0320
  Doc_C: 1/63 + 0    + 0    = 0.0159
  Doc_E: 0    + 0    + 1/63 = 0.0159
```

**Why k=60?**
- From the original RRF paper (Cormack et al., 2009)
- `k=60` means rank-1 scores `1/61 = 0.0164` and rank-10 scores `1/70 = 0.0143` — gentle 13% decay
- Lower `k` (e.g., 1): rank-1 dominates too heavily (`1/2 = 0.5` vs `1/11 = 0.09`)
- Higher `k` (e.g., 1000): all ranks nearly equal, defeats purpose

**Why RRF over score averaging?**
- BM25 scores are unbounded; vector similarities are 0–1; reranker scores are 0–4
- These scales are **incomparable** — you can't average them
- RRF uses only rank positions, which are always comparable

**Post-RRF filter:** We keep results with `distance < 0.8` (relaxed threshold because LLM reranking refines further).

---

### Stage 6: LLM Reranking

**Problem:** RRF gives a good ranking but is based on positional fusion, not deep semantic understanding of the query-document relationship.

**Solution:** Send the top 25 RRF candidates to GPT-4.1 with the query, ask it to score each chunk's relevance 1–10.

```
Prompt to LLM:
  "Score each chunk's relevance to answering the query on a scale of 1-10.
   Return JSON: [{"index": 0, "score": 8}, {"index": 1, "score": 3}, ...]"

  Query: "What does GW_BE_Request latency mean?"

  [0] (page: /Monitoring/Metrics) The GW_BE_Request latency measures...
  [1] (page: /Deployment/Rollback) To rollback a deployment, follow...
  [2] (page: /Monitoring/Alerts) Configure alerts for GW_BE_Request...
  ...

Result:
  [0] score: 9  ← Directly answers the question
  [1] score: 2  ← Irrelevant (deployment, not metrics)
  [2] score: 7  ← Related but about alerts, not definition
```

**Filtering:** Keep only chunks with score ≥ 3. This removes false positives that passed keyword/vector matching but aren't actually relevant.

**Why this is the highest-impact optimization:**

| Without LLM Reranking | With LLM Reranking |
|----------------------|-------------------|
| Top result might be a keyword match that's topically wrong | Top result is the most semantically relevant |
| BM25 "GW_BE_Request" matches ANY page mentioning it | Reranker understands the question asks for *meaning/definition* |
| ~60% precision@3 | ~85% precision@3 |

**Token cost:** ~1500-2500 tokens (25 chunk previews × ~100 tokens each + response).

---

### Stage 7: Context Selection

**Problem:** Even after reranking, we may have 15+ relevant chunks. Can't fit all in the LLM prompt.

**Solution:** Greedy selection with diversity constraints and token budget.

```python
# Configuration:
per_page_cap = 4           # Max chunks from any single wiki page
max_context_tokens = 122904 # model_window - answer_budget - system_overhead

# Algorithm:
for chunk in sorted_by_relevance:
    if page_count[chunk.page] >= 4:  continue  # Diversity
    if total_tokens + chunk.tokens > budget: continue  # Budget
    selected.append(chunk)
```

**Per-page cap (4):** Prevents a single long wiki page from consuming the entire context. Forces diversity across different source pages and perspectives.

**Dynamic token budget:** Calculated as `model_context_window - answer_budget(4096) - system_overhead(1000)`. For GPT-4.1 (128K window), this leaves ~122K tokens for context — far more than we typically need.

---

### Stage 8: Neighbor Chunk Stitching

**Problem:** A chunk may contain only part of an explanation that spans two chunks. The answer might be split across chunk boundaries.

**Solution:** For each selected chunk, fetch its immediate neighbors (chunk_index ± 1) from the same page and stitch them together.

```
Before stitching:
  Selected chunk (index 5): "...the DRI is defined"  ← incomplete

After stitching (fetch chunks 4 and 6):
  Chunk 4: "In our organization, the role of"
  Chunk 5: "the DRI is defined as the person who"    ← original selection
  Chunk 6: "is ultimately accountable for the outcome..."

  Stitched: "In our organization, the role of the DRI is defined
             as the person who is ultimately accountable for the
             outcome..."                                ← complete!
```

**Implementation:** A `fetch_neighbor_chunks` method on the search client queries Azure AI Search with a filter:
```
page_path eq '/path/to/page' and chunk_index eq {target_index}
```

This adds ~200ms latency per selected chunk but dramatically improves answer completeness for information that spans chunk boundaries.

---

### Stage 9: Answer Generation

**Problem:** The LLM must synthesize an answer from retrieved context AND maintain conversational coherence with prior chat turns.

**Solution:** A chain-of-thought system prompt + chat history + context documents.

**System prompt key instructions:**
1. THINK step-by-step about which sources are relevant
2. Lead with a DIRECT ANSWER
3. Provide DETAILED EXPLANATION with supporting details
4. Add STEP-BY-STEP INSTRUCTIONS if applicable
5. Use INLINE CITATIONS: `[Source 1]`, `[Source 3]`
6. State GAPS if context doesn't fully answer

**Message assembly:**
```python
messages = [
    system_message,                    # ~500 tokens (instructions)
    *chat_history[-10:],               # Up to 8K tokens of prior turns
    {
        "role": "user",
        "content": f"Context:\n{stitched_contexts}\n\nQuestion: {query}"
    }
]

response = ai_client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    temperature=0.0,    # Maximum factual precision
    max_tokens=4096     # Comprehensive answers
)
```

**Why temperature=0?**
- We want maximum factual precision — no creative rephrasing of technical content
- Combined with "ONLY use information from sources" instruction, this minimizes hallucination
- For RAG, deterministic output is almost always better

---

## RAG Optimizations & Techniques

### Embedding Model: text-embedding-3-large (3072-dim)

| Model | Dimensions | MTEB Score | Status |
|-------|-----------|------------|--------|
| text-embedding-ada-002 | 1536 | 61.0 | ❌ Legacy (was default) |
| text-embedding-3-small | 1536 | 62.3 | ❌ Lower fidelity |
| **text-embedding-3-large** | **3072** | **64.6** | **✅ Our choice** |

**Why 3072 dimensions?**
- Higher dimensionality captures finer semantic distinctions
- *"request latency"* vs *"request timeout"* are closer in 1536-dim space but distinguishable in 3072-dim
- For technical documentation where terms are precise, this distinction is critical
- Trade-off: 2× storage and slightly slower cosine computation (negligible with HNSW)

### Reranker Score Normalization

**Bug found and fixed:** Azure AI Search's semantic reranker returns scores in the range **0–4** (not 0–1). The original code clipped scores to `min(score, 1.0)`, making all reranked results look identical (distance = 0).

**Fix:**
```python
# Before (broken):
distance = 1.0 - min(max(score, 0.0), 1.0)  # All scores > 1 become distance 0

# After (correct):
if reranker_score is not None:
    s = min(float(reranker_score) / 4.0, 1.0)  # Normalize 0-4 → 0-1
else:
    s = min(float(search_score), 1.0)
distance = 1.0 - max(s, 0.0)
```

### Azure Search Field Design

```
Index: ado_wiki (3072-dim HNSW, cosine)
┌─────────────────────────────────────────────────┐
│ id              String (key, base64-encoded)      │
│ content         String (searchable, BM25)         │
│ content_vector  Collection(Single) [3072-dim]     │
│ page_id         String (filterable)               │
│ page_path       String (searchable, filterable)   │
│ parent_path     String (filterable)               │
│ chunk_index     Int32 (filterable)                │
│ chunk_type      String (filterable, facetable)    │
│ start_line      Int32                             │
│ end_line        Int32                             │
│ token_count     Int32 (filterable)                │
│ section_title   String (searchable, filterable)   │
│ headers_path    String (searchable, filterable)   │
│ last_modified   DateTimeOffset (filterable)       │
│ tags            Collection(String) (filterable)   │
│                                                   │
│ Semantic config: my-semantic-config               │
│   content_fields: [content]                       │
│ Vector profile: my-vector-config                  │
│   algorithm: HNSW (m=4, efC=400, efS=500)        │
└─────────────────────────────────────────────────┘
```

### ID Encoding

Document IDs are `{page_id}_{chunk_index}`, base64url-encoded to be Azure Search key-safe:
```python
encoded = base64.urlsafe_b64encode(f"{page_id}_{chunk_index}".encode()).rstrip(b'=')
```
This prevents path characters (`/`, `[`, `]`) from breaking the index key constraint.

---

## Conversational Memory

### Session State Architecture

```
st.session_state
├── messages: [                        # Full conversation
│   {"role": "user", "content": "What is DRI?"},
│   {"role": "assistant", "content": "DRI stands for..."},
│   {"role": "user", "content": "Who assigns them?"},
│   {"role": "assistant", "content": "DRIs are assigned by..."}
│ ]
├── query_metrics: [                   # Per-query telemetry
│   {"tokens": 7234, "sources": 5, "avg_relevance": 0.82},
│   {"tokens": 5102, "sources": 3, "avg_relevance": 0.91}
│ ]
├── last_contexts: [QueryResult, ...]  # Source docs from last query
├── total_queries: 2
└── processor: EnhancedQueryProcessor  # Cached (one per session)
```

### Chat History in LLM Context

- Last 10 messages included in LLM prompt (up to 8K tokens)
- Older messages still visible in the chat UI but NOT sent to LLM
- This balances context quality with token cost
- `contextualize_query()` uses history to rewrite follow-ups into standalone queries for search

---

## UI & Observability

### Chat Interface Layout

```
┌───────────────────────────────────────────────────┐
│ 🔍 ADO Wiki Assistant                     [New ↻] │
├────────────────────────────┬──────────────────────┤
│                            │ 📊 Session Metrics   │
│ 🧑 What does GW_BE...      │ Total Tokens: 12,336 │
│                            │ Queries: 2           │
│ 🤖 GW_BE_Request latency  │ Sources: 8           │
│    measures the time...    │ Avg Relevance: 86%   │
│    [Source 1] [Source 3]   │                      │
│                            │ Per-query:           │
│ 🧑 How is the backend     │ Q1: 7,234 tok | 82%  │
│    version different?      │ Q2: 5,102 tok | 91%  │
│                            │                      │
│ 🤖 The backend version    │ 📚 Sources (3)       │
│    differs in that...      │ ▸ /Monitoring/Metrics│
│                            │ ▸ /GW/Config         │
│ [Type a message...]        │ ▸ /Perf/Testing      │
└────────────────────────────┴──────────────────────┘
```

### Metrics Tracked

| Metric | Per Query | Per Session |
|--------|-----------|-------------|
| Tokens used | ✅ | ✅ (cumulative) |
| Sources found | ✅ | ✅ (cumulative) |
| Avg relevance | ✅ | ✅ (session average) |
| Query count | — | ✅ |
| Conversation turns | — | ✅ |

---

## Quick Start

### 1. Configure

Copy `.env.example` to `.env` and fill in:

```env
ADO_ORG=your-org
ADO_PROJECT=your-project
ADO_WIKI=your-wiki
ADO_PAT=your-pat
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-large
AZURE_OPENAI_CHAT_MODEL=gpt-4.1
SEARCH_SERVICE_NAME=your-search-service
SEARCH_SERVICE_KEY=your-search-key
INDEX_NAME=ado_wiki
```

### 2. Ingest

```bash
pip install -r requirements.txt
python ingest.py
```

### 3. Query

```bash
streamlit run streamlit_app.py     # Chat UI
python query.py                    # CLI
```

### 4. Deploy

```powershell
.\deploy.ps1
```

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full guide. Summary:

| Step | Command |
|------|---------|
| Set App Settings | `az webapp config appsettings set ...` |
| Set startup | `az webapp config set --startup-file "bash startup.sh"` |
| Deploy | `.\deploy.ps1` |
| Verify | Open `https://<app>.azurewebsites.net` |

**Required Azure resources:**

| Resource | SKU | Purpose |
|----------|-----|---------|
| App Service | B1+ Linux Python 3.12 | Hosts Streamlit |
| Azure AI Search | Basic+ | Hybrid index + semantic ranking |
| Azure OpenAI | Standard | GPT-4.1 + text-embedding-3-large |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ADO_ORG` | — | Azure DevOps organization |
| `ADO_PROJECT` | — | Project name |
| `ADO_WIKI` | — | Wiki name or ID |
| `ADO_PAT` | — | Personal Access Token |
| `AZURE_OPENAI_ENDPOINT` | — | OpenAI resource URL |
| `AZURE_OPENAI_KEY` | — | API key |
| `AZURE_OPENAI_CHAT_MODEL` | `gpt-4.1` | Chat model deployment |
| `AZURE_OPENAI_EMBED_MODEL` | `text-embedding-3-large` | Embedding model deployment |
| `SEARCH_SERVICE_NAME` | — | Search service name |
| `SEARCH_SERVICE_KEY` | — | Search admin key |
| `INDEX_NAME` | `ado_wiki` | Index name |
| `ENVIRONMENT` | `local` | `local` = env vars; `production` = Key Vault |
| `WEBSITES_CONTAINER_START_TIME_LIMIT` | `230` | Set to `600` on Azure |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Vector dimension mismatch (1536 vs 3072) | Wrong embedding model | Set `AZURE_OPENAI_EMBED_MODEL=text-embedding-3-large`; delete index; re-ingest |
| `$'\r': command not found` | Windows line endings in startup.sh | `deploy.ps1` auto-fixes this |
| Key Vault overriding App Settings | `ENVIRONMENT=production` | Set `ENVIRONMENT=local` |
| No search results | Empty index or wrong keys | Run `python ingest.py`; verify `SEARCH_SERVICE_NAME`/`KEY` |
| Application Error on cold start | pip install takes >230s | Set `WEBSITES_CONTAINER_START_TIME_LIMIT=600` |
| `query_language` error | Old SDK incompatibility | Already removed from code |
| `tags` null error | Null not allowed for Collection field | Already fixed: sends `[]` instead of `null` |
| `last_modified` empty string error | Empty string for DateTimeOffset | Already fixed: sends `None` instead of `""` |

---

## Performance Characteristics

| Operation | Latency | Tokens |
|-----------|---------|--------|
| Query contextualization | ~500ms | ~300 |
| Multi-query expansion | ~800ms | ~500 |
| HyDE generation | ~1s | ~600 |
| Embedding per query | ~200ms | — |
| Hybrid search per query | ~300ms | — |
| RRF fusion | <10ms | 0 |
| LLM reranking | ~2s | ~2000 |
| Neighbor stitching | ~500ms | — |
| Answer generation | ~2-4s | ~5000 |
| **Full pipeline (first query)** | **~8-12s** | **~8,000-10,000** |
| **Follow-up query** | **~8-12s** | **~9,000-11,000** |

---

*Built for engineering teams — making wiki knowledge instantly accessible through conversation.*
