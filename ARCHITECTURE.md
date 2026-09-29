# Streaming Live RAG Engine
## System Architecture & Technical Specifications Report

---

## 1. Trigger Policies (Logic Gates)

The retrieval engine employs a 5-gate deterministic routing pipeline to evaluate every incoming turn before firing context queries:
[User Input] ──► G1: Wait ──(Sufficient Context?)──► [Process Local State]
│
(Pass)
▼
G2: Provisional Retrieve ──(High Confidence?)──► G3: Commit
│                                               │
(Low / Ambiguous)                           (Merge Context)
▼                                               ▼
G4: Suppress / G5: Fallback ───────────────► [System Output]
* **G1: Wait Gate:** Evaluates whether local session history contains sufficient state to answer the query. If turn entropy is below a defined threshold, external retrieval is bypassed.
* **G2: Provisional Retrieve Gate:** Executes speculative, low-k dense similarity probes in the background when semantic uncertainty crosses threshold $\tau_{\text{retrieve}}$.
* **G3: Commit Gate:** Re-evaluates provisional retrieval candidates against the full conversation context. If precision scoring passes $\tau_{\text{commit}}$, retrieved chunks are committed to active context.
* **G4: Suppress Gate:** Halts retrieval execution when input patterns match conversational filler, sensitive domains, or explicit constraints, preventing context contamination.
* **G5: Fallback Gate:** Triggers alternative fallback strategies (e.g., re-ranking, query broadening, or direct parametric synthesis) when G3 validation fails or vector searches yield zero high-confidence matches.

---

## 2. Asynchronous Query Decomposition

Compound inputs containing multi-part intents are split into distinct atomic sub-queries and evaluated concurrently via Python’s `asyncio` event loop.

```python
import asyncio
from typing import List, Dict, Any

async def process_subquery(sub_query: str, vector_client) -> Dict[str, Any]:
    """Asynchronously fetches dense vectors and BM25 matches for a single sub-query."""
    return await vector_client.asearch(query=sub_query)

async def decompose_and_retrieve(
    compound_query: str, 
    decomposer, 
    vector_client
) -> List[Dict[str, Any]]:
    # Step 1: Split compound intent into atomic queries
    sub_queries: List[str] = decomposer.split_intent(compound_query)
    
    # Step 2: Schedule concurrent retrieval tasks across vector indexes
    tasks = [process_subquery(sq, vector_client) for sq in sub_queries]
    
    # Step 3: Gather results asynchronously to minimize total network wait time
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    return results
### Part 2: Sections 3 & 4

```markdown
---

## 3. Dynamic State-Diffing Mechanics

To eliminate redundant vector similarity calls across conversational turns, `SessionStateDiffer` tracks active knowledge graph triples and semantic hashes. Instead of re-querying the full session state, it calculates delta updates ($v_n \rightarrow v_{n+1}$).

Turn 1 State (v1): {Entities: [A, B], Hashes: [0x8A1]}
Turn 2 Input:      "How does that relate to project Y?"
Diff Engine:       Calculates Δ = {Added: [Y], Invalidated: []}
Execution:         Retrieves vector embeddings ONLY for [Y]
Turn 2 State (v2): {Entities: [A, B, Y], Hashes: [0x8A1, 0x3F9]}
* **Graph Hash Verification:** Computes a lightweight checksum of active session entities and parameters.
* **Delta Extraction ($\Delta$):** Identifies newly introduced entities or topic shifts, isolating modified state tokens.
* **Selective Vector Fetch:** Formulates retrieval payloads exclusively for $\Delta$, preserving prior context nodes and preventing full-vector re-retrievals.
* **Version Bump:** Merges new context nodes into the active graph, incrementing state from $v1 \rightarrow v2$.

---

## 4. Latency & Precision Trade-Offs (BM25 + Dense RRF)

Hybrid retrieval in Qdrant combines exact keyword matching (BM25) with semantic dense vectors using Reciprocal Rank Fusion (RRF). Tuning this pipeline involves balancing latency and retrieval accuracy.

| Component / Metric | BM25 (Sparse) | Dense Vector (HNSW) | Hybrid (BM25 + Dense RRF) |
| :--- | :--- | :--- | :--- |
| **Primary Strength** | Exact keyword/code match | Semantic/conceptual overlap | High overall recall & precision |
| **P95 Latency Impact** | Low (~2–5 ms) | Moderate (~12–25 ms) | Higher (~20–40 ms) |
| **Compute Overhead** | Negligible (CPU bound) | Moderate (GPU/Vector RAM) | Double fetch + Fusion overhead |
| **Precision Strategy** | Strong for unique identifiers | Strong for high-level concepts | Mitigates single-model edge cases |

### Reciprocal Rank Fusion Calculation
RRF normalizes and merges rank positions across sparse and dense result sets:

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where:
* $M$: Retrieval models (BM25 and Dense HNSW).
* $r_m(d)$: Rank position of document $d$ in model $m$.
* $k$: Smoothing constant (typically set to $60$).

> **Optimization Note:** To keep P95 latency under target thresholds without sacrificing retrieval quality, G2 executes speculative sparse BM25 lookups first, escalating to full hybrid RRF via G3 only when top-1 confidence falls below operational thresholds.
