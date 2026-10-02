import os
import asyncio
import threading
from typing import List, Dict, Any

COLLECTION_NAME = "samsung_rag_docs"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "qdrant/bm25"

_cached_client = None
_client_lock = threading.Lock()
_search_lock = threading.Lock()

def get_qdrant_client(url: str = None):
    """
    Returns a QdrantClient thread-safely.
    1. Tries HTTP Qdrant server (default: http://localhost:6333 or QDRANT_URL).
    2. Fallbacks automatically to local embedded storage (./qdrant_db) if server is offline.
    """
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    with _client_lock:
        if _cached_client is not None:
            return _cached_client

        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=url, timeout=1.5, check_compatibility=False)
            client.get_collections()
            print(f"[Qdrant] Connected to remote server at {url}")
            _cached_client = client
            return client
        except Exception:
            try:
                from qdrant_client import QdrantClient
                db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_db")
                os.makedirs(db_dir, exist_ok=True)
                print(f"[Qdrant] HTTP server unreachable. Using local embedded database at {db_dir}")
                client = QdrantClient(path=db_dir)
                _cached_client = client
                return client
            except Exception as e:
                print(f"[Qdrant Error] Failed to initialize Qdrant client: {e}")
                return None

def ensure_collection_populated():
    """Ensures collection is created and populated with sample docs if empty."""
    client = get_qdrant_client()
    if client is None:
        return
    try:
        if not client.collection_exists(COLLECTION_NAME):
            print(f"[Qdrant Setup] Collection '{COLLECTION_NAME}' missing. Auto-initializing vectorstore...")
            from setup_vectorstore import init_hybrid_collection
            from ingest import ingest_documents, sample_corpus
            init_hybrid_collection()
            ingest_documents(sample_corpus)
    except Exception as e:
        print(f"[Qdrant Setup Error] {e}")

def hybrid_search(query_text: str, top_k: int = 3, client=None) -> List[Dict[str, Any]]:
    """
    Runs Dense (Semantic) and Sparse (BM25 Keyword) searches in parallel,
    and fuses rankings via Reciprocal Rank Fusion (RRF).
    Uses a thread lock to ensure thread-safety for fastembed under concurrent sub-queries.
    """
    try:
        from qdrant_client import models
        if client is None:
            client = get_qdrant_client()

        if client is None:
            return []

        if not client.collection_exists(COLLECTION_NAME):
            ensure_collection_populated()

        with _search_lock:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(text=query_text, model=DENSE_MODEL),
                        using="dense",
                        limit=10
                    ),
                    models.Prefetch(
                        query=models.Document(text=query_text, model=SPARSE_MODEL),
                        using="sparse",
                        limit=10
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k
            )

        return [
            {
                "citation": hit.payload.get("citation_label", "[Doc_Ref]"),
                "doc_id": hit.payload.get("doc_id", "Unknown"),
                "section": hit.payload.get("section", ""),
                "text": hit.payload.get("text", ""),
                "score": float(hit.score)
            }
            for hit in results.points
        ]
    except Exception as e:
        print(f"[Hybrid Search Exception] Query '{query_text}': {e}")
        return []


async def async_hybrid_search(query_text: str, top_k: int = 3, client=None) -> List[Dict[str, Any]]:
    """
    Asynchronously executes hybrid search in a worker thread to allow concurrent
    sub-query evaluation via asyncio.gather (Section 2 of System Architecture).
    """
    return await asyncio.to_thread(hybrid_search, query_text, top_k, client)

def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates search results by citation and text content, keeping highest RRF score."""
    seen = {}
    for item in results:
        key = (item.get("citation"), item.get("text"))
        if key not in seen or item.get("score", 0) > seen[key].get("score", 0):
            seen[key] = item
    
    sorted_docs = sorted(list(seen.values()), key=lambda x: x.get("score", 0), reverse=True)
    return sorted_docs

if __name__ == "__main__":
    ensure_collection_populated()
    test_query = "What is the cancellation fee and submission deadline?"
    hits = hybrid_search(test_query)
    
    print(f"\n--- Hybrid Search Results for Query: '{test_query}' ---")
    if hits:
        for hit in hits:
            print(f"Citation: {hit['citation']} | RRF Score: {hit['score']:.4f}")
            print(f"Content:  {hit['text']}\n")
    else:
        print("No vector results found.")