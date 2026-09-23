from qdrant_client import QdrantClient, models

# Connect to local Qdrant instance
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "samsung_rag_docs"

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "qdrant/bm25"

def hybrid_search(query_text: str, top_k: int = 3):
    """
    Runs Dense (Semantic) and Sparse (BM25 Keyword) searches in parallel,
    and fuses rankings via Reciprocal Rank Fusion (RRF).
    """
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
            "citation": hit.payload["citation_label"],
            "text": hit.payload["text"],
            "score": hit.score
        }
        for hit in results.points
    ]

if __name__ == "__main__":
    test_query = "What is the cancellation fee and submission deadline?"
    hits = hybrid_search(test_query)
    
    print(f"\n--- Hybrid Search Results for Query: '{test_query}' ---")
    for hit in hits:
        print(f"Citation: {hit['citation']} | RRF Score: {hit['score']:.4f}")
        print(f"Content:  {hit['text']}\n")