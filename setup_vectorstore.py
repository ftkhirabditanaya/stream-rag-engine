from qdrant_client import models
from search_engine import get_qdrant_client, COLLECTION_NAME

def init_hybrid_collection():
    client = get_qdrant_client()
    if client is None:
        print("[Error] Failed to acquire Qdrant client.")
        return

    try:
        # Remove old collection if it exists to start fresh
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        # Create hybrid collection (Dense + Sparse BM25)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully with Hybrid Vectors!")
    except Exception as e:
        print(f"Failed to initialize collection: {e}")

if __name__ == "__main__":
    init_hybrid_collection()