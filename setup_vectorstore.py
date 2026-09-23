from qdrant_client import QdrantClient, models

# Connect to local Qdrant instance
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "samsung_rag_docs"

def init_hybrid_collection():
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

if __name__ == "__main__":
    init_hybrid_collection()
    