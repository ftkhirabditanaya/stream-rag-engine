import uuid
from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Connect to local Qdrant instance
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "samsung_rag_docs"

# Fast, lightweight embedding models
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "qdrant/bm25"

def ingest_documents(raw_texts: list[dict]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    points = []

    for item in raw_texts:
        chunks = text_splitter.split_text(item["text"])
        
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            
            # Grounding metadata for Gate G6 citations
            payload = {
                "doc_id": item["doc_id"],
                "section": item["section"],
                "citation_label": f"[{item['doc_id']} {item['section']}]",
                "text": chunk
            }

            # Generate both Dense and BM25 Sparse vectors automatically
            points.append(
                models.PointStruct(
                    id=point_id,
                    payload=payload,
                    vector={
                        "dense": models.Document(text=chunk, model=DENSE_MODEL),
                        "sparse": models.Document(text=chunk, model=SPARSE_MODEL)
                    }
                )
            )

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=32
    )
    print(f"Successfully ingested {len(points)} document chunks into '{COLLECTION_NAME}'!")

if __name__ == "__main__":
    sample_corpus = [
        {
            "doc_id": "Doc_01", 
            "section": "§1.2", 
            "text": "The event venue capacity in Pune is strictly limited to 150 guests. Registration closes on Sept 28th."
        },
        {
            "doc_id": "Doc_02", 
            "section": "§3.1", 
            "text": "Cancellation requests made less than 48 hours before the event will incur a 50% non-refundable fee."
        },
        {
            "doc_id": "Doc_03",
            "section": "§4.5",
            "text": "All hackathon project submissions must be uploaded before September 30th with a runnable Docker setup."
        }
    ]
    ingest_documents(sample_corpus)