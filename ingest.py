import uuid
from qdrant_client import models
from search_engine import get_qdrant_client, COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL

def simple_text_splitter(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Lightweight text splitter replacing heavy framework imports."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += (chunk_size - chunk_overlap)
    return chunks

def ingest_documents(raw_texts: list[dict]):
    client = get_qdrant_client()
    if client is None:
        print("[Error] Failed to acquire Qdrant client for ingestion.")
        return False

    points = []

    for item in raw_texts:
        chunks = simple_text_splitter(item["text"], chunk_size=500, chunk_overlap=50)
        
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            
            # Grounding metadata for Gate G6 citations
            payload = {
                "doc_id": item["doc_id"],
                "section": item.get("section", "§1.0"),
                "citation_label": f"[{item['doc_id']} {item.get('section', '§1.0')}]",
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

    try:
        client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points,
            batch_size=32
        )
        print(f"Successfully ingested {len(points)} document chunks into '{COLLECTION_NAME}'!")
        return True
    except Exception as e:
        print(f"Failed to ingest documents: {e}")
        return False

sample_corpus = [
    {
        "doc_id": "Doc_01", 
        "section": "§1.2", 
        "text": "The main event venue capacity in Pune is strictly limited to 150 guests due to fire safety regulations. Early bird registration closes on Sept 28th at midnight."
    },
    {
        "doc_id": "Doc_02", 
        "section": "§3.1", 
        "text": "Cancellation requests made less than 48 hours before the event start time will incur a 50% non-refundable fee. Full refunds are provided for cancellations made 7+ days in advance."
    },
    {
        "doc_id": "Doc_03",
        "section": "§4.5",
        "text": "All hackathon project submissions must be uploaded to the official portal before September 30th at 11:59 PM EST with a fully runnable Docker container setup and complete README documentation."
    },
    {
        "doc_id": "Doc_04",
        "section": "§2.4",
        "text": "Streaming RAG Architecture utilizes Reciprocal Rank Fusion (RRF) combining dense semantic vectors (MiniLM-L6-v2) and sparse BM25 token frequencies to achieve sub-50ms P95 retrieval latency."
    },
    {
        "doc_id": "Doc_05",
        "section": "§5.1",
        "text": "Technical support and live Q&A sessions are available via Discord channel #dev-help from 9 AM to 6 PM UTC daily during the hackathon sprint."
    },
    {
        "doc_id": "Doc_06",
        "section": "§6.3",
        "text": "API Rate Limits: WebSocket streaming connections are capped at 100 messages per minute per IP address. Exceeding this limit triggers temporary HTTP 429 backoff throttling."
    }
]

if __name__ == "__main__":
    ingest_documents(sample_corpus)