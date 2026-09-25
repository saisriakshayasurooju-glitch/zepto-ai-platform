from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DB = ROOT / "chroma_db"
COLLECTION_NAME = "zepto_policies"

MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_documents():
    chunks = []
    for path in sorted(DOCS.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        # The supplied documents are short enough that one chunk per document
        # is acceptable under the brief.
        chunks.append({
            "id": path.stem,
            "document_id": path.stem,
            "text": text
        })
    return chunks


def build_collection():
    model = SentenceTransformer(MODEL_NAME)
    chunks = chunk_documents()

    client = chromadb.PersistentClient(path=str(DB))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"document_id": c["document_id"]} for c in chunks],
        embeddings=embeddings
    )

    print(f"Indexed {len(chunks)} documents into {COLLECTION_NAME}.")
    return collection


if __name__ == "__main__":
    build_collection()
