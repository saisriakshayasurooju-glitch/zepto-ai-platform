# Module 3 — Support Assistant

## Architecture

```text
8 policy documents
      |
      v
ingest.py: chunk per document
      |
      v
SentenceTransformer(all-MiniLM-L6-v2)
      |
      v
ChromaDB collection: zepto_policies
      |
      | query embedding + cosine retrieval
      v
LangGraph: classify_intent
      |
      +-----------------------------+
      |                             |
policy_question              general_question
      |                             |
      v                             v
retrieve_and_answer            direct_answer
      |                             |
      v                             v
grounded mock answer          fixed mock answer
      |
      v
Pydantic response
      |
      v
FastAPI POST /ask
```

### Data flow

- `ingest.py` reads `docs/doc_01.txt` through `docs/doc_08.txt`.
- Each supplied policy document is used as one chunk, which is allowed because the documents are short.
- `SentenceTransformer("all-MiniLM-L6-v2")` creates local embeddings.
- ChromaDB stores the vectors and document IDs in the `zepto_policies` collection.
- `graph.py` creates the LangGraph `StateGraph`.
- `classify_intent` uses the required keyword heuristic in mock mode.
- Policy queries go to `retrieve_and_answer`, which embeds the query and retrieves the top three chunks using cosine similarity.
- The mock generation step returns an answer beginning with `Based on the retrieved context:`.
- General queries go to `direct_answer`, which returns the fixed mock response.
- `SupportResponse` validates `answer`, `sources`, and `confidence`.
- `main.py` exposes the graph through FastAPI at `POST /ask`.

## MOCK_LLM

The default is mock mode:

```text
MOCK_LLM unset or MOCK_LLM=1
```

No LLM API is called. Classification is keyword-based, retrieval is real, and generation is deterministic.

The optional path:

```text
MOCK_LLM=0
```

uses a Groq-compatible OpenAI endpoint when `GROQ_API_KEY` is supplied. This path is not required for the graded baseline.

## Setup

From `support_assistant/`:

```bash
pip install -r requirements.txt
python ingest.py
uvicorn main:app --reload
```

Then open another terminal.

### Example 1 — policy query

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"How long does delivery take?\"}"
```

Expected shape:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials ...",
  "sources": ["doc_01", "doc_04", "doc_06"],
  "confidence": 1.0
}
```

The exact ordering of secondary retrieved sources can depend on the local embedding/index version.

### Example 2 — general query

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the capital of India?\"}"
```

Expected:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Docker

Build:

```bash
docker build -t zepto-support .
```

Run:

```bash
docker run --rm -p 7860:7860 zepto-support
```

The API is then available at:

```text
http://127.0.0.1:7860/ask
```

The Dockerfile builds the local Chroma index during image construction, so the container can run without an LLM API key.
