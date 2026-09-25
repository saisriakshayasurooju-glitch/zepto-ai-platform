# Zepto Data & AI Platform Capstone

This repository contains the three required modules from the capstone brief:

- `data_pipeline/` — web scraping, cleaning, SQLite schema, SQL queries and Pandas equivalence checks.
- `analytics/` — Titanic EDA, visualization, correlation analysis, classification, imbalance handling, RF tuning, fare regression and model persistence.
- `support_assistant/` — offline RAG-style support assistant using Sentence Transformers, ChromaDB, LangGraph, Pydantic and FastAPI.

## Repository structure

```text
zepto-ai-platform/
├── README.md
├── requirements.txt
├── data_pipeline/
│   ├── pipeline.py
│   ├── queries.sql
│   └── README.md
├── analytics/
│   ├── analysis.py
│   └── README.md
└── support_assistant/
    ├── docs/
    │   ├── doc_01.txt ... doc_08.txt
    ├── ingest.py
    ├── graph.py
    ├── main.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

## Quick start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Then run each module as described in its README.

## Git history requirement

The brief requires at least one feature branch, at least two commits on that branch, and a merge back into `main`.

Example:

```bash
git init
git add .
git commit -m "Initial project structure"

git switch -c feature/capstone-modules
git add .
git commit -m "Add data pipeline and analytics"
git add .
git commit -m "Add support assistant"

git switch main
git merge feature/capstone-modules
git log --oneline --graph --all
```

## Academic note

This repository is an implementation scaffold generated from the supplied capstone specification. Run the scripts, inspect the outputs, and make sure the final repository reflects work you understand and can explain.
