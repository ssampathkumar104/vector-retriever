# vector-retriever

A hands-on demonstration of vector stores and retrievers for use with LLM workflows. This repository contains a Jupyter notebook that walks through document creation, embeddings, vector stores, and retrieval patterns using LangChain integrations and popular embedding/LLM backends.

## Introduction

Modern LLM applications often need to retrieve relevant text or documents from large corpora. Vector retrieval solves this by converting text into dense embeddings and using nearest-neighbor search to find semantically similar documents. This project provides a compact, runnable example (in a Jupyter notebook) that shows how to:

- Represent text as LangChain Document objects with metadata;
- Create embeddings (example uses HuggingFace-based embeddings);
- Put embeddings into a vector store (Chromadb is listed as a dependency);
- Query the vector store with a retriever and use the retrieved context with an LLM such as Groq's ChatGroq.

This is intended for developers and researchers who want a minimal, reproducible example of a retrieval-augmented generation (RAG) pattern using Python and LangChain components.

## What you'll find in this repo

- Vector_retriever.ipynb — The primary, annotated Jupyter notebook that contains the walkthrough and code examples.
- requirements.txt — Python dependencies required to run the notebook.
- .env — Environment variables used by the notebook (API keys). Keep secrets out of public repos; use a local .env or CI secrets.

## Stack / Notable dependencies

- Language(s): Jupyter Notebook / Python
- Runtime: Python 3.10 (notebook metadata shows kernel `venv (3.10.9)`).
- Notable libraries (from requirements.txt):
  - langchain-core / langchain-community — core abstractions for Documents, retrievers, and vector store integration
  - langchain-groq — example LLM integration (ChatGroq)
  - langchain_huggingface / sentence-transformers — embeddings from HuggingFace models
  - chromadb — an example vector store backend
  - python-dotenv — load environment variables from a .env file

## How to run the notebook (quickstart)

1. Clone the repo and enter the directory:

```bash
git clone https://github.com/ssampathkumar104/vector-retriever.git
cd vector-retriever
```

2. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

3. Provide required API keys in a local .env file (or export them in your shell). The notebook expects at least the following environment variables:

- HF_TOKEN — (optional) Hugging Face token for private models or APIs
- GROQ_API_KEY — API key for Groq Chat (used by the ChatGroq example)

Example .env (do NOT commit secrets):

```text
HF_TOKEN=your_hf_token_here
GROQ_API_KEY=your_groq_api_key_here
```

4. Launch Jupyter and open the notebook:

```bash
jupyter lab
# or
jupyter notebook Vector_retriever.ipynb
```

The notebook cells are annotated and demonstrate creating Document objects, creating embeddings, initializing an LLM client (ChatGroq), and running retrieval queries.

## Notes & troubleshooting

- Notebook kernel: the notebook metadata specifies Python 3.10.9 — use a compatible interpreter or the virtual environment above.
- One cell in the notebook imports `HuggingFaceEmbeddings` and shows a NameError related to `nn` not being defined. This typically indicates a missing or incompatible dependency in the sentence-transformers / transformers stack. If you see this error, ensure `sentence-transformers` is installed (requirements.txt lists it) and try upgrading related packages:

```bash
pip install --upgrade sentence-transformers transformers torch
```

- The notebook demonstrates using Groq and HuggingFace integrations; if you don't have API keys or access, you can still run the document/embedding/vector-store cells locally by using open, CPU-based Hugging Face embedding models (the repo already references `all-MiniLM-L6-v2` in the notebook as an example).

## Next steps you might try

- Convert the notebook into a small script or package that loads documents from disk and exposes a simple REST API for retrieval.
- Swap Chromadb for another vector store (Weaviate, Pinecone, or a simple FAISS-based local store) and compare latency and accuracy.
- Add automated tests that verify embedding + retrieval correctness for a small sample corpus.

---

If you'd like, I can also:
- Expand the README with usage examples copied directly from the notebook (code snippets), or
- Open a small patch that converts the notebook into a runnable script with CLI flags.
