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
.\\.venv\\Scripts\\Activate.ps1

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

## Usage examples (copied from Vector_retriever.ipynb)

Below are short, copy-pasteable snippets taken directly from the notebook to help you get started quickly.

1) Load environment variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

# optional: push env vars into os.environ for downstream libraries
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
```

2) Create LangChain Document objects

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# inspect documents
documents
```

3) Initialize a Groq LLM client (ChatGroq)

```python
from langchain_groq import ChatGroq

llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model="Llama3-8b-8192")
llm
```

4) Initialize HuggingFace-based embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

Notes from the notebook:
- One of the notebook cells shows a NameError when importing/initializing HuggingFace embeddings; this often indicates a missing or incompatible backend (sentence-transformers/transformers/torch). If you encounter that error, try:

```bash
pip install --upgrade sentence-transformers transformers torch
```

- The notebook demonstrates the RAG pattern (Documents -> Embeddings -> Vector store -> Retriever -> LLM). Some of the examples assume external API access (Groq) — if you don't have those keys, you can still run the embedding and vector-store parts locally with open models.

## Next steps

If you'd like, I can now:
- Add runnable code showing how to load these Document objects into Chromadb and run a retrieval query, or
- Extract the relevant notebook cells into a small script (retriever_example.py) that runs end-to-end on a local CPU embedding model.

---
