# 🧠 Advanced Multi-Document RAG System

A production-ready Retrieval Augmented Generation (RAG) application built with **Streamlit**, **ChromaDB**, **LangChain**, and **OpenAI / Google Gemini**.

---

## Features

- Upload up to **50 documents** (PDF, DOCX, TXT)
- Automatic text extraction, chunking, and embedding
- **ChromaDB** persistent vector storage — survives restarts
- Two retrieval strategies: **Similarity Search** and **MMR (Maximal Marginal Relevance)**
- Configurable **Top-K** retrieval
- Strict RAG prompt — answers come **only** from your documents
- Source attribution with **similarity scores** for every answer
- Full **conversation history** with per-answer source citations
- Modern, dark-themed Streamlit UI

---

## Project Structure

```
project/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Your API keys (create from .env.example)
├── .env.example            # Template for environment variables
├── README.md               # This file
├── chroma_db/              # ChromaDB persistent storage (auto-created)
├── uploads/                # Upload staging directory (auto-created)
└── utils/
    ├── __init__.py
    ├── document_loader.py  # PDF / DOCX / TXT text extraction
    ├── chunker.py          # Recursive text splitting
    ├── vector_store.py     # ChromaDB CRUD operations
    ├── retriever.py        # Similarity search & MMR
    └── rag_chain.py        # Embeddings + LLM answer generation
```

---

## Quick Start

### 1. Clone / download the project

```bash
cd project
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
# Choose: openai  OR  gemini
LLM_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini (if using Gemini)
GOOGLE_API_KEY=AIza...
```

### 5. Run the application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Usage

1. **Upload documents** — use the sidebar file uploader (PDF, DOCX, or TXT).
2. **Click "Process & Index Files"** — the app extracts text, chunks it, generates embeddings, and stores them in ChromaDB.
3. **Ask a question** — type in the main chat area and click "Get Answer".
4. **View sources** — expand the "Sources" section under each answer to see which chunks were used and their similarity scores.
5. **Switch retrieval mode** — choose between Similarity Search and MMR in the sidebar.
6. **Clear data** — use "Clear All Documents" to wipe the database, or delete individual files.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `OPENAI_API_KEY` | — | Required for OpenAI |
| `GOOGLE_API_KEY` | — | Required for Gemini |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `UPLOADS_DIR` | `./uploads` | Upload staging path |

Chunk settings (in `app.py`):

| Constant | Value |
|---|---|
| `CHUNK_SIZE` | 1000 characters |
| `CHUNK_OVERLAP` | 200 characters |
| `MAX_FILES` | 50 |

---

## RAG Workflow

```
Upload → Extract Text → Chunk → Embed → Store in ChromaDB
                                                  ↓
Question → Embed Query → Retrieve Top-K Chunks → LLM → Answer + Sources
```

---

## Notes

- The LLM is instructed to answer **only** from the provided context. If the answer is not in your documents, it returns: *"Answer not found in the uploaded documents."*
- ChromaDB data persists in `./chroma_db/` — your indexed documents survive application restarts.
- The app tracks which files have been indexed in Streamlit session state. After a restart, re-upload files to re-populate the session (the ChromaDB data is still there).
