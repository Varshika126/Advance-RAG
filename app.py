"""
app.py
------
Advanced Multi-Document RAG Application
Built with Streamlit + ChromaDB + LangChain + OpenAI/Gemini

Run with:
    streamlit run app.py
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file (local dev)
load_dotenv()

# On Streamlit Cloud, secrets are in st.secrets — merge them into os.environ
try:
    import streamlit as _st_secrets_check
    for _key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER",
                 "CHROMA_PERSIST_DIR", "UPLOADS_DIR"]:
        if _key in _st_secrets_check.secrets and not os.getenv(_key):
            os.environ[_key] = _st_secrets_check.secrets[_key]
except Exception:
    pass

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Multi-Document RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local utility imports
# ---------------------------------------------------------------------------
from utils.document_loader import load_document, validate_file
from utils.chunker import chunk_documents
from utils.vector_store import (
    get_chroma_client,
    get_or_create_collection,
    add_chunks_to_collection,
    delete_document_from_collection,
    get_collection_stats,
    query_collection,
    clear_collection,
)
from utils.retriever import similarity_search, mmr_search, format_retrieved_chunks
from utils.rag_chain import embed_texts, embed_query, generate_answer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILES = 50
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./uploads")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Custom CSS for a professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ---- Header ---- */
    .rag-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #1e3a5f;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .rag-header h1 {
        color: #e2e8f0;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .rag-header p {
        color: #94a3b8;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
    }

    /* ---- Status cards ---- */
    .status-card {
        background: #1e2433;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .status-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .status-card .value {
        color: #63b3ed;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .status-card .sub {
        color: #68d391;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }

    /* ---- Answer box ---- */
    .answer-box {
        background: linear-gradient(135deg, #1a2744 0%, #1e2d4a 100%);
        border-left: 4px solid #4299e1;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.7;
    }
    .not-found-box {
        background: linear-gradient(135deg, #2d1b1b 0%, #3d2020 100%);
        border-left: 4px solid #fc8181;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #fed7d7;
        font-size: 1rem;
    }

    /* ---- Source card ---- */
    .source-card {
        background: #1a2035;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .source-filename {
        color: #63b3ed;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .score-badge {
        background: #2d4a6e;
        color: #90cdf4;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .source-text {
        color: #a0aec0;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* ---- Chat history ---- */
    .chat-user {
        background: #1e2d4a;
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
        border: 1px solid #2d4a6e;
    }
    .chat-assistant {
        background: #1a2035;
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
        border: 1px solid #2d3748;
    }

    /* ---- Sidebar ---- */
    .sidebar-section {
        background: #1e2433;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #2d3748;
    }
    .sidebar-section h4 {
        color: #90cdf4;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.8rem 0;
    }

    /* ---- File list ---- */
    .file-item {
        background: #252d3d;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        color: #cbd5e0;
        font-size: 0.82rem;
        border: 1px solid #2d3748;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- Misc ---- */
    div[data-testid="stMetricValue"] { color: #63b3ed !important; }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "chat_history": [],          # List of {role, content, sources, timestamp}
        "processed_files": {},       # filename -> {chunks, upload_time, size}
        "total_queries": 0,
        "chroma_client": None,
        "collection": None,
        "provider": LLM_PROVIDER,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_db():
    """Lazily initialise and cache the ChromaDB client and collection."""
    if st.session_state.chroma_client is None:
        st.session_state.chroma_client = get_chroma_client(CHROMA_PERSIST_DIR)
        st.session_state.collection = get_or_create_collection(
            st.session_state.chroma_client
        )
    return st.session_state.chroma_client, st.session_state.collection


# ---------------------------------------------------------------------------
# Helper: process and index uploaded files
# ---------------------------------------------------------------------------
def process_and_index_files(uploaded_files, provider: str):
    """
    Validate, extract, chunk, embed, and store uploaded files.
    Returns (success_count, error_messages).
    """
    _, collection = get_db()
    success_count = 0
    errors = []

    progress_bar = st.progress(0, text="Processing documents…")
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()

        try:
            # 1. Validate
            validate_file(filename, file_bytes)

            # 2. Skip already-processed files
            if filename in st.session_state.processed_files:
                progress_bar.progress((i + 1) / total, text=f"Skipping duplicate: {filename}")
                continue

            progress_bar.progress((i + 1) / total, text=f"Loading: {filename}")

            # 3. Extract text
            doc = load_document(file_bytes, filename)

            # 4. Chunk
            chunks = chunk_documents([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            if not chunks:
                errors.append(f"⚠️ No text chunks extracted from '{filename}'.")
                continue

            # 5. Embed
            progress_bar.progress((i + 1) / total, text=f"Embedding: {filename}")
            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(texts, provider=provider)

            # 6. Store in ChromaDB
            added = add_chunks_to_collection(collection, chunks, embeddings)

            # 7. Track in session state
            st.session_state.processed_files[filename] = {
                "chunks": added,
                "upload_time": doc["upload_time"],
                "size": doc["file_size"],
                "extension": doc["extension"],
            }

            success_count += 1

        except Exception as e:
            errors.append(f"❌ {filename}: {e}")

    progress_bar.empty()
    return success_count, errors


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 RAG Control Panel")

        # ── LLM Provider ──────────────────────────────────────────────────
        st.markdown("### ⚙️ LLM Provider")
        provider = st.selectbox(
            "Select Provider",
            options=["openai", "gemini"],
            index=0 if st.session_state.provider == "openai" else 1,
            key="provider_select",
        )
        st.session_state.provider = provider

        # ── Document Upload ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📁 Document Upload")

        current_count = len(st.session_state.processed_files)
        remaining = MAX_FILES - current_count
        st.caption(f"Indexed: **{current_count}/{MAX_FILES}** documents")

        uploaded_files = st.file_uploader(
            f"Drop files here (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="file_uploader",
            help="Files are automatically processed after upload",
        )

        # Auto-process: detect newly uploaded files not yet indexed
        if uploaded_files:
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.processed_files
            ]
            if current_count + len(new_files) > MAX_FILES:
                st.error(f"Limit reached. You can add {remaining} more file(s).")
            elif new_files:
                with st.status(f"Processing {len(new_files)} file(s)…", expanded=True) as status:
                    success, errors = process_and_index_files(
                        new_files, provider=st.session_state.provider
                    )
                    if success:
                        status.update(
                            label=f"✅ Indexed {success} file(s) successfully!",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status.update(label="⚠️ Processing finished with errors.", state="error")
                for err in errors:
                    st.warning(err)
                st.rerun()

        # ── Uploaded File List ────────────────────────────────────────────
        if st.session_state.processed_files:
            st.markdown("---")
            st.markdown("### 📋 Indexed Documents")

            for fname, info in list(st.session_state.processed_files.items()):
                ext_icon = {"pdf": "📄", "docx": "📝", "txt": "📃"}.get(
                    info["extension"].lstrip("."), "📎"
                )
                col1, col2 = st.columns([4, 1])
                with col1:
                    size_kb = info["size"] / 1024
                    st.markdown(
                        f"<div class='file-item'>{ext_icon} {fname[:28]}{'…' if len(fname)>28 else ''}"
                        f"<br><small style='color:#718096'>{info['chunks']} chunks · {size_kb:.1f} KB</small></div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                        _, collection = get_db()
                        deleted = delete_document_from_collection(collection, fname)
                        del st.session_state.processed_files[fname]
                        st.success(f"Removed '{fname}' ({deleted} chunks).")
                        st.rerun()

        # ── Retrieval Settings ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Retrieval Settings")

        search_type = st.selectbox(
            "Search Type",
            options=["Similarity Search", "MMR (Diversified)"],
            index=0,
            key="search_type",
        )

        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="top_k",
            help="Number of document chunks to retrieve",
        )

        if search_type == "MMR (Diversified)":
            lambda_mult = st.slider(
                "MMR Lambda (Relevance ↔ Diversity)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key="lambda_mult",
                help="1.0 = pure relevance, 0.0 = pure diversity",
            )
        else:
            lambda_mult = 0.5

        # ── Database Management ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🗄️ Database Management")

        if st.button("🧹 Clear All Documents", use_container_width=True, type="secondary"):
            client, _ = get_db()
            clear_collection(client)
            # Refresh collection reference
            st.session_state.collection = get_or_create_collection(client)
            st.session_state.processed_files = {}
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.success("Database cleared.")
            st.rerun()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        return search_type, top_k, lambda_mult


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def render_main(search_type: str, top_k: int, lambda_mult: float):
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="rag-header">
            <h1>🧠 Advanced Multi-Document RAG System</h1>
            <p>Ask questions across your uploaded documents — answers are grounded strictly in your content.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Status cards ──────────────────────────────────────────────────────
    _, collection = get_db()
    stats = get_collection_stats(collection)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Documents Loaded", len(st.session_state.processed_files))
    with col2:
        st.metric("🧩 Chunks Indexed", stats["total_chunks"])
    with col3:
        st.metric("💬 Total Queries", st.session_state.total_queries)
    with col4:
        db_status = "🟢 Ready" if stats["total_chunks"] > 0 else "🟡 Empty"
        st.metric("🗄️ Database Status", db_status)

    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        for entry in st.session_state.chat_history:
            if entry["role"] == "user":
                st.markdown(
                    f"<div class='chat-user'>🧑 <strong>You:</strong> {entry['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                not_found = "Answer not found in the uploaded documents." in entry["content"]
                box_class = "not-found-box" if not_found else "answer-box"
                st.markdown(
                    f"<div class='{box_class}'>🤖 <strong>Assistant:</strong><br>{entry['content']}</div>",
                    unsafe_allow_html=True,
                )
                # Show sources in expander
                if entry.get("sources"):
                    with st.expander(f"📚 Sources ({len(entry['sources'])} chunks)", expanded=False):
                        for j, src in enumerate(entry["sources"], 1):
                            score_pct = f"{src['similarity_score']:.1%}"
                            st.markdown(
                                f"""
                                <div class='source-card'>
                                    <div class='source-header'>
                                        <span class='source-filename'>📄 {src['filename']}</span>
                                        <span class='score-badge'>Score: {score_pct}</span>
                                    </div>
                                    <div class='source-text'>{src['text'][:400]}{'…' if len(src['text'])>400 else ''}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
        st.markdown("---")

    # ── Question input ────────────────────────────────────────────────────
    st.markdown("### ❓ Ask a Question")

    if stats["total_chunks"] == 0:
        st.info("📂 Upload and index documents using the sidebar to get started.")

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_area(
            "Enter your question",
            placeholder="e.g. What are the main findings of the report?",
            height=100,
            key="question_input",
        )
        submitted = st.form_submit_button("🔍 Get Answer", use_container_width=True, type="primary")

    if submitted and question.strip():
        if stats["total_chunks"] == 0:
            st.warning("No documents indexed yet. Please upload files first.")
            return

        with st.spinner("🔍 Searching documents and generating answer…"):
            try:
                # 1. Embed the query
                q_embedding = embed_query(question.strip(), provider=st.session_state.provider)

                # 2. Retrieve from ChromaDB
                raw_results = query_collection(
                    collection,
                    query_embedding=q_embedding,
                    top_k=max(top_k * 2, 10),  # over-fetch for MMR
                )

                # 3. Apply retrieval strategy
                if search_type == "MMR (Diversified)":
                    chunks = mmr_search(
                        raw_results,
                        query_embedding=q_embedding,
                        top_k=top_k,
                        lambda_mult=lambda_mult,
                    )
                else:
                    chunks = similarity_search(raw_results, top_k=top_k)

                # 4. Format context
                context = format_retrieved_chunks(chunks)

                # 5. Generate answer
                answer = generate_answer(
                    question=question.strip(),
                    context=context,
                    provider=st.session_state.provider,
                )

                # 6. Update session state
                st.session_state.total_queries += 1
                st.session_state.chat_history.append(
                    {"role": "user", "content": question.strip(), "sources": None, "timestamp": datetime.utcnow().isoformat()}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": chunks, "timestamp": datetime.utcnow().isoformat()}
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                return

        st.rerun()

    elif submitted and not question.strip():
        st.warning("Please enter a question.")

    # ── Latest answer (shown immediately after submit before rerun) ───────
    # (handled by chat history rerun above)

    # ── Analytics expander ────────────────────────────────────────────────
    if st.session_state.processed_files:
        with st.expander("📊 Document Analytics", expanded=False):
            st.markdown("#### Indexed Documents")
            for fname, info in st.session_state.processed_files.items():
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.write(f"📄 {fname}")
                with col_b:
                    st.write(f"{info['chunks']} chunks")
                with col_c:
                    st.write(f"{info['size']/1024:.1f} KB")

            st.markdown("#### Database Summary")
            st.json(
                {
                    "total_documents": len(st.session_state.processed_files),
                    "total_chunks": stats["total_chunks"],
                    "total_queries": st.session_state.total_queries,
                    "chroma_persist_dir": CHROMA_PERSIST_DIR,
                    "llm_provider": st.session_state.provider,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                }
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    init_session_state()
    search_type, top_k, lambda_mult = render_sidebar()
    render_main(search_type, top_k, lambda_mult)


if __name__ == "__main__":
    main()
