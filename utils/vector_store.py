"""
vector_store.py
---------------
Pure-Python persistent vector store using numpy + JSON.
No chromadb, no compiled dependencies — works on any Python version.

Storage layout (inside persist_dir):
    metadata.json   — list of chunk metadata dicts
    embeddings.npy  — float32 array of shape (N, dim)
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _meta_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "metadata.json")

def _emb_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "embeddings.npy")


def _load_store(persist_dir: str):
    """Load metadata list and embeddings array from disk. Returns (meta, emb)."""
    meta_file = _meta_path(persist_dir)
    emb_file = _emb_path(persist_dir)

    if os.path.exists(meta_file) and os.path.exists(emb_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        emb = np.load(emb_file).astype(np.float32)
        return meta, emb
    return [], np.empty((0,), dtype=np.float32)


def _save_store(persist_dir: str, meta: list, emb: np.ndarray):
    """Persist metadata and embeddings to disk."""
    os.makedirs(persist_dir, exist_ok=True)
    with open(_meta_path(persist_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    np.save(_emb_path(persist_dir), emb.astype(np.float32))


def _cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and each row of matrix."""
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return np.zeros(len(matrix))
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms[m_norms == 0] = 1e-10
    return (matrix @ query) / (m_norms * q_norm)


# ---------------------------------------------------------------------------
# Public API  (mirrors the old chromadb-based interface)
# ---------------------------------------------------------------------------

class SimpleVectorStore:
    """
    Lightweight persistent vector store backed by numpy + JSON files.
    Drop-in replacement for the chromadb collection interface used in this app.
    """

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self._meta, self._emb = _load_store(persist_dir)

    # ---- write ----

    def upsert(self, ids, embeddings, documents, metadatas):
        """Add or update chunks by chunk_id."""
        existing_ids = {m["chunk_id"]: i for i, m in enumerate(self._meta)}
        new_meta = list(self._meta)
        new_emb_list = list(self._emb) if len(self._emb) > 0 else []

        for chunk_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            record = {**meta, "chunk_id": chunk_id, "text": doc}
            vec = np.array(emb, dtype=np.float32)
            if chunk_id in existing_ids:
                idx = existing_ids[chunk_id]
                new_meta[idx] = record
                new_emb_list[idx] = vec
            else:
                new_meta.append(record)
                new_emb_list.append(vec)

        self._meta = new_meta
        self._emb = np.array(new_emb_list, dtype=np.float32) if new_emb_list else np.empty((0,), dtype=np.float32)
        _save_store(self.persist_dir, self._meta, self._emb)

    def delete(self, ids=None, where=None):
        """Delete chunks by id list or metadata filter."""
        if ids is not None:
            id_set = set(ids)
            keep = [i for i, m in enumerate(self._meta) if m["chunk_id"] not in id_set]
        elif where is not None:
            key, val = next(iter(where.items()))
            keep = [i for i, m in enumerate(self._meta) if m.get(key) != val]
        else:
            return

        self._meta = [self._meta[i] for i in keep]
        self._emb = self._emb[keep] if len(self._emb) > 0 and keep else np.empty((0,), dtype=np.float32)
        _save_store(self.persist_dir, self._meta, self._emb)

    # ---- read ----

    def count(self) -> int:
        return len(self._meta)

    def get(self, where=None, include=None):
        """Return all records, optionally filtered by metadata."""
        if where:
            key, val = next(iter(where.items()))
            records = [m for m in self._meta if m.get(key) == val]
        else:
            records = list(self._meta)

        return {
            "ids": [r["chunk_id"] for r in records],
            "metadatas": [{k: v for k, v in r.items() if k not in ("text", "chunk_id")} for r in records],
            "documents": [r["text"] for r in records],
        }

    def query(self, query_embeddings, n_results=5, include=None, where=None):
        """Return top-n most similar chunks to the query embedding."""
        if len(self._meta) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q = np.array(query_embeddings[0], dtype=np.float32)

        if where:
            key, val = next(iter(where.items()))
            indices = [i for i, m in enumerate(self._meta) if m.get(key) == val]
        else:
            indices = list(range(len(self._meta)))

        if not indices:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        sub_emb = self._emb[indices]
        sims = _cosine_similarity_batch(q, sub_emb)

        # Convert similarity to cosine distance (0=identical, 2=opposite)
        distances = 1.0 - sims

        top_n = min(n_results, len(indices))
        top_local = np.argsort(distances)[:top_n]
        top_global = [indices[i] for i in top_local]

        docs, metas, dists = [], [], []
        for gi, li in zip(top_global, top_local):
            m = self._meta[gi]
            docs.append(m["text"])
            metas.append({k: v for k, v in m.items() if k not in ("text", "chunk_id")})
            dists.append(float(distances[li]))

        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }


# ---------------------------------------------------------------------------
# Factory functions (same interface as before)
# ---------------------------------------------------------------------------

# Module-level cache so we don't reload from disk on every call
_store_cache: Optional[SimpleVectorStore] = None


def get_chroma_client(persist_dir: str) -> str:
    """Returns persist_dir as the 'client' (no real client needed)."""
    return persist_dir


def get_or_create_collection(persist_dir: str) -> SimpleVectorStore:
    """Get or create the vector store."""
    global _store_cache
    if _store_cache is None or _store_cache.persist_dir != persist_dir:
        _store_cache = SimpleVectorStore(persist_dir)
    return _store_cache


def add_chunks_to_collection(
    collection: SimpleVectorStore,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    if not chunks or not embeddings:
        return 0
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "upload_time": c["upload_time"],
        }
        for c in chunks
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(ids)


def delete_document_from_collection(
    collection: SimpleVectorStore, filename: str
) -> int:
    results = collection.get(where={"filename": filename})
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def get_collection_stats(collection: SimpleVectorStore) -> Dict[str, Any]:
    total = collection.count()
    if total == 0:
        return {"total_chunks": 0, "documents": []}
    results = collection.get()
    filenames = list({m["filename"] for m in results.get("metadatas", []) if m})
    return {"total_chunks": total, "documents": sorted(filenames)}


def query_collection(
    collection: SimpleVectorStore,
    query_embedding: List[float],
    top_k: int = 5,
    where: Optional[Dict] = None,
) -> Dict[str, Any]:
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )


def clear_collection(persist_dir: str) -> None:
    """Wipe all stored data."""
    global _store_cache
    import shutil
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    _store_cache = SimpleVectorStore(persist_dir)
