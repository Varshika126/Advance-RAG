"""
vector_store.py
---------------
Manages ChromaDB vector store operations: creation, insertion, deletion, and querying.
"""

import os

# Disable ChromaDB telemetry before importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from typing import List, Dict, Any, Optional
import chromadb

COLLECTION_NAME = "rag_documents"


def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """Create or connect to a persistent ChromaDB client."""
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get an existing ChromaDB collection or create it. Uses cosine similarity."""
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks_to_collection(
    collection: chromadb.Collection,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    """Add document chunks and their embeddings to the collection."""
    if not chunks or not embeddings:
        return 0

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "upload_time": chunk["upload_time"],
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(ids)


def delete_document_from_collection(
    collection: chromadb.Collection, filename: str
) -> int:
    """Delete all chunks belonging to a specific document."""
    results = collection.get(where={"filename": filename})
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def get_collection_stats(collection: chromadb.Collection) -> Dict[str, Any]:
    """Return basic statistics about the collection."""
    total = collection.count()
    if total == 0:
        return {"total_chunks": 0, "documents": []}

    results = collection.get(include=["metadatas"])
    filenames = list(
        {meta["filename"] for meta in results.get("metadatas", []) if meta}
    )
    return {"total_chunks": total, "documents": sorted(filenames)}


def query_collection(
    collection: chromadb.Collection,
    query_embedding: List[float],
    top_k: int = 5,
    where: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Query the collection for the most similar chunks."""
    count = collection.count()
    if count == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def clear_collection(client: chromadb.PersistentClient) -> None:
    """Delete and recreate the collection, clearing all data."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
