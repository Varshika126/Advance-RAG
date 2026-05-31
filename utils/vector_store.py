"""
vector_store.py
---------------
Manages ChromaDB vector store operations: creation, insertion, deletion, and querying.
"""

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings


COLLECTION_NAME = "rag_documents"


def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """
    Create or connect to a persistent ChromaDB client.

    Args:
        persist_dir: Directory path where ChromaDB stores its data.

    Returns:
        A ChromaDB PersistentClient instance.
    """
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Get an existing ChromaDB collection or create it if it doesn't exist.

    Uses cosine similarity for distance metric.
    """
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
    """
    Add document chunks and their embeddings to the ChromaDB collection.

    Args:
        collection: ChromaDB collection object
        chunks: List of chunk dicts from chunker.py
        embeddings: Corresponding embedding vectors

    Returns:
        Number of chunks successfully added.
    """
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

    # ChromaDB upsert handles duplicates gracefully
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
    """
    Delete all chunks belonging to a specific document from the collection.

    Args:
        collection: ChromaDB collection object
        filename: Name of the document to remove

    Returns:
        Number of chunks deleted.
    """
    # Query to find all chunk IDs for this document
    results = collection.get(where={"filename": filename})
    ids_to_delete = results.get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return len(ids_to_delete)


def get_collection_stats(collection: chromadb.Collection) -> Dict[str, Any]:
    """
    Return basic statistics about the collection.

    Returns:
        Dict with total_chunks and unique document filenames.
    """
    total = collection.count()
    if total == 0:
        return {"total_chunks": 0, "documents": []}

    # Fetch all metadata to extract unique filenames
    results = collection.get(include=["metadatas"])
    filenames = list(
        {meta["filename"] for meta in results.get("metadatas", []) if meta}
    )

    return {
        "total_chunks": total,
        "documents": sorted(filenames),
    }


def query_collection(
    collection: chromadb.Collection,
    query_embedding: List[float],
    top_k: int = 5,
    where: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Query the collection for the most similar chunks.

    Args:
        collection: ChromaDB collection object
        query_embedding: Embedding vector for the query
        top_k: Number of results to return
        where: Optional metadata filter

    Returns:
        ChromaDB query results dict.
    """
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()) if collection.count() > 0 else 1,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


def clear_collection(client: chromadb.PersistentClient) -> None:
    """
    Delete and recreate the collection, effectively clearing all data.
    """
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection may not exist yet
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
