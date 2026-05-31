"""
chunker.py
----------
Splits extracted document text into overlapping chunks suitable for embedding.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(
    doc: Dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split a document's text into chunks.

    Args:
        doc: Document dict from document_loader.load_document()
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between consecutive chunks

    Returns:
        List of chunk dicts, each containing:
            - chunk_id: unique identifier (filename + index)
            - text: chunk text content
            - filename: source document name
            - upload_time: from original document
            - chunk_index: position within the document
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(doc["text"])
    chunks = []

    for idx, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        chunks.append(
            {
                "chunk_id": f"{doc['filename']}::chunk_{idx}",
                "text": chunk_text,
                "filename": doc["filename"],
                "upload_time": doc["upload_time"],
                "chunk_index": idx,
            }
        )

    return chunks


def chunk_documents(
    docs: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Chunk a list of documents.

    Returns a flat list of all chunks across all documents.
    """
    all_chunks = []
    for doc in docs:
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(doc_chunks)
    return all_chunks
