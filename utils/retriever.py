"""
retriever.py
------------
Handles retrieval strategies: Similarity Search and MMR (Maximal Marginal Relevance).
"""

import math
from typing import List, Dict, Any, Tuple

import numpy as np


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_search(
    query_results: Dict[str, Any],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return top-k results from ChromaDB query results, sorted by similarity.

    ChromaDB returns cosine distances (0 = identical, 2 = opposite).
    We convert to similarity score: similarity = 1 - (distance / 2).

    Args:
        query_results: Raw results from vector_store.query_collection()
        top_k: Maximum number of results to return

    Returns:
        List of result dicts with text, metadata, and similarity_score.
    """
    documents = query_results.get("documents", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]
    distances = query_results.get("distances", [[]])[0]

    results = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert cosine distance to similarity score (0–1 range)
        similarity = max(0.0, 1.0 - (dist / 2.0))
        results.append(
            {
                "text": doc,
                "filename": meta.get("filename", "Unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "upload_time": meta.get("upload_time", ""),
                "similarity_score": round(similarity, 4),
                "distance": round(dist, 4),
            }
        )

    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]


def mmr_search(
    query_results: Dict[str, Any],
    query_embedding: List[float],
    top_k: int = 5,
    lambda_mult: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Apply Maximal Marginal Relevance (MMR) to diversify retrieved results.

    MMR balances relevance to the query with diversity among selected chunks.
    Higher lambda_mult = more relevance-focused.
    Lower lambda_mult = more diversity-focused.

    Args:
        query_results: Raw results from vector_store.query_collection()
        query_embedding: The query's embedding vector
        top_k: Number of results to return
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        Diversified list of result dicts.
    """
    documents = query_results.get("documents", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]
    distances = query_results.get("distances", [[]])[0]

    if not documents:
        return []

    # Build candidate list with similarity scores
    candidates = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = max(0.0, 1.0 - (dist / 2.0))
        candidates.append(
            {
                "text": doc,
                "filename": meta.get("filename", "Unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "upload_time": meta.get("upload_time", ""),
                "similarity_score": round(similarity, 4),
                "distance": round(dist, 4),
                "query_similarity": similarity,
            }
        )

    if len(candidates) <= top_k:
        return candidates

    # MMR selection loop
    selected = []
    remaining = list(range(len(candidates)))

    # Start with the most relevant candidate
    best_idx = max(remaining, key=lambda i: candidates[i]["query_similarity"])
    selected.append(best_idx)
    remaining.remove(best_idx)

    while len(selected) < top_k and remaining:
        mmr_scores = []
        for i in remaining:
            relevance = candidates[i]["query_similarity"]
            # Max similarity to already-selected candidates (text-based approximation)
            max_redundancy = max(
                _text_overlap_similarity(candidates[i]["text"], candidates[j]["text"])
                for j in selected
            )
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_redundancy
            mmr_scores.append((i, mmr_score))

        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def _text_overlap_similarity(text_a: str, text_b: str) -> float:
    """
    Simple word-overlap Jaccard similarity used as a proxy for MMR redundancy.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def format_retrieved_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.

    Args:
        chunks: List of chunk dicts from similarity_search or mmr_search

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['filename']} | Score: {chunk['similarity_score']:.2%}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)
