"""
rag_chain.py
------------
Handles embedding generation and LLM answer generation for the RAG pipeline.
"""

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_embedding_model(provider: str = "openai"):
    """
    Return the appropriate LangChain embedding model based on provider.

    Args:
        provider: "openai" or "gemini"

    Returns:
        LangChain embeddings object.
    """
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables.")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
            task_type="retrieval_document",
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )


def embed_texts(texts: List[str], provider: str = "openai") -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text strings to embed
        provider: "openai" or "gemini"

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    embedding_model = get_embedding_model(provider)
    embeddings = embedding_model.embed_documents(texts)
    return embeddings


def embed_query(query: str, provider: str = "openai") -> List[float]:
    """
    Generate an embedding for a single query string.

    Args:
        query: The question or search query
        provider: "openai" or "gemini"

    Returns:
        Embedding vector as a list of floats.
    """
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
            task_type="retrieval_query",
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
    return embedding_model.embed_query(query)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

# Strict RAG prompt — the LLM must only use the provided context
RAG_PROMPT_TEMPLATE = """You are a document question-answering assistant.
Answer ONLY using the provided context below.
If the answer is not present in the context, respond exactly with:
"Answer not found in the uploaded documents."
Do not use external knowledge. Do not make assumptions. Do not hallucinate.

Context:
{context}

Question: {question}

Answer:"""


def get_llm(provider: str = "openai"):
    """
    Return the appropriate LangChain LLM based on provider.

    Args:
        provider: "openai" or "gemini"

    Returns:
        LangChain LLM object.
    """
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set in environment variables.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
        )
    else:
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0,
        )


def generate_answer(
    question: str,
    context: str,
    provider: str = "openai",
) -> str:
    """
    Generate an answer using the LLM strictly from the provided context.

    Args:
        question: User's question
        context: Retrieved document context
        provider: "openai" or "gemini"

    Returns:
        LLM-generated answer string.
    """
    if not context.strip():
        return "Answer not found in the uploaded documents."

    llm = get_llm(provider)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        response = llm.invoke(prompt)
        # LangChain chat models return an AIMessage object
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {e}")
