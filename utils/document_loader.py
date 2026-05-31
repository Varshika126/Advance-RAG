"""
document_loader.py
------------------
Handles loading and text extraction from PDF, DOCX, and TXT files.
"""

import os
import io
from datetime import datetime
from typing import List, Dict, Any

import pypdf
from docx import Document


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    """Extract all text from a PDF file given its raw bytes."""
    text = ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_num + 1}]\n{page_text}"
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF '{filename}': {e}")
    return text.strip()


def extract_text_from_docx(file_bytes: bytes, filename: str) -> str:
    """Extract all text from a DOCX file given its raw bytes."""
    text = ""
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX '{filename}': {e}")
    return text.strip()


def extract_text_from_txt(file_bytes: bytes, filename: str) -> str:
    """Extract text from a plain TXT file given its raw bytes."""
    try:
        # Try UTF-8 first, fall back to latin-1
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
    except Exception as e:
        raise ValueError(f"Failed to extract text from TXT '{filename}': {e}")
    return text.strip()


def load_document(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Load a document and extract its text based on file extension.

    Returns a dict with:
        - filename: original file name
        - text: extracted text content
        - upload_time: ISO timestamp of when it was processed
        - file_size: size in bytes
        - extension: file type
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_bytes, filename)
    elif ext == ".docx":
        text = extract_text_from_docx(file_bytes, filename)
    elif ext == ".txt":
        text = extract_text_from_txt(file_bytes, filename)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: PDF, DOCX, TXT."
        )

    if not text:
        raise ValueError(
            f"No text could be extracted from '{filename}'. The file may be empty or image-based."
        )

    return {
        "filename": filename,
        "text": text,
        "upload_time": datetime.utcnow().isoformat(),
        "file_size": len(file_bytes),
        "extension": ext,
    }


def validate_file(filename: str, file_bytes: bytes, max_size_mb: int = 50) -> None:
    """
    Validate a file before processing.
    Raises ValueError with a descriptive message on failure.
    """
    ext = os.path.splitext(filename)[1].lower()
    allowed_extensions = {".pdf", ".docx", ".txt"}

    if ext not in allowed_extensions:
        raise ValueError(
            f"'{filename}' has unsupported extension '{ext}'. "
            f"Allowed: {', '.join(allowed_extensions)}"
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the {max_size_mb} MB limit."
        )

    if len(file_bytes) == 0:
        raise ValueError(f"'{filename}' is empty.")
