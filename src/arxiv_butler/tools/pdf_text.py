"""Tools for extracting text from PDF files.

This module provides LangChain tools for working with *text content* in PDFs.
It uses `pypdf` to extract selectable text; it does not interpret images or figures.
"""

import logging
from pathlib import Path

from langchain.tools import tool
from pypdf import PdfReader


logger = logging.getLogger(__name__)


@tool
def pdf_extract_text(
    file_path: str,
    max_pages: int = 10,
    max_chars: int = 30000,
) -> str:
    """Extract plain text from a local PDF file (text-only).

    Args:
        file_path: Path to a local PDF file. Supports `~`. Relative paths are resolved from the
            current working directory.
        max_pages: Maximum number of pages to read starting from page 1. Must be >= 1.
        max_chars: Maximum number of characters to return across all extracted pages. Must be >= 1.

    Returns:
        A single string containing a header (resolved path + page count) and the extracted text
        for pages 1..N. Output is truncated to `max_chars`.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        ValueError: If `file_path` is not a `.pdf` file.
    """
    logger.info("TOOL CALL: pdf_extract_text")
    path = Path(file_path).expanduser().resolve()
    logger.info(f"Parsing the pdf file from {file_path}")

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    pages_to_read = min(max(1, int(max_pages)), page_count)

    out: list[str] = [
        f"PDF: {path}\n",
        f"Total pages: {page_count}\n",
        f"Extracting pages: 1..{pages_to_read}\n",
    ]

    total = 0
    for i in range(pages_to_read):
        page = reader.pages[i]
        text = (page.extract_text() or "").strip()

        chunk = f"\n--- Page {i + 1} ---\n{text}\n"
        out.append(chunk)
        total += len(chunk)

        if total >= max_chars:
            out.append("\n[TRUNCATED]\n")
            break

    joined = "".join(out)
    return joined[:max_chars]