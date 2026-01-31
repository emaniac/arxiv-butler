"""Tools for downloading arXiv PDFs.

This module provides a LangChain tool to download an arXiv paper PDF given either an
arXiv abstract URL (e.g. `https://arxiv.org/abs/<id>`) or a direct PDF URL, and store
the file locally.
"""

import logging
import re
from pathlib import Path

import requests
from langchain.tools import tool

DEFAULT_DOWNLOAD_DIR = Path("data/papers")
logger = logging.getLogger(__name__)


def _safe_filename(name: str, default: str = "paper") -> str:
    """Sanitize a filename for safe local storage.

    Args:
        name: Proposed filename (with or without extension).
        default: Fallback name if `name` is empty/whitespace.

    Returns:
        A filesystem-safe filename (without directory components), truncated to a reasonable length.
    """
    name = name.strip() or default
    name = re.sub(r"[^\w\-\. ]+", "_", name)
    name = name.replace(" ", "_")
    return name[:180]


def _to_pdf_url(arxiv_or_pdf_url: str) -> str:
    """Convert an arXiv URL to a PDF URL (best-effort).

    Args:
        arxiv_or_pdf_url: Either an arXiv abstract URL (containing `/abs/`) or a PDF URL.

    Returns:
        A URL that points to a PDF resource. If the input is an `/abs/` URL, it is converted to a
        `/pdf/` URL. If it is already a PDF URL, it is returned as-is (or with `.pdf` appended
        where appropriate).
    """
    url = arxiv_or_pdf_url.strip()
    if "/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    if url.endswith(".pdf"):
        return url
    if "/pdf/" in url and not url.endswith(".pdf"):
        return f"{url}.pdf"
    return url


@tool
def arxiv_download_pdf(
    arxiv_or_pdf_url: str,
    output_dir: Path = DEFAULT_DOWNLOAD_DIR,
    filename: str = "",
) -> str:
    """Download an arXiv paper PDF and store it locally.

    Args:
        arxiv_or_pdf_url: arXiv abstract URL (e.g. `https://arxiv.org/abs/<id>`) or a direct PDF URL.
        output_dir: Directory to save the PDF into. Created if it does not exist.
        filename: Optional output filename. If empty, a filename is derived from the URL.
            If the name does not end with `.pdf`, the extension is added.

    Returns:
        Absolute path to the saved PDF file as a string.

    Raises:
        requests.HTTPError: If the HTTP response is not successful.
        requests.RequestException: For network-related errors (timeouts, connection errors, etc.).
        ValueError: If the server responds with HTML (often indicates a non-PDF error page).
    """
    logger.info("TOOL CALL: arxiv_download_pdf")
    pdf_url = _to_pdf_url(arxiv_or_pdf_url)

    output_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        base = _safe_filename(filename)
    else:
        tail = pdf_url.rstrip("/").split("/")[-1]
        base = _safe_filename(tail or "paper")

    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"

    out_path = output_dir / base

    with requests.get(
        pdf_url,
        stream=True,
        timeout=60,
        headers={"User-Agent": "arxiv_butler-arxiv-langgraph/0.1 (python-requests)"},
    ) as resp:
        logger.info(f"Downloading the paper from {pdf_url}")
        resp.raise_for_status()

        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype.lower():
            raise ValueError(f"Expected a PDF but got Content-Type={ctype} from {pdf_url}")

        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
            logger.info(f"Saved the paper to {out_path.resolve()}")

    return str(out_path.resolve())
