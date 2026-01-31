"""Tools for searching arXiv via the public export API.

This module provides LangChain tools to search arXiv using the official export API
(Atom feed) and return a compact, human-readable list of results.
"""
import logging
import textwrap
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

import requests
from langchain.tools import tool


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ArxivPaper:
    title: str
    authors: list[str]
    summary: str
    published: str
    arxiv_url: str
    pdf_url: str


def parse_arxiv_atom(xml_text: str) -> list[ArxivPaper]:
    """Parse an arXiv export API Atom feed into structured paper entries.

    Args:
        xml_text: Raw Atom feed XML as a string.

    Returns:
        A list of parsed `ArxivPaper` entries.
    """
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)

    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()

        authors: list[str] = []
        for a in entry.findall("atom:author", ns):
            name = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)

        arxiv_url = ""
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            rel = link.attrib.get("rel", "")
            href = link.attrib.get("href", "")
            link_type = link.attrib.get("type", "")
            title_attr = link.attrib.get("title", "")

            if rel == "alternate" and href:
                arxiv_url = href

            if href and (title_attr == "pdf" or link_type == "application/pdf"):
                pdf_url = href

        if arxiv_url and not pdf_url:
            pdf_url = arxiv_url.replace("/abs/", "/pdf/")

        papers.append(
            ArxivPaper(
                title=title,
                authors=authors,
                summary=summary,
                published=published,
                arxiv_url=arxiv_url,
                pdf_url=pdf_url,
            )
        )

    return papers


@tool
def arxiv_search(
    query: str,
    max_results: int = 5,
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance",
) -> str:
    """Search arXiv and return a compact, readable list of results.

    Args:
        query: User query text (keywords/title/author). This is passed to arXiv as `all:<query>`.
        max_results: Maximum number of results to return. Clamped to [1, 20].
        sort_by: Sort strategy for arXiv results. One of:
            - "relevance"
            - "lastUpdatedDate"
            - "submittedDate"

    Returns:
        A formatted string containing numbered results with title, authors, published date, arXiv link,
        PDF link, and a short abstract snippet. Returns "No results found on arXiv." if nothing matches.

    Raises:
        requests.HTTPError: If the arXiv API request fails (non-2xx response).
        requests.RequestException: For network-related errors (timeouts, DNS, etc.).
        xml.etree.ElementTree.ParseError: If the API returns malformed XML.
    """
    logger.info("TOOL CALL: arxiv_search")
    max_results = max(1, min(int(max_results), 20))

    base = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    url = f"{base}?{urllib.parse.urlencode(params)}"
    logger.info(f"Calling {url}")
    logger.info(f"Params: {params}")

    resp = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "arxiv_butler-arxiv-langgraph/0.1 (python-requests)"},
    )
    resp.raise_for_status()

    papers = parse_arxiv_atom(resp.text)
    if not papers:
        return "No results found on arXiv."

    blocks: list[str] = []
    for i, p in enumerate(papers, start=1):
        abstract_snippet = textwrap.shorten(
            p.summary.replace("\n", " "),
            width=280,
            placeholder="…",
        )
        authors = ", ".join(p.authors[:6]) + (" et al." if len(p.authors) > 6 else "")
        blocks.append(
            "\n".join(
                [
                    f"[{i}] {p.title}",
                    f"Authors: {authors}",
                    f"Published: {p.published}",
                    f"arXiv: {p.arxiv_url}",
                    f"PDF:   {p.pdf_url}",
                    f"Abstract: {abstract_snippet}",
                ]
            )
        )
    logger.info(f"Found {len(blocks)} papers.")
    return "\n\n".join(blocks)
