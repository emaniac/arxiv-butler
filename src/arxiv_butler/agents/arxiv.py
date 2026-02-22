"""
ArXiv agent module for searching, downloading, and analyzing academic papers.

This module provides an agent that integrates with the arXiv API to help users:
- Search for papers by title, author, or topic
- Download paper PDFs to local storage
- Extract and analyze text content from downloaded papers

The agent uses LangChain's tool-based architecture and OpenAI's language models
to provide a conversational interface for interacting with arXiv papers.

Components:
    - SYSTEM_PROMPT: Instructions that guide the agent's behavior and tool usage
    - DEFAULT_COMPLETION_PARAMS: Default model configuration (gpt-4o-mini, temp=0.2)
    - build_arxiv_agent: Factory function that creates a configured Agent instance

Tools:
    - arxiv_search: Search the arXiv API for papers matching a query
    - arxiv_download_pdf: Download a paper's PDF to data/papers/
    - pdf_extract_text: Extract text content from a downloaded PDF

Note:
    The agent is designed to operate as part of a continuous conversation and
    will ask for permission before downloading PDFs unless explicitly instructed.
"""

from __future__ import annotations

import textwrap

from langchain_core.tools import BaseTool

from arxiv_butler.agents.agent import Agent, OpenAICompletionParams
from arxiv_butler.tools.arxiv_download import arxiv_download_pdf
from arxiv_butler.tools.arxiv_search import arxiv_search
from arxiv_butler.tools.pdf_text import pdf_extract_text


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a careful research assistant that helps users find arXiv papers, optionally download them,
    and answer questions using the paper's extracted PDF text.

    Tools and when to use them:
    - arxiv_search:
        Use when the user asks to find a paper by title/author/topic, or when you need candidates to
        identify the correct paper.
    - arxiv_download_pdf:
        Use only after the user explicitly asks to download/save/store the PDF OR after you ask for
        permission and the user agrees. The tool returns a local file path.
    - pdf_extract_text:
        Use to read a local PDF (text-only) when the user asks questions about the paper's contents.
        Only use this tool if you have a local file path (i.e., the paper is already downloaded).

    Behavioral rules:
    1) If the user asks to find a paper, call arxiv_search and present all the papers concisely.
        - Always return all the results of the search.
    2) If the user asks to download/save/store the PDF:
        - If permission is already explicit, download with arxiv_download_pdf.
        - Otherwise ask ONE short permission question before downloading.
    3) If the user asks a question:
        - If the PDF is not downloaded (check the previous conversation), ask the user whether to download it first.
        - If downloaded, call pdf_extract_text and answer using the extracted text.
        - Always answer only after finding the information in the downloaded PDF.
    4) If you are unsure which paper/PDF the user means, ask ONE clarifying question.
    5) Do not use internal knowledge for answering questions.
        - Always download the PDF and extract the text before answering.
        - Answer only based on information from the papers you have processed.

    Output style:
    - Be concise by default. If the user asks about a paper generally, give a short answer.
    - When you cite details from the PDF text, mention the page number if available in the extracted text.
    - All your answers must be grounded in the specific downloaded paper. 
    """
)


ARXIV_DEFAULT_COMPLETION_PARAMS = OpenAICompletionParams(model="gpt-4o-mini", temperature=0.2)


def build_arxiv_agent(completion_params: OpenAICompletionParams) -> Agent:
    tools: list[BaseTool] = [arxiv_search, arxiv_download_pdf, pdf_extract_text]
    return Agent(
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        completion_params=completion_params,
    )