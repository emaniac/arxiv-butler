"""arXiv Butler.

A little interactive CLI agent that can:
- search arXiv,
- download paper PDFs,
- extract text from local PDFs for Q&A.

Run:
    python -m src.main
"""

import logging
import os
import textwrap

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from arxiv_butler.src.arxiv_butler.tools.arxiv_download import arxiv_download_pdf
from arxiv_butler.src.arxiv_butler.tools.pdf_text import pdf_extract_text
from arxiv_butler.src.arxiv_butler.tools.arxiv_search import arxiv_search

logger = logging.getLogger(__name__)


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
    2) If the user asks to download/save/store the PDF:
         - If permission is already explicit, download with arxiv_download_pdf.
         - Otherwise ask ONE short permission question before downloading.
    3) If the user asks a question about the paper's contents:
         - If the PDF is not downloaded (check the previous conversation), ask the user whether to download it first.
         - If downloaded, call pdf_extract_text and answer using the extracted text.
    4) If you are unsure which paper/PDF the user means, ask ONE clarifying question.

    Output style:
    - Be concise by default. If the user asks about a paper generally, give a short answer.
    - When you cite details from the PDF text, mention the page number if available in the extracted text.
    """
)


class AgentState(BaseModel):
    """Conversation state passed between agent invocations.

    Attributes:
        messages: Accumulated chat messages (user, assistant, tool messages).
    """

    messages: list[AnyMessage]


def build_agent():
    """Create and configure the agent with tools and a system prompt.

    Returns:
        A runnable agent instance created by `create_agent`.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        api_key=api_key,
    )

    tools = [arxiv_search, arxiv_download_pdf, pdf_extract_text]
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def read_input(default_message: str | None = None) -> str:
    """Read a single user input line from the terminal.

    Args:
        default_message: If provided and the user enters an empty line, this value is returned.

    Returns:
        The user inputs text (or `default_message`).

    Raises:
        ValueError: If the user input is empty and `default_message` is None.
        SystemExit: If the user types 'exit' or 'quit'.
    """
    user_text = input("you> ").strip()
    if not user_text:
        if default_message is None:
            raise ValueError("Input cannot be empty")
        logger.info(f"Using the default message: {default_message}")
        return default_message
    if user_text.lower() in {"exit", "quit"}:
        logger.info("Bye.")
        exit(0)
    return user_text


def main() -> None:
    """Run the interactive console loop."""
    agent = build_agent()
    user_query = "Find the paper Pretraining on the Test Set Is All You Need by Rylan Schaeffer"

    state: AgentState = AgentState(messages=[])

    while True:
        user_message = HumanMessage(content=read_input(default_message=user_query))
        state.messages.append(user_message)
        print(f"USER: {user_message.content}")

        result = agent.invoke(state)
        state = AgentState(**result)

        print(f"AGENT: {state.messages[-1].content}")


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOGGING_LEVEL", logging.INFO))
    main()
