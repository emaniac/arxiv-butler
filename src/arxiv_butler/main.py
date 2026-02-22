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

from langchain_core.messages import HumanMessage

from arxiv_butler.agents.arxiv import build_arxiv_agent, ARXIV_DEFAULT_COMPLETION_PARAMS

logger = logging.getLogger(__name__)



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
    agent = build_arxiv_agent(completion_params=ARXIV_DEFAULT_COMPLETION_PARAMS)
    user_query = "Find the paper Pretraining on the Test Set Is All You Need by Rylan Schaeffer"

    while True:
        user_message = read_input(default_message=user_query)
        print(f"USER: {user_message}")
        response = agent.process_message(user_message)
        print(f"AGENT: {response}")


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOGGING_LEVEL", logging.INFO))
    main()
