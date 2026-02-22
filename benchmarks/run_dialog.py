"""Runs a dialog between a user and tested agent."""

import argparse
import json
import logging
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from arxiv_butler.agents.agent import Agent
from arxiv_butler.agents.arxiv import build_arxiv_agent, ARXIV_DEFAULT_COMPLETION_PARAMS
from benchmarks.data_models import load_evaluation_examples_from_path
from benchmarks.metrics import compute_correctness
from benchmarks.user_agent import build_user_agent

logger = logging.getLogger(__name__)


class Args(BaseModel):
    """Arguments for the script."""

    dataset_path: Path
    log_level: str


def parse_args() -> Args:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOGGING_LEVEL", logging.getLevelName(logging.INFO)),
        help="Logging level.",
    )
    return Args(**vars(parser.parse_args()))


def simulate_conversation(user_agent: Agent, tested_agent: Agent, max_turns: int = 100) -> str | None:
    """Simulates a conversation between two agents.

    Args:
        user_agent: The user agent.
        tested_agent: The tested agent.
        max_turns: Maximum number of turns in the conversation.

    Returns:
        The final message from the user agent after the conversation.
    """
    print("Simulating conversation...")
    user_agent_message = user_agent.process_message("START")
    print(f"USER: {user_agent_message}")
    for _ in range(max_turns):
        tested_agent_message = tested_agent.process_message(user_agent_message)
        print(f"AGENT: {tested_agent_message}")
        user_agent_message = user_agent.process_message(tested_agent_message)
        print(f"USER: {user_agent_message}")
        if "TERMINATE" in user_agent_message:
            logger.info("Ending the simulation.")
            return user_agent_message.replace("TERMINATE", "")
    logger.warning(f"Could not resolve the task in the given number of turns ({max_turns}).")
    return None


def main(dataset_path: Path) -> None:
    """Runs a dialog between a user and the tested agent.

    Args:
        dataset_path: Path to the dataset JSON file.
    """
    examples = load_evaluation_examples_from_path(dataset_path)
    arxiv_agent = build_arxiv_agent(completion_params=ARXIV_DEFAULT_COMPLETION_PARAMS)
    llm_client = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.2
    )
    metrics = []

    for example in examples:
        user_agent = build_user_agent(example["intent"])
        answer = simulate_conversation(user_agent, arxiv_agent)
        example_metrics = {
            "correctness": compute_correctness(
                query=example["intent"],
                actual_answer=answer,
                reference_answer=example["reference"],
                llm_client=llm_client,
            )
        }
        metrics.append(example_metrics)
        arxiv_agent.clear_messages()

    print(metrics)


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOGGING_LEVEL", logging.INFO))
    args = parse_args()
    main(dataset_path=args.dataset_path)
