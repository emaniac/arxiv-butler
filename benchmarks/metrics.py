import json
import logging
import textwrap
from enum import Enum
from typing import Any

from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langsmith.schemas import Example
from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)


class Metric(Enum):
    """Metrics to evaluate the agent's performance."""

    CORRECTNESS = "correctness"


class EvaluationResult(BaseModel):
    """Result of evaluating the agent's performance."""

    metric: Metric
    score: float
    explanation: float


def parse_response_to_evaluation_result(response: str, metric: Metric) -> EvaluationResult | None:
    """Parse the response from the LLM into an EvaluationResult object.

    Args:
        response: The response string from the LLM.
        metric: The metric to evaluate against.

    Returns:
        EvaluationResult object if parsing is successful, None otherwise.
    """
    try:
        parsed = json.loads(response)
        return EvaluationResult(metric=metric, **parsed)
    except (json.JSONDecodeError, TypeError, ValidationError):
        logger.warning(f"Failed to parse response: '{response}'")
        return None

def compute_correctness(query: str, actual_answer: str, reference_answer: str, llm_client: ChatOpenAI) -> EvaluationResult | None:
    prompt_template = textwrap.dedent("""
        You are an expert evaluator assessing the correctness of an AI assistant's response.

        You will be given:
        1. A user query/request
        2. The actual answer provided by the assistant
        3. A reference/expected answer

        Your task is to evaluate how correct the actual answer is compared to the reference answer.
        Consider:
        - Factual accuracy
        - Completeness of information
        - Relevance to the query

        Provide a correctness score from 0.0 to 1.0, where:
        - 0.0 = completely incorrect or irrelevant
        - 0.5 = partially correct
        - 1.0 = fully correct and complete

        Respond with a JSON object containing:
        - "score": a number between 0.0 and 1.0
        - "explanation": a brief explanation of your scoring decision

        USER QUERY:
        {query}

        ACTUAL ANSWER:
        {actual_answer}

        REFERENCE ANSWER:
        {reference_answer}

        RESPONSE (JSON):
    """)

    prompt = prompt_template.format(
        query=query, actual_answer=actual_answer, reference_answer=reference_answer
    )
    response = llm_client.invoke(prompt)
    return parse_response_to_evaluation_result(response.content)


def evaluate(example: Example, trajectory: list[AnyMessage]) -> dict[str, Any]:
    ...


