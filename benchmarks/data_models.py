import json
from pathlib import Path

from pydantic import BaseModel


class EvaluationExample(BaseModel):
    """Example for evaluating the agent's performance."""

    name: str
    intent: str
    reference: str


def load_evaluation_examples_from_path(path: Path) -> list[EvaluationExample]:
    """Loads evaluation examples from a JSON file."""
    with path.open() as f:
        examples = json.load(f)
    return [EvaluationExample(**example) for example in examples]

