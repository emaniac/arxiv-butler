from langchain_core.messages import AnyMessage

from arxiv_butler.constants import TERMINATE


def extract_answer(trajectory: list[AnyMessage]) -> str | None:

    for message in trajectory:
        if TERMINATE in message.content:
            return message.content.replace(TERMINATE, "")
    return None