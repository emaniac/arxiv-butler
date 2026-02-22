"""UserAgent simulating the user."""

import textwrap

from arxiv_butler.agents.agent import Agent, OpenAICompletionParams
from arxiv_butler.constants import TERMINATE

SYSTEM_PROMPT_TEMPLATE = textwrap.dedent(f"""
    You are an assistant that interacts with another agent.
    Your goal is to fulfil a specific task only by interacting with the other agent.
    When your task is completed, respond how was the task resolved and add {TERMINATE}.
    if the task was a question, answer with a short answer.
    
    The other agent is an arXiv search assistant.
    1. It searches arXiv for papers matching a query.
    2. It downloads PDFs of the top search results.
    3. It extracts text content from downloaded PDFs and answers the specific question.
    
    Ask only one of these tasks at a time.

    When you get the message "START", you start your conversation.

    # Your task:
    {{intent}}
""")


def build_user_agent(intent: str) -> Agent:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(intent=intent)
    return Agent(
        tools=[],
        system_prompt=system_prompt,
        completion_params=OpenAICompletionParams(
            model="gpt-4o-mini", temperature=0.2
        ),
    )


if __name__ == "__main__":
    agent = build_user_agent("Find the paper Pretraining on the Test Set Is All You Need by Rylan Schaeffer")
    print(agent.process_message("START"))