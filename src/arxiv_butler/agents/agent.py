import os

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import Tool, BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class OpenAICompletionParams(BaseModel):
    """Class representing completion parameters for OpenAI.

    Attributes:
        model: Name of OpenAI model to use.
        temperature: Sampling temperature.
        max_tokens: Max number of tokens to generate.
        logprobs: Whether to return logprobs.
        stream_options: Streaming configuration (e.g. {"include_usage": True}).
        use_responses_api: Whether to use the responses API.
    """

    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    logprobs: bool | None = None
    use_responses_api: bool | None = None


class Agent:
    """Agent class for managing conversation state and executing tools."""

    def __init__(self, tools: list[BaseTool], system_prompt: str, completion_params: OpenAICompletionParams) -> None:
        self._messages: list[AnyMessage] = []
        api_key = os.getenv("OPENAI_API_KEY")
        self._llm = ChatOpenAI(api_key=api_key, **completion_params.model_dump())
        self._agent = create_agent(model=self._llm, tools=tools, system_prompt=system_prompt)

    def process_message(self, message: str) -> str:
        user_message = HumanMessage(content=message)
        self._messages.append(user_message)
        result = self._agent.invoke(dict(messages=self._messages))
        self._messages = result["messages"]
        return self._messages[-1].content

    def clear_messages(self) -> None:
        self._messages = []
