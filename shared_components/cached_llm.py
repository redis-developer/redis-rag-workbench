from typing import Any, Optional

from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig



class CachedLLM(Runnable):
    def __init__(self, llm, llmcache):
        self.llm = llm
        self.llmcache = llmcache
        self.last_is_cache_hit = False

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs
    ) -> str:
        if isinstance(input, dict):
            question = input.get("query") or input.get("input")
        elif isinstance(input, str):
            question = input
        elif isinstance(input, StringPromptValue):
            question = input.text
        elif isinstance(input, ChatPromptValue):
            # Extract the last human message from the chat history
            human_message = next(m for m in input.messages if m.type == "human")
            question = human_message.content.strip()
        else:
            raise ValueError(f"Unexpected input type: {type(input)}")

        if not isinstance(question, str):
            raise TypeError(f"Question must be a string, got {type(question)}")

        cached_responses = self.llmcache.check(
            prompt=question, return_fields=["prompt", "response", "metadata"]
        )
        if cached_responses:
            self.last_is_cache_hit = True
            return cached_responses[0]["response"]

        self.last_is_cache_hit = False
        response = self.llm.invoke(input, **kwargs)
        text = response.content if hasattr(response, "content") else str(response)
        self.llmcache.store(prompt=question, response=text)
        return text

    def get_last_cache_status(self) -> bool:
        return self.last_is_cache_hit
