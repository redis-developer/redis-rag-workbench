from typing import Any, Optional
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompt_values import StringPromptValue, ChatPromptValue

class CachedLLM(Runnable):
    def __init__(self, llm, llmcache):
        self.llm = llm
        self.llmcache = llmcache
        self.last_is_cache_hit = False
        print("DEBUG: CachedLLM initialized")

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs
    ) -> str:
        if isinstance(input, dict):
            print(f"DEBUG: CachedLLM input is dict ==> {input}")
            question = input.get("query") or input.get("input")
        elif isinstance(input, str):
            print(f"DEBUG: CachedLLM input is str ==> {input}")
            question = input
        elif isinstance(input, StringPromptValue):
            print(f"DEBUG: CachedLLM input is StringPromptValue ==> {input}")
            question = input.text
        elif isinstance(input, ChatPromptValue):
            print(f"DEBUG: CachedLLM input is ChatPromptValue ==> {input}")
            # Extract the last human message from the chat history
            print(f"DEBUG: CachedLLM input is ChatPromptValue ==> {input}")
            human_message = next(m for m in input.messages if m.type == "human")
            question = human_message.content.strip()
        else:
            raise ValueError(f"Unexpected input type: {type(input)}")

        if not isinstance(question, str):
            raise TypeError(f"Question must be a string, got {type(question)}")

        print(f"DEBUG: Checking cache for question: {question}...")
        cached_responses = self.llmcache.check(
            prompt=question, return_fields=["prompt", "response", "metadata"]
        )
        if cached_responses:
            self.last_is_cache_hit = True
            print("DEBUG: Cache hit")
            return cached_responses[0]["response"]

        self.last_is_cache_hit = False
        print("DEBUG: Cache miss, querying LLM")
        response = self.llm.invoke(input, **kwargs)
        text = response.content if hasattr(response, "content") else str(response)
        print(f"DEBUG: Storing in cache: {question[:50]}...")
        self.llmcache.store(prompt=question, response=text)
        return text

    def get_last_cache_status(self) -> bool:
        return self.last_is_cache_hit