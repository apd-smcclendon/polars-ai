import sys
sys.path.append('C:/Users/aaron/OneDrive/Documents/polarsai/polars-ai/pandasai')
from prompts.base import Prompt


class LangchainLLM:
    """
    Class to wrap Langchain LLMs 
    """

    langchain_llm = None

    def __init__(self, langchain_llm):
        self._langchain_llm = langchain_llm

    def call(self, instruction: Prompt, value: str, suffix: str = "") -> str:
        prompt = str(instruction) + value + suffix
        return langchain_llm.predict(prompt)

    @property
    def type(self) -> str:
        return "langchain_" + self._langchain_llm._llm_type