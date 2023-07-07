""" Base class to implement a new LLM

This module is the base class to integrate the various LLMs API. This module also
includes the Base LLM classes for OpenAI, HuggingFace and Google PaLM.

Example:

    ```
    from .base import BaseOpenAI

    class CustomLLM(BaseOpenAI):

        Custom Class Starts here!!
    ```
"""

import ast
import re
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Any, Dict, Optional
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.llms.base import LLM

import openai
import requests

from ..exceptions import (
    APIKeyNotFoundError,
    MethodNotImplementedError,
    NoCodeFoundError,
)
from ..helpers._optional import import_dependency
from ..prompts.base import Prompt

load_dotenv()

class Query:
    """Base class to query an LLM and extraxt the python code"""

    def _polish_code(self, code: str) -> str:
        """
        Polish the code by removing the leading "python" or "py",  \
        removing the imports and removing trailing spaces and new lines.

        Args:
            code (str): Code

        Returns:
            str: Polished code
        """
        if re.match(r"^(python|py)", code):
            code = re.sub(r"^(python|py)", "", code)
        if re.match(r"^`.*`$", code):
            code = re.sub(r"^`(.*)`$", r"\1", code)
        code = code.strip()
        return code

    def _is_python_code(self, string):
        """
        Return True if it is valid python code.
        Args:
            string (str):

        Returns (bool): True if Python Code otherwise False

        """
        try:
            ast.parse(string)
            return True
        except SyntaxError:
            return False

    def _extract_code(self, response: str, separator: str = "```") -> str:
        """
        Extract the code from the response.

        Args:
            response (str): Response
            separator (str, optional): Separator. Defaults to "```".

        Raises:
            NoCodeFoundError: No code found in the response

        Returns:
            str: Extracted code from the response
        """
        code = response
        if len(code.split(separator)) > 1:
            code = code.split(separator)[1]
        code = self._polish_code(code)
        if not self._is_python_code(code):
            raise NoCodeFoundError("No code found in the response")

        return code

    def generate_code(self, instruction: Prompt, prompt: str) -> str:
        """
        Generate the code based on the instruction and the given prompt.

        Returns:
            str: Code
        """
        return self._extract_code(self.call(instruction, prompt, suffix="\n\nCode:\n"))


class ContentHandler(LLMContentHandler):
    # Below definitions are set as attributes
    # by LLMContentHandler
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        """
        method defined to allow custom SM endpoint to be used directly
        this should be used if AWS CLI is set up

        Parameters
        ----------
        prompt : str
            model query.
        model_kwargs : TYPE, optional
            Parameters. The default is {}.

        Returns
        -------
        bytes
            API result.

        """
        input_str = json.dumps({"text_inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        """
        method used to convert model output into human readable JSON
        format

        Parameters
        ----------
        output : bytes
            readable model response.

        Returns
        -------
        str
            decoded API call output JSON.

        """
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_texts"][0]


class CustomLLM(LLM):

    def _call(self, prompt: str, stop=None) -> str:
        """
        Method used by Langchain to call a loaded llm model. Uses model from 
        .env var API endpoint to query

        Parameters
        ----------
        prompt : str
            text to be input into model.
        stop : TYPE, optional
            The default is None.

        Returns
        -------
        str
            model response.

        """
        _response = requests.post(
            os.environ.get('ENDPOINT_NAME'), 
            json.dumps({"user_token": os.environ.get('TOKEN'), 
                        "prompt": prompt})
        )
        _res = json.loads(_response.content)
        return _res["body"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        property attribute from Langchain to define custom class
        model name reference. Fixed here as flan but could be set in .env

        Returns
        -------
        Mapping[str, Any]
            model name.

        """
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        """
        property attribute from Langchain to define custom class
        reference to show custom class being used

        Returns
        -------
        str
            custom.

        """
        return "custom"

class BaseOpenAI(Query, ABC):
    """Base class to implement a new OpenAI LLM
    LLM base class, this class is extended to be used with OpenAI API.

    """

    api_token: str
    temperature: float = 0
    max_tokens: int = 512
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0.6
    stop: Optional[str] = None
    # support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None

    def _set_params(self, **kwargs):
        """
        Set Parameters
        Args:
            **kwargs: ["model", "engine", "deployment_id", "temperature","max_tokens",
            "top_p", "frequency_penalty", "presence_penalty", "stop", ]

        Returns: None

        """

        valid_params = [
            "model",
            "engine",
            "deployment_id",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        Get the default parameters for calling OpenAI API

        Returns (Dict): A dict of OpenAi API parameters

        """

        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def completion(self, prompt: str) -> str:
        """
        Query the completion API

        Args:
            prompt (str): Prompt

        Returns:
            str: LLM response
        """
        params = {**self._default_params, "prompt": prompt}

        if self.stop is not None:
            params["stop"] = [self.stop]

        response = openai.Completion.create(**params)

        return response["choices"][0]["text"]

    def chat_completion(self, value: str) -> str:
        """
        Query the chat completion API

        Args:
            value (str): Prompt

        Returns:
            str: LLM response
        """
        params = {
            **self._default_params,
            "messages": [
                {
                    "role": "system",
                    "content": value,
                }
            ],
        }

        if self.stop is not None:
            params["stop"] = [self.stop]

        response = openai.ChatCompletion.create(**params)

        return response["choices"][0]["message"]["content"]