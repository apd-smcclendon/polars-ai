# -*- coding: utf-8 -*-
import ast
import io
import logging
import re
import sys
sys.path.append('C:/Users/aaron/OneDrive/Documents/polarsai/polars-ai/pandasai')
import uuid
import time
from contextlib import redirect_stdout
from typing import List, Optional, Union, Dict, Type
import importlib.metadata

#__version__ = importlib.metadata.version(__package__ or __name__)
import astor
import polars as pl
from constants import (
    WHITELISTED_BUILTINS,
    WHITELISTED_LIBRARIES,
)
from exceptions import BadImportError, LLMNotFoundError
from helpers._optional import import_dependency
from helpers.cache import Cache
from helpers.notebook import Notebook
from helpers.save_chart import add_save_chart
from helpers.shortcuts import Shortcuts
from llm.base import Query
from llm.langchain_base import LangchainLLM
from prompts.base import Prompt
from prompts.correct_error_prompt import CorrectErrorPrompt
from prompts.correct_multiples_prompt import CorrectMultipleDataframesErrorPrompt
from prompts.generate_python_code import GeneratePythonCodePrompt
from prompts.generate_response import GenerateResponsePrompt
from prompts.multiple_dataframes import MultipleDataframesPrompt


def get_version():
    """
    Get the version from the package metadata
    """
    from importlib.metadata import version

    return version("polarsai")


#__version__ = get_version()


class PolarsAI(Shortcuts):
    """
    PandasAI is a wrapper around a LLM to make dataframes conversational.

    This ...

    Note:
        Do not include the `self` parameter in the ``Args`` section.
    Args:
        _llm (obj): LLMs option to be used for API access
       
    Returns (str): Response to a Question related to Data

    """
    _query = Query
    _verbose: bool = False
    _is_conversational_answer: bool = False
    _enforce_privacy: bool = False
    _max_retries: int = 3
    _in_notebook: bool = False
    _original_instructions: dict = {
        "question": None,
        "df_head": None,
        "num_rows": None,
        "num_columns": None,
    }
    _cache: Cache = None
    _enable_cache: bool = True
    _prompt_id: Optional[str] = None
    _additional_dependencies: List[dict] = []
    _custom_whitelisted_dependencies: List[str] = []
    _start_time: float = 0
    _enable_logging: bool = True
    _logger: logging.Logger = None
    _logs: List[str] = []
    last_code_generated: Optional[str] = None
    last_code_executed: Optional[str] = None
    code_output: Optional[str] = None
    last_error: Optional[str] = None

    def __init__(
        self,
        llm_type=None,
        conversational=False,
        verbose=False,
        enforce_privacy=False,
        save_charts=False,
        save_charts_path=None,
        enable_cache=True,
        custom_whitelisted_dependencies=None,
        enable_logging=True,
        non_default_prompts: Optional[Dict[str, Type[Prompt]]] = None,
    ):
        """

        __init__ method of the Class PandasAI

        Args:
            llm (string): LLMs option to be used 
            conversational (bool): Whether to return answer in conversational form.
            Default to False
            verbose (bool): To show the intermediate outputs e.g. python code
            generated and execution step on the prompt.  Default to False
            enforce_privacy (bool): Execute the codes with Privacy Mode ON.
            Default to False
            save_charts (bool): Save the charts generated in the notebook.
            Default to False
            enable_cache (bool): Enable the cache to store the results.
            Default to True
            middlewares (list): List of middlewares to be used. Default to None
            custom_whitelisted_dependencies (list): List of custom dependencies to
            be used. Default to None
            enable_logging (bool): Enable the logging. Default to True
            non_default_prompts (dict): Mapping from keys to replacement prompt classes.
            Used to override specific types of prompts. Defaults to None.
        """

        # configure the logging
        # noinspection PyArgumentList
        # https://stackoverflow.com/questions/61226587/pycharm-does-not-recognize-logging-basicconfig-handlers-argument
        if enable_logging:
            handlers = [logging.FileHandler("polarsai.log")]
        else:
            handlers = []

        if verbose:
            handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        self._logger = logging.getLogger(__name__)

        if llm is None:
            raise LLMNotFoundError(
                "An LLM should be provided to instantiate a PolarsAI class instance"
            )
        self._load_llm(llm)
        self._is_conversational_answer = conversational
        self._verbose = verbose
        self._enforce_privacy = enforce_privacy
        self._save_charts = save_charts
        self._save_charts_path = save_charts_path
        self._process_id = str(uuid.uuid4())
        self._logs = []

        self._non_default_prompts = (
            {} if non_default_prompts is None else non_default_prompts
        )

        self.notebook = Notebook()
        self._in_notebook = self.notebook.in_notebook()

        self._enable_cache = enable_cache
        if self._enable_cache:
            self._cache = Cache()

        #if middlewares is not None:
        #    self.add_middlewares(*middlewares)

        if custom_whitelisted_dependencies is not None:
            self._custom_whitelisted_dependencies = custom_whitelisted_dependencies

    def _load_llm(self, llm_type: str):
        """

        Args:
            llm_type (object): llm instantiation
        """
        match llm_type:
            case "LlamaCpp":
                llm = LlamaCpp(
                    model_path=self.model_path, 
                    callbacks=callbacks, 
                    verbose=False
                )
            case "OpenAI":
                llm = OpenAI(
                temperature=0.9,
                openai_api_key=os.environ.get("OPENAI_TOKEN")
                )
            case "SageMaker":
                content_handler = ContentHandler()
                llm = SagemakerEndpoint(
                    endpoint_name=self._endpoint,
                    region_name=self._aws_region,
                    model_kwargs=self.parameters,
                    content_handler=_content_handler,
                )
            case "Custom":
                llm = self.LangchainLLM
            case _:
                raise BadImportError("llm not recognized")
        setattr(_query, llm, llm)

    def conversational_answer(self, question: str, answer: str) -> str:
        """
        Returns the answer in conversational form about the resultant data.

        Args:
            question (str): A question in Conversational form
            answer (str): A summary / resultant Data

        Returns (str): Response

        """

        generate_response_instruction = self._non_default_prompts.get(
            "generate_response", GenerateResponsePrompt
        )(question=question, answer=answer)
        return self._query.call(generate_response_instruction, "")

    def run(
        self,
        data_frame: Union[pl.DataFrame, List[pl.DataFrame]],
        prompt: str,
        is_conversational_answer: bool = None,
        show_code: bool = False,
        anonymize_df: bool = True,
        use_error_correction_framework: bool = True,
    ) -> Union[str, pl.DataFrame]:
        """
        Run the PolarsAI to make Dataframes Conversational.

        Args:
            data_frame (Union[pd.DataFrame, List[pd.DataFrame]]): A polars Dataframe
            prompt (str): A prompt to query about the Dataframe
            is_conversational_answer (bool): Whether to return answer in conversational
            form. Default to False
            show_code (bool): To show the intermediate python code generated on the
            prompt. Default to False
            anonymize_df (bool): Running the code with Sensitive Data. Default to True
            use_error_correction_framework (bool): Turn on Error Correction mechanism.
            Default to True

        Returns (str): Answer to the Input Questions about the DataFrame

        """

        self._start_time = time.time()

        self.log(f"Question: {prompt}")
        self.log(f"Running PolarsAI with {self._llm_type} LLM...")

        self._prompt_id = str(uuid.uuid4())
        self.log(f"Prompt ID: {self._prompt_id}")

        try:
            if self._enable_cache and self._cache.get(prompt):
                self.log("Using cached response")
                code = self._cache.get(prompt)
            else:
                rows_to_display = 0 if self._enforce_privacy else 5

                multiple: bool = isinstance(data_frame, list)

                if multiple:
                    heads = [
                        dataframe.head(rows_to_display)
                        for dataframe in data_frame
                    ]

                    multiple_dataframes_instruction = self._non_default_prompts.get(
                        "multiple_dataframes", MultipleDataframesPrompt
                    )
                    code = self._query.generate_code(
                        multiple_dataframes_instruction(dataframes=heads),
                        prompt,
                    )

                    self._original_instructions = {
                        "question": prompt,
                        "df_head": heads,
                    }

                else:
                    df_head = data_frame.head(rows_to_display)

                    generate_code_instruction = self._non_default_prompts.get(
                        "generate_python_code", GeneratePythonCodePrompt
                    )(
                        prompt=prompt,
                        df_head=df_head,
                        num_rows=data_frame.shape[0],
                        num_columns=data_frame.shape[1],
                    )
                    code = self._query.generate_code(
                        generate_code_instruction,
                        prompt,
                    )

                    self._original_instructions = {
                        "question": prompt,
                        "df_head": df_head,
                        "num_rows": data_frame.shape[0],
                        "num_columns": data_frame.shape[1],
                    }

                self.last_code_generated = code
                self.log(
                    f"""
                        Code generated:
                        ```
                        {code}
                        ```
                    """
                )

                if self._enable_cache and self._cache:
                    self._cache.set(prompt, code)

            if show_code and self._in_notebook:
                self.notebook.create_new_cell(code)

            answer = self.run_code(
                code,
                data_frame,
                use_error_correction_framework=use_error_correction_framework,
            )
            self.code_output = answer
            self.log(f"Answer: {answer}")

            if is_conversational_answer is None:
                is_conversational_answer = self._is_conversational_answer
            if is_conversational_answer:
                answer = self.conversational_answer(prompt, answer)
                self.log(f"Conversational answer: {answer}")

            self.log(f"Executed in: {time.time() - self._start_time}s")

            return answer
        except Exception as exception:
            self.last_error = str(exception)
            print(exception)
            return (
                "Unfortunately, I was not able to answer your question, "
                "because of the following error:\n"
                f"\n{exception}\n"
            )

    def clear_cache(self):
        """
        Clears the cache of the PandasAI instance.
        """
        if self._cache:
            self._cache.clear()

    def __call__(
        self,
        data_frame: Union[pl.DataFrame, List[pl.DataFrame]],
        prompt: str,
        is_conversational_answer: bool = None,
        show_code: bool = False,
        anonymize_df: bool = True,
        use_error_correction_framework: bool = True,
    ) -> Union[str, pl.DataFrame]:
        """
        __call__ method of PandasAI class. It calls the `run` method.

        Args:
            data_frame:
            prompt:
            is_conversational_answer:
            show_code:
            anonymize_df:
            use_error_correction_framework:

        Returns (str): Answer to the Input Questions about the DataFrame.

        """

        return self.run(
            data_frame,
            prompt,
            is_conversational_answer,
            show_code,
            anonymize_df,
            use_error_correction_framework,
        )

    def _check_imports(self, node: Union[ast.Import, ast.ImportFrom]):
        """
        Add whitelisted imports to _additional_dependencies.

        Args:
            node (object): ast.Import or ast.ImportFrom

        Raises:
            BadImportError: If the import is not whitelisted

        """
        if isinstance(node, ast.Import):
            module = node.names[0].name
        else:
            module = node.module

        library = module.split(".")[0]

        if library == "pandas":
            return

        if library in WHITELISTED_LIBRARIES + self._custom_whitelisted_dependencies:
            for alias in node.names:
                self._additional_dependencies.append(
                    {
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname or alias.name,
                    }
                )
            return

        if library not in WHITELISTED_BUILTINS:
            raise BadImportError(library)

    def _is_df_overwrite(self, node: ast.stmt) -> bool:
        """
        Remove df declarations from the code to prevent malicious code execution.

        Args:
            node (object): ast.stmt

        Returns (bool):

        """

        return (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and re.match(r"df\d{0,2}$", node.targets[0].id)
        )

    def _clean_code(self, code: str) -> str:
        """
        A method to clean the code to prevent malicious code execution

        Args:
            code(str): A python code

        Returns (str): Returns a Clean Code String

        """

        tree = ast.parse(code)

        new_body = []

        # clear recent optional dependencies
        self._additional_dependencies = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._check_imports(node)
                continue
            if self._is_df_overwrite(node):
                continue
            new_body.append(node)

        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree).strip()

    def _get_environment(self) -> dict:
        """
        Returns the environment for the code to be executed.

        Returns (dict): A dictionary of environment variables
        """

        return {
            "pd": pd,
            **{
                lib["alias"]: getattr(import_dependency(lib["module"]), lib["name"])
                if hasattr(import_dependency(lib["module"]), lib["name"])
                else import_dependency(lib["module"])
                for lib in self._additional_dependencies
            },
            "__builtins__": {
                **{builtin: __builtins__[builtin] for builtin in WHITELISTED_BUILTINS},
            },
        }

    def _retry_run_code(self, code: str, e: Exception, multiple: bool = False):
        """
        A method to retry the code execution with error correction framework.

        Args:
            code (str): A python code
            e (Exception): An exception
            multiple (bool): A boolean to indicate if the code is for multiple
            dataframes

        Returns (str): A python code
        """

        if multiple:
            error_correcting_instruction = self._non_default_prompts.get(
                "correct_multiple_dataframes_error",
                CorrectMultipleDataframesErrorPrompt,
            )(
                code=code,
                error_returned=e,
                question=self._original_instructions["question"],
                df_head=self._original_instructions["df_head"],
            )

        else:
            error_correcting_instruction = self._non_default_prompts.get(
                "correct_error", CorrectErrorPrompt
            )(
                code=code,
                error_returned=e,
                question=self._original_instructions["question"],
                df_head=self._original_instructions["df_head"],
                num_rows=self._original_instructions["num_rows"],
                num_columns=self._original_instructions["num_columns"],
            )

        return self._llm.generate_code(error_correcting_instruction, "")

    def run_code(
        self,
        code: str,
        data_frame: pl.DataFrame,
        use_error_correction_framework: bool = True,
    ) -> str:
        """
        A method to execute the python code generated by LLMs to answer the question
        about the input dataframe. Run the code in the current context and return the
        result.

        Args:
            code (str): A python code to execute
            data_frame (pd.DataFrame): A full Polars DataFrame
            use_error_correction_framework (bool): Turn on Error Correction mechanism.
            Default to True

        Returns (str): String representation of the result of the code execution.

        """

        multiple: bool = isinstance(data_frame, list)

        # Add save chart code
        if self._save_charts:
            code = add_save_chart(
                code, self._prompt_id, self._save_charts_path, not self._verbose
            )

        # Get the code to run removing unsafe imports and df overwrites
        code_to_run = self._clean_code(code)
        self.last_code_executed = code_to_run
        self.log(
            f"""
Code running:
```
{code_to_run}
```"""
        )

        environment: dict = self._get_environment()

        if multiple:
            environment.update(
                {f"df{i}": dataframe for i, dataframe in enumerate(data_frame, start=1)}
            )
        else:
            environment["df"] = data_frame

        # Redirect standard output to a StringIO buffer
        with redirect_stdout(io.StringIO()) as output:
            count = 0
            while count < self._max_retries:
                try:
                    # Execute the code
                    exec(code_to_run, environment)
                    code = code_to_run
                    break
                except Exception as e:
                    if not use_error_correction_framework:
                        raise e

                    count += 1

                    code_to_run = self._retry_run_code(code, e, multiple)

        captured_output = output.getvalue().strip()
        if code.count("print(") > 1:
            return captured_output

        # Evaluate the last line and return its value or the captured output
        # We do this because we want to return the right value and the right
        # type of the value. For example, if the last line is `df.head()`, we
        # want to return the head of the dataframe, not the captured output.
        lines = code.strip().split("\n")
        last_line = lines[-1].strip()

        match = re.match(r"^print\((.*)\)$", last_line)
        if match:
            last_line = match.group(1)

        try:
            result = eval(last_line, environment)

            # In some cases, the result is a tuple of values. For example, when
            # the last line is `print("Hello", "World")`, the result is a tuple
            # of two strings. In this case, we want to return a string
            if isinstance(result, tuple):
                result = " ".join([str(element) for element in result])

            return result
        except Exception:
            return captured_output

    def log(self, message: str):
        """Log a message"""
        self._logger.info(message)
        self._logs.append(message)

    @property
    def logs(self) -> List[str]:
        """Return the logs"""
        return self._logs

    def process_id(self) -> str:
        """Return the id of this PandasAI object."""
        return self._process_id

    @property
    def last_prompt_id(self) -> str:
        """Return the id of the last prompt that was run."""
        if self._prompt_id is None:
            raise ValueError("Pandas AI has not been run yet.")
        return self._prompt_id

    @property
    def last_prompt(self) -> str:
        """Return the last prompt that was executed."""
        if self._llm:
            return self._llm.last_prompt
