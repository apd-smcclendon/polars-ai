""" Prompt to correct Python Code on Error
```
Today is {today_date}.
You are provided with a python polars dataframe (df) with {num_rows} rows and {num_columns} columns.
This is the metadata of the dataframe:
{df_head}.

The user asked the following question:
{question}

You generated this python code:
{code}

It fails with the following error:
{error_returned}

Correct the python code and return a new python code (do not import anything) that fixes the above
mentioned error. Do not generate the same code again.
```
"""  # noqa: E501
from datetime import date

from .base import Prompt


class CorrectErrorPrompt(Prompt):
    """Prompt to Correct Python code on Error"""

    text: str = """
Today is {today_date}.
You are provided with a python polars dataframe (df) with {num_rows} rows and {num_columns} columns.
This is the metadata of the dataframe:
{df_head}.

The user asked the following question:
{question}

You generated this python code:
{code}

It fails with the following error:
{error_returned}

Correct the python code and return a new python code using the polars library (do not import anything) that fixes the above mentioned error. Do not generate the same code again.
"""  # noqa: E501

    def __init__(self, **kwargs):
        super().__init__(**kwargs, today_date=date.today())
