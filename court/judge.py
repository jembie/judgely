from typing import Dict, List, Optional
from openai import OpenAI
from .base import BaseTemplate
import pandas as pd


class Judge(BaseTemplate):

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1-1.5b-max",
        system_message: Optional[
            str
        ] = """
        You are an expert clinical doctor.
        """,
    ):
        super().__init__(model, system_message)
