from typing import Dict, List, Optional
from openai import OpenAI
from .base import BaseTemplate
import pandas as pd
from utils import ClientConfig


class Judge(BaseTemplate):

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1-1.5b-max",
        system_message: Optional[
            str
        ] = """You are an assistant that always responds in the following format:
            - Answer: <YOUR ANSWER>.
            - Score: a score from 1 to 5.
            - Reason: a reasoning for your choice of score.
            """,
        *,
        client_config: Optional[ClientConfig] = None,
    ):
        super().__init__(model, system_message)
