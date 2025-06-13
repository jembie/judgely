from typing import Dict, List, Optional
from .base import BaseTemplate
from utils import ClientConfig


# TODO: Implement/Update jury's role
class Jury(BaseTemplate):

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1:8b",
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
        super().__init__(model, system_message, client_config=client_config)
