from typing import Optional

from utils import ClientConfig

from .base import BaseTemplate


class Judge(BaseTemplate):

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1-1.5b-max",
        system_message: Optional[
            str
        ] = """
            You are a strict JSON formatter.
            Always respond with a JSON object matching this schema:
            {
            "judge_score": "<Category>",
            "judge_confidence": <float between 0 and 10>
            }
            """,
        *,
        client_config: Optional[ClientConfig] = None,
    ):
        super().__init__(model, system_message, client_config=client_config)
