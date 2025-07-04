from typing import Optional

from utils import ClientConfig

from .base import BaseTemplate


class Judge(BaseTemplate):

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1-1.5b-max",
        system_message: Optional[
            str
        ] = """You are an assistant that receives two text snippets and must compare their semantic meaning. You will always responds in the following format:
            - Answer, either one of those rankings: ["No semantic relation at all", "Same domain, but no matching semantical meaning", "Some matching semantical meaning", "Great match in semantical meaning", "Identical semantic meaning"]
            - Score: a score from 1 to 5 of how semantically similar they are.
            - Reason: a reasoning for your choice of score.
            """,
        *,
        client_config: Optional[ClientConfig] = None,
    ):
        super().__init__(model, system_message, client_config=client_config)
