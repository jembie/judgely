from typing import Optional

from utils import ClientConfig

from .base import BaseTemplate


class Jury(BaseTemplate):

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1:8b",
        system_message: Optional[str] = "You are an expert clinical Doctor",
        *,
        client_config: Optional[ClientConfig] = None,
    ):
        super().__init__(model, system_message, client_config=client_config)
