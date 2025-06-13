from typing import Dict, List, Optional
from .base import BaseTemplate
from utils import ClientConfig


# TODO: Implement/Update jury's role
class Jury(BaseTemplate):

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1-1.5b-max",
        system_message: Optional[str] = "You are an expert clinical doctor.",
        *,
        client_config: Optional[ClientConfig] = None,
    ):
        super().__init__(model, system_message, client_config=client_config)
