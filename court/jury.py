from typing import Dict, List, Optional
from ollama import ChatResponse
from .base import BaseModel


# TODO: Implement/Update jury's role
class Jury(BaseModel):

    def __init__(self, model: Optional[str] = "deepseek-r1:1.5b"):
        self.model = model
        super().__init__()

    def chat(self, messages: List[Dict]) -> str | None:
        response: ChatResponse = self.client.chat(self.model, messages=messages)
        return response.message.content
