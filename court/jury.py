from typing import Dict, List, Optional
from ollama import ChatResponse
from .base import BaseModel


# TODO: Implement/Update jury's role
class Jury(BaseModel):

    def __init__(self, model: Optional[str] = "deepseek-r1:1.5b", system_message: Optional[str] = "You are an expert clinical doctor."):
        super().__init__(model, system_message)

    def chat(self, messages: List[Dict]) -> str | None:
        response: ChatResponse = self.client.chat(self.model, messages=messages)
        return response.message.content
