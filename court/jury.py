from typing import Dict, List, Optional
from ollama import ChatResponse
from .base import BaseModel


# TODO: Implement/Update jury's role
class Jury(BaseModel):

    def __init__(self, model: Optional[str] = "deepseek-r1-1.5b-max", system_message: Optional[str] = "You are an expert clinical doctor."):
        super().__init__(model, system_message)

    def format_response(self, response: str) -> str:
        cut_think_tags = response.split("</think>")[1]
        formatted_input = "[START INPUT]" + cut_think_tags + "\n\n[END INPUT]"

        return formatted_input

    def chat(self, messages: List[Dict]) -> str | None:
        response: ChatResponse = self.client.chat(self.model, messages=messages)
        formatted_response = self.format_response(response=response.message.content)

        return formatted_response or ""
