from typing import Dict, List, Optional
from ollama import ChatResponse
from .base import BaseModel


class Judge(BaseModel):

    def __init__(self, model: Optional[str] = "deepseek-r1:1.5b", system_message: Optional[str] = "You are an expert clinical doctor."):
        super().__init__(model, system_message)

    def chat(self, messages: List[Dict]) -> str | None:
        full_message = [self.system_message_dict] + messages

        response: ChatResponse = self.client.chat(self.model, messages=full_message)
        return response.message.content or "Chatting with Judge failed."
