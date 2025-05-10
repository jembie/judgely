from typing import Dict, List, Optional

from ollama import ChatResponse, _client


class Judge:

    def __init__(self, model: Optional[str] = "deepseek-r1:1.5b"):

        self.client = _client
        self.model = model

    def judge(self, messages: List[Dict]) -> ChatResponse:
        response: ChatResponse = self.client.chat(self.model, messages=messages)
        return response.message.content
