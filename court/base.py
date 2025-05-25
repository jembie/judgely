from typing import Dict, List
from ollama import ChatResponse, Client
from abc import ABC, abstractmethod


class BaseModel(ABC):
    _client_instance = None

    def __init__(self):
        if BaseModel._client_instance is None:
            BaseModel._client_instance = Client()
        self.client = BaseModel._client_instance

    @abstractmethod
    def chat(self, messages: List[Dict]) -> ChatResponse: ...
