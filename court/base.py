from typing import Dict, List
from ollama import ChatResponse, Client
from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self):
        self.client = Client()

    @abstractmethod
    def chat(self, messages: List[Dict]) -> ChatResponse: ...
