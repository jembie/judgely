from typing import Dict, List, Optional
from ollama import ChatResponse, Client
from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, model: Optional[str] = "", system_message: Optional[str] = ""):
        self.client = Client()
        self.model = model
        self.system_message = system_message
        self.system_message_dict = {"role": "system", "content": system_message}

    @abstractmethod
    def chat(self, messages: List[Dict]) -> ChatResponse: ...
