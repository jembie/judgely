from typing import Dict, List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion
from abc import ABC
from utils import ClientConfig, Message


class BaseTemplate(ABC):

    def __init__(self, model: Optional[str] = "", system_message: Optional[str] = "", *, client_config: Optional[ClientConfig] = None):
        self.client = (
            OpenAI(base_url=client_config.base_url, api_key=client_config.api_key)
            if client_config
            else OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
        )

        self.model = model
        self.system_message = system_message
        self.system_message_dict = {"role": "system", "content": system_message}

    def chat(self, message: Message) -> str | None:
        messages = [self.system_message_dict] + [message]
        # set the `temperature` to `0.0` to have consistency in the evaluation. Might lead to worse results at times, but we rather want to be consistency (slightly) worse than have random lucky shots of success.
        response: ChatCompletion = self.client.chat.completions.create(messages=messages, model=self.model, temperature=0.0)
        # This returns the raw string output of the model. If it's a 'thinking' capable mode, then the '<think> ... </think>' content is included within the string.
        return response.choices[0].message.content or f"Chatting with {self.__class__.__name__} has failed."
