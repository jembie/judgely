from typing import Dict, List, Optional
from ollama import ChatResponse
from .base import BaseModel
import pandas as pd


class Judge(BaseModel):

    def __init__(
        self,
        model: Optional[str] = "deepseek-r1-1.5b-max",
        system_message: Optional[
            str
        ] = """
        You are an expert clinical doctor.
        Your task is to determine whether the following answers (tagged with [START INPUT] for showcasing the beginning of an section to evaluate and [END INPUT] for demonstrating the end of the section to be evaluated.) match the [QUESTION] and [ANSWER TO QUESTION] sections.
        Give a rating of confidence to the [INPUT]s a the end.
        """,
    ):
        super().__init__(model, system_message)

    def chat(self, messages: List[Dict]) -> str:
        full_message = [self.system_message_dict] + messages

        response: ChatResponse = self.client.chat(self.model, messages=full_message)
        return response.message.content or "Chatting with Judge failed."
