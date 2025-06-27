from court import Judge, Jury
from utils import TestSetGenerator, ClientConfig
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")


def run():
    config = ClientConfig(BASE_URL, API_KEY)
    # generator = TestSetGenerator()
    # questions = generator.generate_questions(amount=1)
    question_template = {"role": "user", "content": "What is the point of points?"}

    # launch()
    jury = Jury(model="llama-3.1-70b-instruct-q4km", client_config=config)

    reply = jury.chat(messages=question_template)
    print(reply)


def main():
    # answers = generator.generate_answers()
    run()


if __name__ == "__main__":
    main()
