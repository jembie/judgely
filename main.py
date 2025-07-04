import os

from dotenv import load_dotenv

from court import Judge, Jury
from pipeline import Pipeline
from utils import BalancedGenerator, ClientConfig

load_dotenv()

BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")


def run(iterations: int = 5):
    config = ClientConfig(BASE_URL, API_KEY)
    generator = BalancedGenerator()
    generator.generate_set(amount=1)

    # # launch()
    jury = Jury(model="Llama-4-Maverick-17B-128E-Instruct-FP8", client_config=config)
    judge = Judge(model="Llama-4-Maverick-17B-128E-Instruct-FP8", client_config=config)

    pipeline = Pipeline(judge=judge, jury=jury, generator=generator)

    # Run multiple iterations for analysis
    for _ in range(iterations):
        pipeline.query()


if __name__ == "__main__":
    run()
