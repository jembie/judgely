import os

from dotenv import load_dotenv

from court import Judge, Jury
from pipeline import Pipeline
from utils import BalancedGenerator, ClientConfig, SimpleGenerator
from pydantic import BaseModel, Field
from enum import Enum

from analysis import make_plots, ScoreComparison


load_dotenv()

BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")
config = ClientConfig(BASE_URL, API_KEY)


def run_queries(
    iterations: int = 5,
    questions: int = 10,
    judge_model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8",
    jury_model: str = "Qwen3-235B-A22B-Instruct-2507-FP8",
    **llm_params,
):
    generator = BalancedGenerator()
    generator.generate_set(amount=questions)

    # jury = Jury(model=jury_model, client_config=config)
    # judge = Judge(model=judge_model, client_config=config)


#
# pipeline = Pipeline(judge=judge, jury=jury, generator=generator)
#
# Run multiple iterations for analysis
# for _ in range(iterations):
# pipeline.query(max_completion_tokens=2048, **llm_params)


def run_analysis():
    make_plots()
    scores = ScoreComparison()
    scores.count()


class Category(str, Enum):
    SAME = "same"
    DIFFERENT = "different"
    CLUELESS = "don't know"


class ResponseFormat(BaseModel):
    judge_score: Category
    judge_confidence: float = Field(ge=0, le=10, description="Confidence score from 0 to 10")


def run_judge(
    iterations: int = 5,
    questions: int = 10,
    judge_model: str = "deepseek-r1:8b",
    **llm_params,
):
    generator = SimpleGenerator()
    generator.generate_set(questions)

    judge = Judge(model=judge_model, client_config=config)

    pipeline = Pipeline(judge=judge, generator=generator)

    for _ in range(iterations):
        pipeline.compare(**llm_params)


if __name__ == "__main__":
    # run_queries()
    # run_analysis()
    run_judge(response_format=ResponseFormat)
#
