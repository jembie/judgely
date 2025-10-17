import os

from dotenv import load_dotenv

from court import Judge, Jury
from pipeline import Pipeline
from utils import BalancedGenerator, ClientConfig

from analysis import make_plots, ScoreComparison


load_dotenv()

BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")


def run_queries(
    iterations: int = 5,
    questions: int = 10,
    judge_model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8",
    jury_model: str = "Qwen3-235B-A22B-Instruct-2507-FP8",
    **llm_params,
):
    """This function initiates and executes the questionaire pipeline.

    Args:
        iterations (int, optional): How often the querying (of the exact same randomly chosen questions) should be repeated. Defaults to 5.
        questions (int, optional): Amount of questions to randomly choose per category. Defaults to 10.
        judge_model (str, optional): Judging model that outputs a numerical and textual score, based on the input of the jury model and the gold standard answer. Defaults to "Llama-4-Maverick-17B-128E-Instruct-FP8".
        jury_model (str, optional): LLM model that answers questions and generates a response. Defaults to "Qwen3-235B-A22B-Instruct-2507-FP8".
    """

    config = ClientConfig(BASE_URL, API_KEY)
    generator = BalancedGenerator()
    generator.generate_set(amount=questions)

    jury = Jury(model=jury_model, client_config=config)
    judge = Judge(model=judge_model, client_config=config)

    pipeline = Pipeline(judge=judge, jury=jury, generator=generator)

    # Run multiple iterations for analysis
    for _ in range(iterations):
        pipeline.query(max_completion_tokens=2048, **llm_params)


def run_analysis():
    make_plots()
    scores = ScoreComparison()
    scores.count()


if __name__ == "__main__":
    run_queries()
    # run_analysis()
#
