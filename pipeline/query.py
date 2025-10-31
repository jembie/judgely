import re
from pathlib import Path
from typing import List
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd

from court import Judge, Jury
from utils import BASE_PATH, BalancedGenerator, DataHolder, SimpleGenerator, Message


class Pipeline:

    def __init__(self, judge: Judge = None, jury: Jury = None, generator: BalancedGenerator | SimpleGenerator = None):
        self.judge = judge
        self.jury = jury
        self.generator = generator
        self._run_dir_set = False
        self._run_dir = "1"
        self.results_path = None

    def _prepare_data_for_judge(self, jury_replies: List[str], dataholder: DataHolder):
        for index, answer in enumerate(dataholder.answers):
            answer["content"] = f"1:\n{answer["content"]}\n 2:{jury_replies[index]}"

    def _get_next_run_directory(self, results_path: Path) -> Path:
        if not results_path.exists():
            results_path.mkdir(parents=True)

        run_dirs = [directory for directory in results_path.iterdir() if directory.is_dir() and directory.name.startswith("run_")]

        # If we have no directories / never started a run
        if not run_dirs:
            self._run_dir_set = True
            return results_path / f"run_{self._run_dir}"

        if not self._run_dir_set:
            run_numbers = []
            for run in run_dirs:
                try:
                    number = int(run.name.split("_")[1])
                    run_numbers.append(number)
                except (IndexError, ValueError):
                    # Ignore all directories that don't follow the run_XYZ schema, where XYZ is any arbitrary number
                    print(f"Couldn't correctly find the 'number' for: {run.name}. Skipping...")
                    continue

            self._run_dir = str(max(run_numbers) + 1)
            self._run_dir_set = True

        return results_path / f"run_{self._run_dir}"

    def _convert_replies_into_dataframe(self, judge_replies: List[str], jury_replies: List[str]) -> pd.DataFrame:

        dfs: List[pd.DataFrame] = []
        for index, raw_reply in enumerate(judge_replies):

            processed = re.split(r"\n(?=- Answer:)", raw_reply.strip())[0]

            answer_match = re.search(r"- Answer:\s*(.*)", processed)
            score_match = re.search(r"- Score:\s*(.*)", processed)
            reason_match = re.search(r"- Reason:\s*(.*)", processed)
            # Parse each entry into a dictionary

            data = [
                {
                    "Answer": answer_match.group(1) if answer_match else None,
                    "Score": float(score_match.group(1)) if score_match else None,
                    "Reason": reason_match.group(1) if reason_match else None,
                    "Jury": jury_replies[index],
                }
            ]

            dfs.append(pd.DataFrame(data=data))

        return pd.concat(dfs, ignore_index=True)

    def _save_results(self, df: pd.DataFrame, dataholder: DataHolder) -> None:
        dt = datetime.now()
        formatted_date = dt.strftime("%Y-%b-%d-%Hh")

        if self.results_path is None:
            self.results_path = BASE_PATH / "data" / "results" / self.judge.model / dataholder.dataset_name / formatted_date

        run_dir = self._get_next_run_directory(results_path=self.results_path)
        run_dir.mkdir(exist_ok=True, parents=True)

        output_file = f"{run_dir}/{dataholder.qtype}.csv"
        df.index = dataholder.indices
        df.index.name = "Position"
        df.to_csv(output_file)

    def query(self, **kwargs):
        self._run_dir_set = False

        for dataholder in self.generator.data:

            jury_replies = []
            for question in dataholder.questions:
                jury_replies.append(self.jury.chat(question, **kwargs))

            self._prepare_data_for_judge(jury_replies=jury_replies, dataholder=dataholder)

            judge_replies = []
            for answer in dataholder.answers:
                judge_replies.append(self.judge.chat(answer, **kwargs))

            df: pd.DataFrame = self._convert_replies_into_dataframe(judge_replies=judge_replies, jury_replies=jury_replies)

            self._save_results(df=df, dataholder=dataholder)

    def _save_compare_results(self, replies: List, dataholder: DataHolder, iteration_nr: int):

        dt = datetime.now()
        formatted_date = dt.strftime("%Y-%b-%d-%H-%M")

        judge_sc = f"JUDGE_SCORE_{iteration_nr}_{self.judge.model}"
        judge_conf = f"JUDGE_CONFIDENCE_{iteration_nr}_{self.judge.model}"
        iter_start = f"IterationNr_{iteration_nr}_start_{self.judge.model}"

        if Path("results.csv").exists():
            df = pd.read_csv("results.csv")
        else:
            df = dataholder.dataset

        df[judge_sc] = ""
        df[judge_conf] = ""
        df[iter_start] = formatted_date

        for index, reply in zip(df.index, replies):
            df.loc[index, judge_sc] = reply.judge_score.value
            df.loc[index, judge_conf] = reply.judge_confidence

        df.to_csv("results.csv", index=False)

    def _prep_for_comparison(
        self,
        dataholder: DataHolder,
    ) -> List[Message]:
        merged = []
        for questions_and_answers in zip(dataholder.questions, dataholder.answers):
            question, answer = questions_and_answers
            merged_content = f"1: {question["content"]}\n2: {answer["content"]}"
            merged.append({"role": "user", "content": merged_content})

        return merged

    def compare(self, iteration_nr: int, **kwargs):

        self._run_dir_set = False

        judge_replies = []
        for dataholder in self.generator.data:

            merged = self._prep_for_comparison(dataholder)
            start = timer()
            for request in merged:
                response = self.judge.chat(request, **kwargs)
                judge_replies.append(response)

            self._save_compare_results(judge_replies, dataholder, iteration_nr)
            end = timer()
            print(f"Execution of Iteration '{iteration_nr}' took: {end - start}s")
