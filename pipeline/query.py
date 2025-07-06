import re
from pathlib import Path
from typing import List

import pandas as pd

from court import Judge, Jury
from utils import BASE_PATH, BalancedGenerator, DataHolder


class Pipeline:

    def __init__(self, judge: Judge, jury: Jury, generator: BalancedGenerator):
        self.judge = judge
        self.jury = jury
        self.generator = generator
        self._run_dir_set = False
        self._run_dir = "1"

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
        for raw_reply in judge_replies:

            entries = re.split(r"\n(?=- Answer:)", raw_reply.strip())

            # Parse each entry into a dictionary
            data = []
            for index, entry in enumerate(entries):
                answer_match = re.search(r"- Answer:\s*(.*)", entry)
                score_match = re.search(r"- Score:\s*(.*)", entry)
                reason_match = re.search(r"- Reason:\s*(.*)", entry)

                data.append(
                    {
                        "Answer": answer_match.group(1) if answer_match else None,
                        "Score": float(score_match.group(1)) if score_match else None,
                        "Reason": reason_match.group(1) if reason_match else None,
                        "Jury": jury_replies[index],
                    }
                )
            dfs.append(pd.DataFrame(data=data))

        return pd.concat(dfs, ignore_index=True)

    def _save_results(self, df: pd.DataFrame, dataholder: DataHolder) -> None:
        results_path = BASE_PATH / "data" / "results" / self.judge.model / dataholder.dataset_name

        run_dir = self._get_next_run_directory(results_path=results_path)
        run_dir.mkdir(exist_ok=True)

        output_file = f"{run_dir}/{dataholder.qtype}.csv"
        df.index = dataholder.indices
        df.index.name = "Position"
        df.to_csv(output_file)

    def query(self):
        self._run_dir_set = False

        for dataholder in self.generator.data:

            jury_replies = []
            for question in dataholder.questions:
                jury_replies.append(self.jury.chat(question))

            self._prepare_data_for_judge(jury_replies=jury_replies, dataholder=dataholder)

            judge_replies = []
            for answer in dataholder.answers:
                judge_replies.append(self.judge.chat(answer))

            df: pd.DataFrame = self._convert_replies_into_dataframe(judge_replies=judge_replies, jury_replies=jury_replies)

            self._save_results(df=df, dataholder=dataholder)
