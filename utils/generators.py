import pandas as pd
from typing import Dict, List
import numpy as np

from .constants import BASE_PATH
from pathlib import Path


class TestSetGenerator:

    def __init__(self, csv_dir_path: Path | str = ""):
        self.csv_dict: Dict[str, pd.DataFrame] = {}
        self.csv_dir = csv_dir_path if csv_dir_path else Path(BASE_PATH, "data", "csv")
        self._load_csvs()

    def _load_csvs(self) -> bool:
        csv_files = [path for path in self.csv_dir.glob("**/*") if path.is_file()]

        if not csv_files:
            print(f"No files were found within the {self.csv_dir.name} directory, aborting...")
            return False

        for csv in csv_files:
            dataset_name = csv.stem
            self.csv_dict[dataset_name] = pd.read_csv(csv)
            return True

    def generate_set(self, seed: int = 42, amount: int = 10) -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(seed=seed)

        results = {}
        for dataset_name, df in self.csv_dict.items():
            df_len = len(df) - 1

            # Generate a np.ndarray[amount] with values ranging within the interval of [0, LAST_ROW]
            random_sequence = rng.integers(low=0, high=df_len, size=amount)
            results[dataset_name] = df.iloc[random_sequence]

        return results

    def generate_questions(self) -> List[Dict]:
        question_set = self.generate_set(amount=3)
        question_template = {"role": "user", "content": ""}

        results = []
        for _, questions_df in question_set.items():
            for row in questions_df.itertuples():
                template = question_template.copy()
                question = row.Question
                # answer = row.Answer

                template["content"] = question
                results.append(template)

        return results

    def create_format(self, question, answer) -> str:
        format_question = "[QUESTION]" + question + "[QUESTION]"
        format_answer = "\n[ANSWER TO QUESTION]" + answer + "[ANSWER TO QUESTION]"

        return format_question + format_answer

    def generate_answers(self) -> List[Dict]:
        question_set = self.generate_set(amount=3)
        question_template = {"role": "user", "content": ""}

        results = []
        for _, questions_df in question_set.items():
            for row in questions_df.itertuples():
                template = question_template.copy()
                question = row.Question
                answer = row.Answer

                template["content"] = self.create_format(question, answer)
                results.append(template)

        return results
