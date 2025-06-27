import pandas as pd
from typing import Dict, List, Tuple, TypedDict
import numpy as np
from dataclasses import dataclass

from .constants import BASE_PATH
from pathlib import Path


class Message(TypedDict):
    role: str
    content: str


MessageTemplate: Message = {"role": "user", "content": ""}


@dataclass
class DataHolder:
    questions: List[Message]
    answers: List[Message]
    rows_used: np.ndarray
    qtype: str
    qtype_total_entries: int
    data_set_name: str
    original_indices: list


class BalancedGenerator:

    def __init__(self, csv_dir_path: Path | str = ""):
        self.csv_dict: Dict[str, pd.DataFrame] = {}
        self.csv_dir = csv_dir_path if csv_dir_path else Path(BASE_PATH, "data", "csv")
        self._load_csvs()
        self.data: List[DataHolder] = []

    def _load_csvs(self) -> bool:
        csv_files = [path for path in self.csv_dir.glob("**/*") if path.is_file() and path.suffix == ".csv"]

        if not csv_files:
            print(f"No files were found within the {self.csv_dir.name} directory, aborting...")
            return False

        for csv in csv_files:
            dataset_name = csv.stem
            self.csv_dict[dataset_name] = pd.read_csv(csv)

        return True

    def _get_unique_qtypes(self, df: pd.DataFrame) -> List[np.ndarray]:
        return df["qtype"].unique()

    def _validate_amount(self, amount: int, df_len: int) -> bool:
        if amount > df_len:
            raise IndexError(
                f"The value provided for 'amount' ({amount =} > {df_len =}) is too high. Every 'qtype' must have at least as many entries as 'amount' for 'BalancedSetGenerator' instances."
            )
        return True

    def generate_set(self, seed: int = 42, amount: int = 5) -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(seed=seed)

        for dataset_name, df in self.csv_dict.items():
            unique_qtypes = self._get_unique_qtypes(df)

            for qtype in unique_qtypes:
                original_indices = df[df["qtype"] == qtype].drop_duplicates(subset=["Question"]).index.values.tolist()
                subset_df = df[df["qtype"] == qtype].drop_duplicates(subset=["Question"]).reset_index()
                df_len = len(subset_df)

                # Make sure that the input is amount <= df_len
                self._validate_amount(amount=amount, df_len=df_len)

                # Generate a np.ndarray[amount] with values ranging within the interval of [0, LAST_ROW] of the qtype
                try:
                    random_sequence = rng.integers(low=0, high=df_len - 1, size=amount)
                except Exception:
                    print(f"[ERROR] while trying to parse rng.integers with {df_len =}")

                chosen_rows = subset_df.iloc[random_sequence]
                questions, answers = self._generate_questions_answers(chosen_rows)

                self.data.append(
                    DataHolder(
                        qtype=qtype,
                        data_set_name=dataset_name,
                        questions=questions,
                        answers=answers,
                        qtype_total_entries=df_len,
                        rows_used=random_sequence,
                        original_indices=original_indices,
                    )
                )

    def _generate_questions_answers(self, df: pd.DataFrame) -> Tuple[List[Message], List[Message]]:
        questions = []
        answers = []
        for row in df.itertuples():
            questions_template = MessageTemplate.copy()
            answers_template = MessageTemplate.copy()

            questions_template["content"] = row.Question
            answers_template["content"] = row.Answer

            questions.append(questions_template)
            answers.append(answers_template)

        return questions, answers
