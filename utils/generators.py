from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import numpy as np
import pandas as pd

from .constants import BASE_PATH


class Message(TypedDict):
    role: str
    content: str


MessageTemplate: Message = {"role": "user", "content": ""}


@dataclass
class DataHolder:
    questions: List[Message]
    answers: List[Message]
    qtype: str
    total_entries: int
    dataset_name: str
    indices: np.ndarray


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
        np.random.seed(seed=seed)

        for dataset_name, df in self.csv_dict.items():
            unique_qtypes = self._get_unique_qtypes(df)

            for qtype in unique_qtypes:
                qtype_df = df[df["qtype"] == qtype].drop_duplicates(subset=["Question"])
                qtype_indices = qtype_df.index.values.tolist()
                qtype_len = len(qtype_df)

                # Make sure that the input is amount <= df_len
                self._validate_amount(amount=amount, df_len=qtype_len)

                # Generate a np.ndarray[amount] with values ranging within the interval of [0, LAST_ROW] of the qtype
                try:
                    random_sequence = np.random.choice(qtype_indices, size=amount, replace=False)
                except Exception:
                    print(f"[ERROR] while trying to parse np.random.choice with {qtype_len =}")

                chosen_rows = qtype_df.loc[random_sequence]
                questions, answers = self._generate_questions_answers(chosen_rows)

                self.data.append(
                    DataHolder(
                        qtype=qtype,
                        dataset_name=dataset_name,
                        questions=questions,
                        answers=answers,
                        total_entries=qtype_len,
                        indices=random_sequence,
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
