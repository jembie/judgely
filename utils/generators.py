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

    def generate_set(self, seed: int = 42, amount: int = 5) -> Dict[str, pd.DataFrame]:
        np.random.seed(seed=seed)

        for dataset_name, df in self.csv_dict.items():
            unique_qtypes = df["qtype"].unique()

            for qtype in unique_qtypes:
                qtype_df = df[df["qtype"] == qtype].drop_duplicates(subset=["Question"])
                qtype_indices = qtype_df.index.values.tolist()
                qtype_len = len(qtype_df)

                # Make sure that the input is amount <= df_len
                _validate_amount(amount, qtype_len)

                # Generate a np.ndarray[amount] with values ranging within the interval of [0, LAST_ROW] of the qtype
                try:
                    random_sequence = np.random.choice(qtype_indices, size=amount, replace=False)
                except Exception:
                    print(f"[ERROR] while trying to parse np.random.choice with {qtype_len=}")

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


class SimpleGenerator:
    def __init__(self, data_path: Path | str = ""):
        # Temporary storage for loaded files
        self._raw_dataframes: Dict[str, pd.DataFrame] = {}
        self._data_files_path = Path(data_path) if data_path else Path(BASE_PATH, "data")
        self.data: List[DataHolder] = []

    def generate_set(self, seed: int = 42, amount: int = None) -> Dict[str, pd.DataFrame]:
        np.random.seed(seed=seed)

        for dataset_name, df in self._raw_dataframes.items():
            if amount is None:
                amount = len(df.index)

            # Make sure that the input is amount <= df_len
            _validate_amount(amount, len(df.index))

            questions, answers = self._generate_questions_answers(df)

            self.data.append(
                DataHolder(
                    dataset_name=dataset_name,
                    questions=questions,
                    answers=answers,
                    total_entries=amount,
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


def _validate_amount(amount: int, df_len: int) -> None:
    if amount > df_len:
        raise IndexError(
            f"The value provided for 'amount' ({amount=} > {df_len=}) is too high. "
            f"Every 'qtype' must have at least as many entries as 'amount' "
            f"for 'BalancedSetGenerator' instances."
        )


def load_files_as_pds(self) -> None:
    """Loads all readable data files from a given directory into pandas DataFrames.

    Args:
        data_path (Path): Path object representing the base directory
            containing the data files.

    Returns:
        Dict[str, pd.DataFrame]: A dict of pandas DataFrames, where the 'key' equals to the name of the file.
    """
    files_found = [path for path in self._data_files_path.glob("*") if path.is_file()]

    results = {}
    for found in files_found:
        # Give us the file extension (.<ext>) and then remove the '.' leaving us only with <ext>
        file_extension = found.suffix.lstrip(".")

        read_method = getattr(pd, f"read_{file_extension}")
        if callable(read_method):
            results[found.stem] = read_method(found)

    return results
