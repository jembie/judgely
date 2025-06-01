import pandas as pd
from typing import Dict, List
import numpy as np

from constants import BASE_PATH
from pathlib import Path


class TestSetGenerator:

    def __init__(self):
        self.csv_dict: Dict[str, pd.DataFrame] = {}
        self.csv_dir = Path(BASE_PATH, "data", "csv")

    def load_csvs(self) -> bool:
        csv_files = [path for path in self.csv_dir.glob("**/*") if path.is_file()]

        if not csv_files:
            print(f"No files were found within the {self.csv_dir.name} directory, aborting...")
            return False

        for csv in csv_files:
            dataset_name = csv.stem
            self.csv_dict[dataset_name] = pd.read_csv(csv)
            return True

    def generate_set(self, seed: int = 42, amount: int = 10) -> pd.DataFrame:
        rng = np.random.default_rng(seed=seed)

        results = {}
        for dataset_name, df in self.csv_dict.items():
            df_len = len(df) - 1

            # Generate a np.ndarray[amount] with values ranging within the interval of [0, LAST_ROW]
            random_sequence = rng.integers(low=0, high=df_len, size=amount)
            results[dataset_name] = df.iloc[random_sequence]

        return results


if __name__ == "__main__":

    t = TestSetGenerator()

    t.load_csvs()
    t.generate_set()
