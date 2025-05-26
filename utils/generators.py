import pandas as pd
from typing import Dict, List

from constants import BASE_PATH
from pathlib import Path


class TestSetGenerator:

    def __init__(self):
        self.csv_dict = {}
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


if __name__ == "__main__":

    t = TestSetGenerator()

    t.load_csvs()
