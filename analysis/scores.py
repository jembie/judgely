import pandas as pd
from pathlib import Path
from utils import BASE_PATH


class ScoreComparison:
    def __init__(self, csv_dir_path: Path | str = ""):
        self.dfs: pd.DataFrame = None
        self.csv_dir = csv_dir_path if csv_dir_path else Path(BASE_PATH, "data", "results")
        self._load_csvs()

    def _load_csvs(self):
        csv_files = [path for path in self.csv_dir.glob("**/*") if path.is_file() and path.suffix == ".csv"]

        csvs = []
        if not csv_files:
            print(f"No files were found within the {self.csv_dir.name} directory, aborting...")
            return None

        for csv in csv_files:
            csvs.append(pd.read_csv(csv, usecols=["Score", "Answer"]))

        self.dfs = pd.concat(csvs)

    def count(self):
        answers = self.dfs["Answer"].str.replace('"', "", regex=False)
        print(answers.value_counts(dropna=False))

        score = self.dfs["Score"]
        print(score.value_counts(dropna=False))

        # Check if any textual scores had a higher score than numericals.
        replacement_map = {
            "No semantic relation at all meaning": 1.0,
            "Same domain, but no matching semantical meaning": 2.0,
            "Some matching semantical meaning": 3.0,
            "Great match in semantical meaning": 4.0,
            "Identical semantic meaning": 5.0,
        }
        self.dfs["Answer"] = self.dfs["Answer"].str.replace('"', "", regex=False).replace(replacement_map).fillna(0.0)

        subset = self.dfs[self.dfs["Answer"] > self.dfs["Score"]]
        print(subset.value_counts())
