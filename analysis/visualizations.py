import seaborn as sns
import pandas as pd
import numpy as np
from utils import BASE_PATH


import matplotlib.pyplot as plt


def preprocess() -> pd.DataFrame: ...


def read_csvs():
    data_path = BASE_PATH / "data" / "results"

    files = [path for path in data_path.glob("**/*") if path.suffix == ".csv"]

    # This is very ugly :sunglasses:
    dataset_name, model_name = files[0].parents[1].name, files[0].parents[2].name

    dataframes = []
    for csv_file in files:
        qtype = csv_file.stem

        df = pd.read_csv(csv_file, index_col="Position", usecols=["Position", "Answer", "Score"])
        df["qtype"] = qtype
        df["Answer"] = df["Answer"].str.replace('"', "", regex=False)

        dataframes.append(df)

    df_concat = pd.concat(dataframes)
    count(df_concat)

    return df_concat


def _create_bar_plot(df: pd.DataFrame, qtype: str):

    df_plot = df.reset_index()

    plt.figure(figsize=(14, 8))
    plt.style.use("seaborn-v0_8-whitegrid")

    x_pos = np.arange(len(df_plot))
    width = 0.7

    colors = {"max": "#FF6B6B", "mean": "#3FB913", "min": "#2A90D0"}

    # Create overlapping bars with gradient effect
    bars_max = plt.bar(x_pos, df_plot["max"], width=width, alpha=0.4, label="Max", color=colors["max"], edgecolor="white", linewidth=1.5)

    bars_mean = plt.bar(
        x_pos, df_plot["mean"], width=width, alpha=0.8, label="Mean", color=colors["mean"], edgecolor="white", linewidth=1.5
    )

    bars_min = plt.bar(x_pos, df_plot["min"], width=width, alpha=0.9, label="Min", color=colors["min"], edgecolor="white", linewidth=1.5)

    # Add value labels on top of bars
    for i, (pos, mean_val, max_val, min_val) in enumerate(zip(x_pos, df_plot["mean"], df_plot["max"], df_plot["min"])):

        plt.text(pos, max_val + 0.1, f"{max_val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9, color=colors["max"])
        plt.text(pos, mean_val - 0.1, f"{mean_val:.1f}", ha="center", va="center", fontweight="bold", fontsize=9, color="black")
        plt.text(pos, min_val / 2, f"{min_val:.1f}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")

    plt.xticks(x_pos, df_plot["Position"], rotation=45, ha="right", fontsize=11)
    plt.xlabel(f"Question IDs for qtype: '{qtype}'", fontsize=14, fontweight="bold")
    plt.ylabel("Aggregated Score", fontsize=14, fontweight="bold")
    plt.title("Model Scoring Consistency", fontsize=16, fontweight="bold", pad=20)

    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=12, title="Score Statistics", title_fontsize=13)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.gca().set_facecolor("white")

    plt.show()


def count(df: pd.DataFrame) -> pd.DataFrame:
    qytpes = df["qtype"].unique()

    for qtype in qytpes:
        qtype_df = df[df["qtype"] == qtype]

        group_result = qtype_df["Score"].groupby(["Position"]).agg(["mean", "min", "max"])

        _create_bar_plot(group_result, qtype)


if __name__ == "__main__":
    read_csvs()
