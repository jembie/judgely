import seaborn as sns
import pandas as pd
import numpy as np
from utils import BASE_PATH
from typing import Set, List, Dict
from pathlib import Path
from itertools import product


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def contains_filetype(path: Path, filetype: str) -> Set[Path]:
    return {csv_file.parent for csv_file in path.rglob(f"*.{filetype}")}


def read_csvs() -> List[pd.DataFrame]:
    data_path = BASE_PATH / "data" / "results"
    csv_dirs = contains_filetype(data_path, "csv")

    subdir_frames = []
    for csv_dir in csv_dirs:
        dataframes = []

        csv_files = [f for f in csv_dir.glob("*.csv")]
        for csv_file in csv_files:

            qtype = csv_file.stem
            run_nr = csv_file.parent

            df = pd.read_csv(csv_file, index_col="Position", usecols=["Position", "Answer", "Score"])
            df["qtype"] = qtype
            df["Answer"] = df["Answer"].str.replace('"', "", regex=False)
            df["run_nr"] = run_nr.stem

            dataframes.append(df)

        if dataframes:
            subdir_frames.append(pd.concat(dataframes))

    return scatter_plot(subdir_frames)


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

        plt.text(pos, max_val, f"{max_val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9, color=colors["max"])
        plt.text(pos, mean_val - 0.1, f"{mean_val:.1f}", ha="center", va="center", fontweight="bold", fontsize=9, color="black")
        plt.text(pos, min_val - 0.1, f"{min_val:.1f}", ha="center", va="center", fontweight="bold", fontsize=9, color="white")

    plt.xticks(x_pos, df_plot["Position"], rotation=45, ha="right", fontsize=11)
    plt.xlabel(f"Question IDs for qtype: '{qtype}'", fontsize=14, fontweight="bold")
    plt.ylabel("Aggregated Score", fontsize=14, fontweight="bold")
    plt.title("Model Scoring Consistency", fontsize=16, fontweight="bold", pad=20)

    plt.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=12, title="Score Statistics", title_fontsize=13)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.gca().set_facecolor("white")

    # plt.show()
    img_path = BASE_PATH / "data" / "img"
    img_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{img_path}/{qtype}.svg", format="svg")


def count(df: pd.DataFrame) -> pd.DataFrame:
    qytpes = df["qtype"].unique()

    for qtype in qytpes:
        qtype_df = df[df["qtype"] == qtype]

        group_result = qtype_df["Score"].groupby(["Position"]).agg(["mean", "min", "max"])

        _create_bar_plot(group_result, qtype)


def _make_subset(df: pd.DataFrame, run: str, qtype: str) -> pd.DataFrame:
    replacement_map = {
        "No semantic relation at all meaning": 1.0,
        "Same domain, but no matching semantical meaning": 2.0,
        "Some matching semantical meaning": 3.0,
        "Great match in semantical meaning": 4.0,
        "Identical semantic meaning": 5.0,
    }

    mask = (df["run_nr"] == run) & (df["qtype"] == qtype)
    subset = df.loc[mask]

    subset.Answer.replace(to_replace=replacement_map, inplace=True)
    return subset


def _create_scatter(df: pd.DataFrame, qtype):

    # Reset index to get Position as a column if needed
    df_plot = df.reset_index()

    plt.figure(figsize=(14, 8))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create x positions (numeric positions for plotting)
    x_pos = np.arange(len(df_plot))

    # Create a color palette with different colors for each unique Position/index
    unique_positions = df_plot["Position"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_positions)))  # Use tab10 colormap
    color_map = dict(zip(unique_positions, colors))

    # Create scatter plot with both Score and Answer
    for i, (pos_idx, row) in enumerate(df_plot.iterrows()):
        position = row["Position"]
        color = color_map[position]

        _plot_scatter(i, row, "Score", "_", position, color)
        _plot_scatter(i, row, "Answer", "|", position, color)

    # Set x-axis labels to exact Position values (same as bar plot)
    plt.xticks(x_pos, df_plot["Position"], ha="right", fontsize=11)
    plt.yticks(range(6))
    plt.ylim(0, 5.5)

    # Customize the plot
    plt.xlabel("Question IDs", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title(f"Scoring and Answer Consistency for {qtype}", fontsize=16, fontweight="bold", pad=20)

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)  # Make room for legend
    plt.gca().set_facecolor("white")

    # Make Custom Legends
    legend_elements = [
        Line2D([0], [0], marker="_", color="gray", markersize=15, linestyle="None", label="Score"),
        Line2D([0], [0], marker="|", color="gray", markersize=15, linestyle="None", label="Axis"),
    ]
    plt.legend(handles=legend_elements, loc="lower right", frameon=True, fancybox=True, shadow=True)

    plt.show()


def _plot_scatter(i, row, column, marker, position, color):
    plt.scatter(
        i,
        row[column],
        color=color,
        marker=marker,
        s=250,
        alpha=0.8,
        edgecolors="white",
        linewidth=1.5,
        label=f"Score - {position}" if i == 0 else "",
    )


def scatter_plot(dfs: List[pd.DataFrame]):

    for df in dfs:

        qtypes = df["qtype"].unique()
        runs = df["run_nr"].unique()

        cartesian_product = product(runs, qtypes)

        for run, qtype in cartesian_product:
            subset = _make_subset(df=df, run=run, qtype=qtype).sort_index()

            _create_scatter(subset, qtype)


if __name__ == "__main__":
    read_csvs()
