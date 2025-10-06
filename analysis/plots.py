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


def make_plots() -> List[pd.DataFrame]:
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


def create_scatter(df: pd.DataFrame, qtype, img_path: Path):
    # Reset index to get Position as a column if needed
    df_plot = df.reset_index()
    plt.figure(figsize=(14, 8))
    plt.style.use("seaborn-v0_8-whitegrid")
    unique_positions = df_plot["Position"].unique()
    unique_runs = sorted(df_plot["run_nr"].unique())

    # Create x positions with more spacing between positions
    x_spacing = 5
    x_pos = np.arange(len(unique_positions)) * x_spacing

    # Create a color palette with different colors for each unique run_nr
    colors = ["#2aa199", "#ff3b5b", "#433bff", "#ffad3b", "#8e19cd"]
    colors_cycled = [colors[i % len(colors)] for i in range(len(unique_runs))]
    color_map = dict(zip(unique_runs, colors_cycled))

    # CRITICAL: offset_range must be smaller than x_spacing to prevent overlap
    offset_range = 3
    offsets = np.linspace(-offset_range / 2, offset_range / 2, len(unique_runs))
    offset_map = dict(zip(unique_runs, offsets))

    # Create scatter plot with both Score and Answer
    for run_nr in unique_runs:
        run_data = df_plot[df_plot["run_nr"] == run_nr]
        color = color_map[run_nr]
        offset = offset_map[run_nr]
        for _, row in run_data.iterrows():
            # Find the x position for this Position
            pos_index = np.where(unique_positions == row["Position"])[0][0]
            # Plot with offset and new spacing
            x_position = x_pos[pos_index] + offset
            plot_scatter(x_position, row, "Score", "_", color)
            plot_scatter(x_position, row, "Answer", "|", color)

    # Set x-axis labels to exact Position values with new spacing
    plt.xlim(-offset_range / 2 - 1, x_pos[-1] + offset_range / 2 + 2.5)  # Added more right margin
    plt.xticks(x_pos, unique_positions, ha="right", fontsize=11)
    plt.yticks(range(6))
    plt.ylim(0, 5.5)

    # Customize the plot
    plt.xlabel("Question IDs", fontsize=14, fontweight="bold")
    plt.ylabel("Scoring Range", fontsize=14, fontweight="bold")
    plt.title(f"Consistency of Scoring between Numerical and Textual Scoring for {qtype}", fontsize=16, fontweight="bold", pad=20)
    plt.grid(True, alpha=0.35, linestyle="--")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)
    plt.gca().set_facecolor("white")

    # Make Custom Legends
    legend_elements = [
        Line2D([0], [0], marker="_", color="gray", markersize=15, linestyle="None", label="Numerical Score"),
        Line2D([0], [0], marker="|", color="gray", markersize=15, linestyle="None", label="Textual Score"),
    ]
    for run_nr in unique_runs:
        color = color_map[run_nr]
        legend_elements.append(Line2D([0], [0], marker="o", color=color, markersize=10, linestyle="None", label=f"Iteration {run_nr[-1]}"))
    plt.legend(handles=legend_elements, loc="lower right", frameon=True, fancybox=True, shadow=True)

    img_path.mkdir(exist_ok=True, parents=True)

    plt.tight_layout()
    plt.savefig(f"{img_path}/{qtype}.svg", format="svg")


def plot_scatter(x_position, row, column, marker, color):
    plt.scatter(
        x_position,
        row[column],
        color=color,
        marker=marker,
        s=250,
        alpha=0.8,
        edgecolors="white",
        linewidth=2,
    )


def scatter_plot(dfs: List[pd.DataFrame], img_path: Path = BASE_PATH / "data" / "img" / "scatterplot"):
    replacement_map = {
        "No semantic relation at all meaning": 1.0,
        "Same domain, but no matching semantical meaning": 2.0,
        "Some matching semantical meaning": 3.0,
        "Great match in semantical meaning": 4.0,
        "Identical semantic meaning": 5.0,
    }

    merged = pd.concat(dfs)
    merged.Answer = merged.Answer.replace((replacement_map))

    qtypes = merged["qtype"].unique()

    # Group by qtype and plot all runs together
    for qtype in qtypes:
        # Get all data for this qtype across all runs
        qtype_data = merged[merged["qtype"] == qtype]
        create_scatter(qtype_data, qtype, img_path=img_path)

    print(f"\nSaved all files within: '{img_path}'\n")
