from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "table_representation_comparison_20pct_tvd.csv"
OUTPUT = ROOT / "paper" / "revision" / "paper_cross_dataset_representation_tvd.png"


def main() -> None:
    results = pd.read_csv(INPUT)
    dataset_names = {
        "uber": "Uber",
        "google_play_app_reviews": "Google Apps",
    }
    representation_names = {
        "TF-IDF cohort": "TF-IDF",
        "intfloat/multilingual-e5-small": "E5",
    }
    methods = ["No defense", "CARD", "CARD-T"]
    method_labels = ["No defense", "ReviewGuard", "ReviewGuard (sim. verified)"]
    colors = ["#9AA1A9", "#3E78A8", "#D9822B"]
    combinations = [
        ("uber", "TF-IDF cohort"),
        ("uber", "intfloat/multilingual-e5-small"),
        ("google_play_app_reviews", "TF-IDF cohort"),
        ("google_play_app_reviews", "intfloat/multilingual-e5-small"),
    ]

    values = np.zeros((len(combinations), len(methods)))
    for group_index, (dataset, representation) in enumerate(combinations):
        subset = results[
            (results["dataset"] == dataset)
            & (results["representation"] == representation)
        ].set_index("method")
        for method_index, method in enumerate(methods):
            values[group_index, method_index] = float(subset.loc[method, "tvd_to_clean"])

    x = np.arange(len(combinations))
    width = 0.23
    fig, ax = plt.subplots(figsize=(7.4, 3.25))

    for method_index, (label, color) in enumerate(zip(method_labels, colors)):
        positions = x + (method_index - 1) * width
        bars = ax.bar(
            positions,
            values[:, method_index],
            width,
            color=color,
            label=label,
        )
        for bar, value in zip(bars, values[:, method_index]):
            ax.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.7,
            )

    ax.set_ylabel("TVD")
    ax.set_ylim(0, 0.162)
    ax.set_xticks(
        x,
        [
            f"{dataset_names[dataset]}\n{representation_names[representation]}"
            for dataset, representation in combinations
        ],
    )
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    ax.grid(axis="y", color="#D9DDE2", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
