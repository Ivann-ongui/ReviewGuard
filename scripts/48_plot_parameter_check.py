from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "hyperparameter_sensitivity.csv"
OUTPUT = ROOT / "paper" / "revision" / "paper_parameter_sensitivity_tvd.png"


def select(rows: pd.DataFrame, **fixed: float) -> pd.DataFrame:
    mask = pd.Series(True, index=rows.index)
    for column, value in fixed.items():
        mask &= rows[column] == value
    return rows[mask]


def main() -> None:
    rows = pd.read_csv(INPUT)
    panels = [
        ("$K$", "cv_splits", select(rows, midpoint=0.45, sharpness=10.0, cohort_weight_floor=0.05)),
        ("$\\tau$", "midpoint", select(rows, cv_splits=3, sharpness=10.0, cohort_weight_floor=0.05)),
        ("$\\alpha$", "sharpness", select(rows, cv_splits=3, midpoint=0.45, cohort_weight_floor=0.05)),
        ("$\\omega$", "cohort_weight_floor", select(rows, cv_splits=3, midpoint=0.45, sharpness=10.0)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25), sharey=True)
    defaults = {"cv_splits": 3, "midpoint": 0.45, "sharpness": 10.0, "cohort_weight_floor": 0.05}
    for ax, (title, column, subset) in zip(axes, panels):
        subset = subset.sort_values(column)
        colors = ["#D9822B" if value == defaults[column] else "#3E78A8" for value in subset[column]]
        ax.scatter(subset[column], subset["tvd_to_clean"], c=colors, s=35, zorder=3)
        for x_value, y_value in zip(subset[column], subset["tvd_to_clean"]):
            ax.annotate(
                f"{y_value:.4f}",
                (x_value, y_value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        ax.set_title(title)
        ax.set_xticks(subset[column].tolist())
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("TVD")
    axes[0].set_ylim(0.0043, 0.0238)
    fig.suptitle("One-at-a-time parameter check under one fixed attack stream and seed", fontsize=10)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D9822B", markeredgecolor="#D9822B", label="Default", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3E78A8", markeredgecolor="#3E78A8", label="Alternative", markersize=6),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.78))
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
