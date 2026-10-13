from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_lib import label_distribution, prepare_real_reviews, split_real_reviews


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "attack_sweep_results.csv"
RAW = ROOT / "uber_reviews_without_reviewid.csv"
OUTPUT = ROOT / "paper" / "revision" / "intro_problem_baseline_example.png"


def main() -> None:
    raw = pd.read_csv(RAW)
    split = split_real_reviews(prepare_real_reviews(raw), seed=42)
    clean = label_distribution(split.train_real["sentiment"])

    results = pd.read_csv(RESULTS)
    subset = results[
        (results["attack_ratio"] == 0.20)
        & results["method"].isin(
            ["polluted_unweighted", "cohort_weighted", "dm_weighted", "cohort_dm_weighted"]
        )
    ].set_index("method")

    rows = [
        ("Pre-injection reference", clean, 0.0),
        ("No defense", subset.loc["polluted_unweighted"], subset.loc["polluted_unweighted", "tvd_to_clean"]),
        ("Group-only weighting", subset.loc["cohort_weighted"], subset.loc["cohort_weighted", "tvd_to_clean"]),
        ("Loss-only weighting", subset.loc["dm_weighted"], subset.loc["dm_weighted", "tvd_to_clean"]),
        ("ReviewGuard", subset.loc["cohort_dm_weighted"], subset.loc["cohort_dm_weighted", "tvd_to_clean"]),
    ]

    labels = [row[0] for row in rows]
    values = []
    tvds = []
    for _, row, tvd in rows:
        if isinstance(row, dict):
            values.append([row["negative"], row["neutral"], row["positive"]])
        else:
            values.append([row["negative_rate"], row["neutral_rate"], row["positive_rate"]])
        tvds.append(float(tvd))
    values = np.asarray(values)

    colors = ["#C84B4B", "#A5ABB3", "#3B78A8"]
    categories = ["Negative", "Neutral", "Positive"]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.4, 3.15))
    left = np.zeros(len(labels))
    for idx, (category, color) in enumerate(zip(categories, colors)):
        ax.barh(y, values[:, idx], left=left, height=0.58, color=color, label=category)
        left += values[:, idx]

    for idx, tvd in enumerate(tvds):
        ax.text(1.015, idx, f"TVD {tvd:.3f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("Rating-derived sentiment proportion")
    ax.set_xticks(np.linspace(0, 1, 6), ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.48, 1.18), frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#D9DDE2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
