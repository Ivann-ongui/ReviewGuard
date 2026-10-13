"""False-positive diagnostic for natural low-rating bursts on the Uber stream.

This is an additive diagnostic.  It does not alter the ReviewGuard method,
existing experiment scripts, paper sources, or any main-result artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_lib import (
    add_loss_mixture_scores,
    add_weight_formula_columns,
    build_canonical_scored_train,
    prepare_real_reviews,
    score_cohorts,
    split_real_reviews,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "uber_reviews_without_reviewid.csv"
RESULTS = ROOT / "results"
DATASET = "Uber"
BASE_WEIGHT_COL = "formula_product"


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def redact_example_text(value: object, limit: int = 200) -> str:
    """Remove common direct identifiers from diagnostic examples before export."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[redacted email]", text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d\-(). ]{6,}\d)(?!\w)", "[redacted phone]", text)
    return text[:limit]


def score_base_non_injected(analysis: pd.DataFrame) -> pd.DataFrame:
    """Run unchanged ReviewGuard base weighting on the non-injected split."""
    scored, _ = score_cohorts(analysis, seed=42)
    scored = add_loss_mixture_scores(scored, cv_splits=3, seed=42, weight_floor=0.05)
    scored = add_weight_formula_columns(scored, floor=0.03)
    scored["reviewguard_base_weight"] = scored[BASE_WEIGHT_COL].astype(float)
    return scored


def make_bucket_table(scored: pd.DataFrame, freq: str, min_count: int) -> tuple[pd.DataFrame, float]:
    work = scored.copy()
    work["bucket_start_time"] = pd.to_datetime(work["at"], errors="coerce").dt.floor(freq)
    low = work.loc[work["score"].astype(float) <= 2].copy()
    counts = (
        work.groupby("bucket_start_time", dropna=True)
        .size()
        .rename("total_review_count")
        .to_frame()
        .join(low.groupby("bucket_start_time", dropna=True).size().rename("low_rating_review_count"), how="left")
        .fillna({"low_rating_review_count": 0})
        .reset_index()
    )
    counts["low_rating_review_count"] = counts["low_rating_review_count"].astype(int)
    counts["low_rating_share"] = counts["low_rating_review_count"] / counts["total_review_count"]
    mean = float(counts["low_rating_review_count"].mean())
    std = float(counts["low_rating_review_count"].std(ddof=0))
    threshold = mean + 2.0 * std
    counts["selection_threshold"] = threshold
    counts["meets_count_threshold"] = counts["low_rating_review_count"] >= threshold
    counts["meets_min_count"] = counts["low_rating_review_count"] >= min_count
    counts["initial_candidate"] = counts["meets_count_threshold"] & counts["meets_min_count"]

    low["normalized_text"] = normalize_text(low["content"])
    duplicate = (
        low.groupby("bucket_start_time", dropna=True)["normalized_text"]
        .value_counts()
        .groupby(level=0)
        .max()
        .rename("max_duplicate_count")
        .reset_index()
    )
    duplicate = duplicate.merge(
        low.groupby("bucket_start_time", dropna=True).size().rename("low_rating_review_count_check").reset_index(),
        on="bucket_start_time",
        how="left",
    )
    duplicate["max_duplicate_share"] = duplicate["max_duplicate_count"] / duplicate["low_rating_review_count_check"]
    counts = counts.merge(
        duplicate[["bucket_start_time", "max_duplicate_share"]],
        on="bucket_start_time",
        how="left",
    )
    counts["max_duplicate_share"] = counts["max_duplicate_share"].fillna(0.0)
    counts["retained_natural_burst"] = counts["initial_candidate"] & (counts["max_duplicate_share"] <= 0.30)
    counts["exclusion_reason"] = np.select(
        [
            ~counts["initial_candidate"],
            counts["max_duplicate_share"] > 0.30,
        ],
        ["below_count_or_dispersion_threshold", "max_duplicate_share_gt_0.30"],
        default="retained",
    )
    return counts, threshold


def find_natural_bursts(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, float]:
    hourly, hourly_threshold = make_bucket_table(scored, "h", min_count=5)
    hourly_retained = hourly.loc[hourly["retained_natural_burst"]]
    if not hourly_retained.empty:
        return hourly, hourly_retained, "hourly", hourly_threshold

    daily, daily_threshold = make_bucket_table(scored, "D", min_count=15)
    daily_retained = daily.loc[daily["retained_natural_burst"]]
    return daily, daily_retained, "daily", daily_threshold


def group_duplicate_share(group: pd.DataFrame) -> float:
    if group.empty:
        return float("nan")
    return float(normalize_text(group["content"]).value_counts(normalize=True).max())


def group_row(
    group: pd.DataFrame,
    name: str,
    bucket_granularity: str,
    n_buckets: int | None = None,
    burst_low_count_mean: float | None = None,
) -> dict:
    weights = group["reviewguard_base_weight"].astype(float)
    return {
        "dataset": DATASET,
        "bucket_granularity_used": bucket_granularity,
        "group": name,
        "number_of_reviews": int(len(group)),
        "number_of_buckets": pd.NA if n_buckets is None else int(n_buckets),
        "mean_low_rating_count_per_retained_natural_burst_bucket": (
            np.nan if burst_low_count_mean is None else float(burst_low_count_mean)
        ),
        "max_duplicate_share": group_duplicate_share(group),
        "median_weight": float(weights.median()) if len(weights) else np.nan,
        "mean_weight": float(weights.mean()) if len(weights) else np.nan,
        "percent_weight_lt_0_3": float((weights < 0.3).mean() * 100) if len(weights) else np.nan,
        "percent_weight_lt_0_5": float((weights < 0.5).mean() * 100) if len(weights) else np.nan,
    }


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Auto-generated false-positive diagnostic; base ReviewGuard weights only.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{False-positive diagnostic on Uber natural low-rating bursts. Natural bursts are selected from the non-injected analysis split using only time-bucket counts and a diversity filter; lower weights for injected attacks indicate stronger coordinated evidence.}",
        "\\label{tab:natural_burst_false_positive}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Group & \\#Reviews & \\#Buckets & Max dup. share & Median $w$ & Mean $w$ & $\\%\,w<0.3$ & $\\%\,w<0.5$\\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        n_buckets = "--" if pd.isna(row["number_of_buckets"]) else str(int(row["number_of_buckets"]))
        group = str(row["group"]).replace("_", "\\_")
        lines.append(
            f"{group} & {int(row['number_of_reviews']):,} & {n_buckets} & "
            f"{row['max_duplicate_share']:.3f} & {row['median_weight']:.3f} & {row['mean_weight']:.3f} & "
            f"{row['percent_weight_lt_0_3']:.1f} & {row['percent_weight_lt_0_5']:.1f}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_paragraph(summary: pd.DataFrame, granularity: str, n_bursts: int) -> None:
    rows = summary.set_index("group")
    normal = rows.loc["Normal low-rating reviews"]
    natural = rows.loc["Natural low-rating burst"]
    injected = rows.loc["Injected coordinated attack"]
    text = (
        "\\paragraph{False-positive check on natural low-rating bursts.} "
        f"Using the non-injected Uber analysis split, we identified {n_bursts} text-diverse {granularity} "
        "low-rating burst bucket(s) without using attack labels. "
        f"Natural-burst reviews retained a median base contribution weight of {natural['median_weight']:.3f} "
        f"(mean {natural['mean_weight']:.3f}), compared with {normal['median_weight']:.3f} "
        f"(mean {normal['mean_weight']:.3f}) for other low-rating reviews and {injected['median_weight']:.3f} "
        f"(mean {injected['mean_weight']:.3f}) for the injected 20\\% coordinated-attack rows. "
        f"Only {natural['percent_weight_lt_0_3']:.1f}\\% of natural-burst reviews had $w<0.3$, versus "
        f"{injected['percent_weight_lt_0_3']:.1f}\\% of injected rows. This diagnostic does not establish "
        "spam ground truth; it indicates that a temporally concentrated low-rating event with diverse wording "
        "is not downweighted to the same extent as the coordinated injected stream."
    )
    (RESULTS / "natural_burst_false_positive_text.tex").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW)
    real = prepare_real_reviews(raw)
    split = split_real_reviews(real, seed=42)
    analysis = split.train_real.copy()
    if "review_id" not in analysis.columns:
        analysis["review_id"] = analysis["row_id"].astype(str)

    # Groups A/B: score only non-injected analysis reviews, as specified.
    scored_clean = score_base_non_injected(analysis)
    bucket_table, retained, granularity, _ = find_natural_bursts(scored_clean)
    scored_clean["bucket_start_time"] = pd.to_datetime(scored_clean["at"], errors="coerce").dt.floor(
        "h" if granularity == "hourly" else "D"
    )
    retained_buckets = set(retained["bucket_start_time"])
    low_clean = scored_clean.loc[scored_clean["score"].astype(float) <= 2].copy()
    natural = low_clean.loc[low_clean["bucket_start_time"].isin(retained_buckets)].copy()
    normal = low_clean.loc[~low_clean["bucket_start_time"].isin(retained_buckets)].copy()

    # Group C: unchanged 20% attack stream, base weights only, injected rows only.
    attack_train, _, _ = build_canonical_scored_train(raw, attack_ratio=0.20)
    attack_train["reviewguard_base_weight"] = attack_train[BASE_WEIGHT_COL].astype(float)
    injected = attack_train.loc[attack_train["is_injected_attack"].astype(int) == 1].copy()

    summary_rows = [
        group_row(normal, "Normal low-rating reviews", granularity),
        group_row(
            natural,
            "Natural low-rating burst",
            granularity,
            n_buckets=len(retained),
            burst_low_count_mean=float(retained["low_rating_review_count"].mean()) if len(retained) else None,
        ),
    ]
    type_labels = {
        "template_review_bomb": "Injected template attack",
        "paraphrase_review_bomb": "Injected paraphrase attack",
        "slow_drift_attack": "Injected slow-drift attack",
        "mixed_rating_camouflage": "Injected mixed-rating camouflage",
    }
    for attack_type, label in type_labels.items():
        part = injected.loc[injected["attack_type"] == attack_type]
        if not part.empty:
            summary_rows.append(group_row(part, label, granularity))
    summary_rows.append(group_row(injected, "Injected coordinated attack", granularity))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "natural_burst_false_positive_summary.csv", index=False, float_format="%.6f")

    candidate_windows = bucket_table.loc[bucket_table["initial_candidate"]].copy()
    candidate_windows.insert(0, "dataset", DATASET)
    candidate_windows.insert(1, "bucket_granularity", granularity)
    window_metrics = (
        low_clean.groupby("bucket_start_time")["reviewguard_base_weight"]
        .agg(
            median_weight="median",
            mean_weight="mean",
            percent_weight_lt_0_3=lambda s: (s < 0.3).mean() * 100,
            percent_weight_lt_0_5=lambda s: (s < 0.5).mean() * 100,
        )
        .reset_index()
    )
    candidate_windows = candidate_windows.merge(window_metrics, on="bucket_start_time", how="left")
    candidate_columns = [
        "dataset",
        "bucket_granularity",
        "bucket_start_time",
        "total_review_count",
        "low_rating_review_count",
        "low_rating_share",
        "max_duplicate_share",
        "median_weight",
        "mean_weight",
        "percent_weight_lt_0_3",
        "percent_weight_lt_0_5",
        "selection_threshold",
        "retained_natural_burst",
        "exclusion_reason",
    ]
    candidate_windows[candidate_columns].sort_values("bucket_start_time").to_csv(
        RESULTS / "natural_burst_candidate_windows.csv", index=False, float_format="%.6f"
    )

    examples = (
        natural.sort_values(["bucket_start_time", "review_id"])
        .groupby("bucket_start_time", group_keys=False)
        .head(5)
        .copy()
    )
    examples["dataset"] = DATASET
    examples["bucket_granularity"] = granularity
    examples["app_version"] = examples["reviewCreatedVersion"].fillna("UNKNOWN").astype(str)
    examples["text"] = examples["content"].map(redact_example_text)
    examples = examples.rename(columns={"at": "timestamp", "score": "rating", "reviewguard_base_weight": "ReviewGuard_weight"})
    examples = examples[
        [
            "dataset",
            "bucket_granularity",
            "bucket_start_time",
            "review_id",
            "timestamp",
            "rating",
            "app_version",
            "text",
            "ReviewGuard_weight",
        ]
    ]
    examples.to_csv(RESULTS / "natural_burst_examples.csv", index=False, float_format="%.6f")

    write_latex_table(summary, RESULTS / "natural_burst_false_positive_table.tex")
    write_paragraph(summary, granularity, len(retained))

    print(f"analysis_rows={len(analysis)}")
    print(f"granularity={granularity}; retained_bursts={len(retained)}; natural_reviews={len(natural)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
