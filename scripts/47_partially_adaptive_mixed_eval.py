"""Evaluate the partially_adaptive_mixed attack on Uber at a 20% ratio.

The attack generator is deliberately non-white-box: it varies topical wording,
time, ratings, and app versions without consulting ReviewGuard scores.  This
script runs only base ReviewGuard weighting (the product of group and loss
reliability); it does not invoke simulated verified-sample calibration.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_lib import (
    build_canonical_scored_train,
    label_distribution,
    total_variation_distance,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "uber_reviews_without_reviewid.csv"
RESULTS = ROOT / "results"
ATTACK_TYPE = "partially_adaptive_mixed"
ATTACK_RATIO = 0.20


def normalized_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW)
    train, reference, _ = build_canonical_scored_train(
        raw,
        attack_ratio=ATTACK_RATIO,
        only_attack_type=ATTACK_TYPE,
    )
    injected = train.loc[train["is_injected_attack"].astype(int) == 1].copy()
    injected["normalized_text"] = normalized_text(injected["content"])
    injected["timestamp"] = pd.to_datetime(injected["at"], errors="coerce")
    injected["hour"] = injected["timestamp"].dt.floor("h")
    injected["reviewguard_base_weight"] = injected["formula_product"].astype(float)

    # Constraint checks are intentionally explicit so this attack remains
    # reproducible and auditable rather than an implicit white-box tweak.
    expected_attack_rows = 1800
    expected_ratings = {1: 900, 2: 540, 3: 180, 4: 180}
    observed_ratings = injected["score"].astype(int).value_counts().sort_index().to_dict()
    raw_versions = raw["reviewCreatedVersion"].dropna().astype(str)
    top_versions = raw_versions[raw_versions.str.strip().ne("") & raw_versions.ne("UNKNOWN")].value_counts().head(3)
    assert len(injected) == expected_attack_rows
    assert injected["normalized_text"].nunique() == len(injected)
    assert observed_ratings == expected_ratings
    assert set(injected["reviewCreatedVersion"].dropna().astype(str)).issubset(set(top_versions.index))
    assert (injected["timestamp"].max() - injected["timestamp"].min()) <= pd.Timedelta(days=6)

    no_defense_tvd = total_variation_distance(reference, label_distribution(train["sentiment"]))
    reviewguard_tvd = total_variation_distance(
        reference,
        label_distribution(train["sentiment"], train["formula_product"]),
    )
    relative_reduction = 100.0 * (no_defense_tvd - reviewguard_tvd) / no_defense_tvd
    version_share = injected["reviewCreatedVersion"].fillna("UNKNOWN").astype(str).value_counts(normalize=True)
    hourly_counts = injected.groupby("hour").size()
    duplicate_share = injected["normalized_text"].value_counts(normalize=True).max()

    summary = pd.DataFrame(
        [
            {
                "dataset": "Uber",
                "attack_type": ATTACK_TYPE,
                "attack_ratio": ATTACK_RATIO,
                "analysis_reviews": int(len(train) - len(injected)),
                "injected_reviews": int(len(injected)),
                "no_defense_tvd": no_defense_tvd,
                "reviewguard_base_tvd": reviewguard_tvd,
                "relative_tvd_reduction_percent": relative_reduction,
                "injected_attack_median_weight": float(injected["reviewguard_base_weight"].median()),
                "injected_attack_mean_weight": float(injected["reviewguard_base_weight"].mean()),
            }
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "max_duplicate_share", "value": float(duplicate_share)},
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "max_hourly_injected_count", "value": int(hourly_counts.max())},
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "dominant_version_share", "value": float(version_share.iloc[0])},
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "low_rating_share", "value": float((injected["score"].astype(int) <= 2).mean())},
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "number_of_unique_normalized_texts", "value": int(injected["normalized_text"].nunique())},
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "campaign_start", "value": injected["timestamp"].min().isoformat(sep=" ")},
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "campaign_end", "value": injected["timestamp"].max().isoformat(sep=" ")},
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "top_3_real_versions", "value": " | ".join(top_versions.index.tolist())},
            {"dataset": "Uber", "attack_type": ATTACK_TYPE, "metric": "injected_rating_counts", "value": "1-star:900; 2-star:540; 3-star:180; 4-star:180"},
        ]
    )
    audit_rows = injected[
        [
            "row_id",
            "attack_type",
            "content",
            "normalized_text",
            "score",
            "timestamp",
            "reviewCreatedVersion",
            "reviewguard_base_weight",
        ]
    ].sort_values("row_id")

    summary.to_csv(RESULTS / "partially_adaptive_mixed_20pct_summary.csv", index=False, float_format="%.6f")
    diagnostics.to_csv(RESULTS / "partially_adaptive_mixed_20pct_diagnostics.csv", index=False)
    audit_rows.to_csv(RESULTS / "partially_adaptive_mixed_20pct_injected_weights.csv", index=False, float_format="%.6f")

    print(summary.to_string(index=False))
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
