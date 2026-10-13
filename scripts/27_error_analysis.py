from pathlib import Path

import pandas as pd

from experiment_lib import build_canonical_scored_train


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "uber_reviews_without_reviewid.csv"
REPORTS = ROOT / "reports"


KEEP_COLS = [
    "row_id",
    "attack_type",
    "is_injected_attack",
    "score",
    "sentiment",
    "content",
    "at",
    "reviewCreatedVersion",
    "contamination_score",
    "sample_weight",
    "dm_clean_probability",
    "dm_sample_weight",
    "cohort_dm_sample_weight",
    "burst_score",
    "duplicate_score",
    "low_rating_cluster_score",
    "version_concentration_score",
    "semantic_compactness_score",
]


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW)
    train, _, _ = build_canonical_scored_train(raw, 0.20)

    high_risk_threshold = train["contamination_score"].quantile(0.90)
    low_risk_threshold = train["contamination_score"].quantile(0.25)
    high_dm_threshold = train["dm_clean_probability"].quantile(0.90)
    low_dm_threshold = train["dm_clean_probability"].quantile(0.10)

    false_positive = train[(train["is_injected_attack"] == 0) & (train["contamination_score"] >= high_risk_threshold)]
    false_negative = train[(train["is_injected_attack"] == 1) & (train["contamination_score"] <= low_risk_threshold)]
    high_dm_attack = train[(train["is_injected_attack"] == 1) & (train["dm_clean_probability"] >= high_dm_threshold)]
    low_dm_clean = train[(train["is_injected_attack"] == 0) & (train["dm_clean_probability"] <= low_dm_threshold)]

    cases = {
        "error_analysis_false_positive.csv": false_positive.sort_values("contamination_score", ascending=False).head(100),
        "error_analysis_false_negative.csv": false_negative.sort_values("contamination_score", ascending=True).head(100),
        "error_analysis_high_dm_attack.csv": high_dm_attack.sort_values("dm_clean_probability", ascending=False).head(100),
        "error_analysis_low_dm_clean.csv": low_dm_clean.sort_values("dm_clean_probability", ascending=True).head(100),
    }
    for name, df in cases.items():
        df[[c for c in KEEP_COLS if c in df.columns]].to_csv(REPORTS / name, index=False)

    target_texts = [
        "Uber is unusable after the old update. Driver cancels and I still get charged.",
        "So far my experience with Uber has been awesome",
    ]
    train[train["content"].isin(target_texts)][
        [c for c in KEEP_COLS if c in train.columns]
    ].to_csv(REPORTS / "review_level_target_examples.csv", index=False)

    summary = pd.DataFrame(
        [
            {"case": "high_risk_clean", "rows": len(false_positive), "threshold": high_risk_threshold},
            {"case": "low_risk_attack", "rows": len(false_negative), "threshold": low_risk_threshold},
            {"case": "high_dm_attack", "rows": len(high_dm_attack), "threshold": high_dm_threshold},
            {"case": "low_dm_clean", "rows": len(low_dm_clean), "threshold": low_dm_threshold},
        ]
    )
    summary.to_csv(REPORTS / "error_analysis_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
