from pathlib import Path

import pandas as pd

from experiment_lib import build_canonical_scored_train, calibrate_contribution_weights, label_distribution, total_variation_distance


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "uber_reviews_without_reviewid.csv"
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW)
    rows = []
    for ratio in [0.05, 0.10, 0.20, 0.30, 0.40]:
        train, clean_dist, detection = build_canonical_scored_train(raw, ratio)
        train["trusted_calibrated_card_weight"] = calibrate_contribution_weights(train, "cohort_dm_sample_weight")
        trusted_clean_rows = sum(max(1, int(round(len(group) * 0.05))) for _, group in train[train["is_injected_attack"] == 0].groupby("sentiment"))

        high_dm_threshold = train["dm_clean_probability"].quantile(0.90)
        high_dm_attack = train[(train["is_injected_attack"] == 1) & (train["dm_clean_probability"] >= high_dm_threshold)]
        for method, col in [
            ("No defense", None),
            ("Loss-mixture only", "dm_sample_weight"),
            ("CARD", "cohort_dm_sample_weight"),
            ("CARD-T", "trusted_calibrated_card_weight"),
        ]:
            dist = label_distribution(train["sentiment"], train[col] if col else None)
            rows.append(
                {
                    "attack_ratio": ratio,
                    "method": method,
                    "trusted_clean_ratio": 0.05,
                    "trusted_clean_rows": trusted_clean_rows,
                    "tvd_to_clean": total_variation_distance(clean_dist, dist),
                    "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                    "mean_weight_clean": 1.0 if col is None else float(train.loc[train["is_injected_attack"] == 0, col].mean()),
                    "mean_weight_attack": 1.0 if col is None else float(train.loc[train["is_injected_attack"] == 1, col].mean()),
                    "mean_weight_high_dm_attack": 1.0 if col is None else (float(high_dm_attack[col].mean()) if len(high_dm_attack) else 0.0),
                    "high_dm_attack_rows": int(len(high_dm_attack)),
                    **detection,
                }
            )
        print(f"finished trusted calibration ratio={ratio}")
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "trusted_calibrated_card_eval.csv", index=False)
    pivot = out.pivot_table(index="method", columns="attack_ratio", values="tvd_to_clean")
    pivot.to_csv(REPORTS / "table_trusted_calibrated_card_tvd.csv")
    print(out[["attack_ratio", "method", "tvd_to_clean", "mean_weight_high_dm_attack"]].to_string(index=False))


if __name__ == "__main__":
    main()
