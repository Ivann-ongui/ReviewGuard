from pathlib import Path

import pandas as pd

from experiment_lib import DEFAULT_COHORT_FEATURE_WEIGHTS, build_canonical_scored_train, label_distribution, total_variation_distance


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "uber_reviews_without_reviewid.csv"
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW)
    rows = []
    variants = {"all_features": DEFAULT_COHORT_FEATURE_WEIGHTS}
    for feature in DEFAULT_COHORT_FEATURE_WEIGHTS:
        weights = dict(DEFAULT_COHORT_FEATURE_WEIGHTS)
        weights[feature] = 0.0
        variants[f"without_{feature}"] = weights
    for name, weights in variants.items():
        train, clean_dist, detection = build_canonical_scored_train(raw, 0.20, feature_weights=weights)
        dist = label_distribution(train["sentiment"], train["cohort_dm_sample_weight"])
        rows.append(
            {
                "variant": name,
                "tvd_to_clean": total_variation_distance(clean_dist, dist),
                "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                "mean_weight_clean": float(train.loc[train["is_injected_attack"] == 0, "cohort_dm_sample_weight"].mean()),
                "mean_weight_attack": float(train.loc[train["is_injected_attack"] == 1, "cohort_dm_sample_weight"].mean()),
                **detection,
            }
        )
        print(f"finished {name}")
    out = pd.DataFrame(rows).sort_values("tvd_to_clean")
    out.to_csv(REPORTS / "cohort_feature_ablation.csv", index=False)
    print(out[["variant", "tvd_to_clean", "attack_detection_roc_auc", "top_15_percent_precision"]].to_string(index=False))


if __name__ == "__main__":
    main()
