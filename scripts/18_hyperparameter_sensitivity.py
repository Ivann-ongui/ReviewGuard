from pathlib import Path

import pandas as pd

from experiment_lib import build_canonical_scored_train, label_distribution, total_variation_distance


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "uber_reviews_without_reviewid.csv"
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW)
    rows = []
    out_path = REPORTS / "hyperparameter_sensitivity.csv"
    configs = []
    for midpoint in [0.35, 0.45, 0.55]:
        configs.append({"midpoint": midpoint, "sharpness": 10.0, "cv_splits": 3, "cohort_weight_floor": 0.05})
    for sharpness in [6.0, 10.0, 14.0]:
        configs.append({"midpoint": 0.45, "sharpness": sharpness, "cv_splits": 3, "cohort_weight_floor": 0.05})
    for cv_splits in [3, 5]:
        configs.append({"midpoint": 0.45, "sharpness": 10.0, "cv_splits": cv_splits, "cohort_weight_floor": 0.05})
    for floor in [0.01, 0.03, 0.05]:
        configs.append({"midpoint": 0.45, "sharpness": 10.0, "cv_splits": 3, "cohort_weight_floor": floor})

    unique_configs = [dict(t) for t in {tuple(sorted(c.items())) for c in configs}]
    for config in unique_configs:
        train, clean_dist, detection = build_canonical_scored_train(
            raw,
            0.20,
            sigmoid_midpoint=config["midpoint"],
            sigmoid_sharpness=config["sharpness"],
            cv_splits=int(config["cv_splits"]),
            cohort_weight_floor=config["cohort_weight_floor"],
        )
        dist = label_distribution(train["sentiment"], train["cohort_dm_sample_weight"])
        rows.append(
            {
                **config,
                "tvd_to_clean": total_variation_distance(clean_dist, dist),
                "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                **detection,
            }
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"done {config}")
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    summary = out.agg({"tvd_to_clean": ["mean", "std", "min", "max"], "negative_abs_error": ["mean", "std", "min", "max"]})
    summary.to_csv(REPORTS / "hyperparameter_sensitivity_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
