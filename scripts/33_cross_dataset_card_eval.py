from pathlib import Path

import pandas as pd

from experiment_lib import build_canonical_scored_train, calibrate_contribution_weights, label_distribution, total_variation_distance


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASETS = {
    "uber": ROOT / "uber_reviews_without_reviewid.csv",
    "google_play_app_reviews": ROOT / "data" / "processed" / "google_play_app_reviews_standardized.csv",
}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rows = []
    for dataset_name, path in DATASETS.items():
        raw = pd.read_csv(path)
        for ratio_label, ratio in [("10pct", 0.10), ("20pct", 0.20)]:
            train, clean_dist, detection = build_canonical_scored_train(raw, ratio)
            rows_count = int((train["is_injected_attack"] == 1).sum())
            train["card_t_weight"] = calibrate_contribution_weights(train, "cohort_dm_sample_weight")
            methods = {
                "No defense": None,
                "Cohort only": "sample_weight",
                "Loss-mixture only": "dm_sample_weight",
                "CARD": "cohort_dm_sample_weight",
                "CARD-T": "card_t_weight",
            }
            for method, weight_col in methods.items():
                dist = label_distribution(train["sentiment"], train[weight_col] if weight_col else None)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "attack_setting": ratio_label,
                        "attack_rows": rows_count,
                        "method": method,
                        "tvd_to_clean": total_variation_distance(clean_dist, dist),
                        "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                        **detection,
                    }
                )
            print(f"finished dataset={dataset_name} setting={ratio_label}")
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "cross_dataset_card_eval.csv", index=False)
    pivot = out.pivot_table(index=["dataset", "method"], columns="attack_setting", values="tvd_to_clean")
    pivot.to_csv(REPORTS / "table_cross_dataset_card_tvd.csv")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
