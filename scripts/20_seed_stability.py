from pathlib import Path

import pandas as pd

from experiment_lib import build_scored_train, label_distribution, total_variation_distance


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "uber_reviews_without_reviewid.csv"
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW)
    rows = []
    attack_rows = int(round(9000 * 0.20))
    for seed in [1, 2, 3, 4, 5]:
        train, clean_dist, detection = build_scored_train(raw, attack_rows, seed=20000 + seed)
        dist = label_distribution(train["sentiment"], train["cohort_dm_sample_weight"])
        rows.append(
            {
                "seed": seed,
                "tvd_to_clean": total_variation_distance(clean_dist, dist),
                "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                **detection,
            }
        )
        print(f"finished seed={seed}")
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "seed_stability.csv", index=False)
    summary = out.agg(["mean", "std", "min", "max"])
    summary.to_csv(REPORTS / "seed_stability_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
