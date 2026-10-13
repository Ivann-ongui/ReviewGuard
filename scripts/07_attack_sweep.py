from pathlib import Path
import json

import pandas as pd

from experiment_lib import (
    bootstrap_tvd_ci,
    build_canonical_scored_train,
    method_distributions,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "uber_reviews_without_reviewid.csv"
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW)
    attack_ratios = [0.05, 0.10, 0.20, 0.30, 0.40]
    all_rows = []
    ci_rows = []

    for ratio in attack_ratios:
        train, clean_dist, detection = build_canonical_scored_train(raw, ratio)
        attack_rows = int((train["is_injected_attack"] == 1).sum())

        for row in method_distributions(train, clean_dist):
            row.update(
                {
                    "attack_ratio": ratio,
                    "attack_rows": attack_rows,
                    **detection,
                }
            )
            all_rows.append(row)

        for method, weight_col in [
            ("polluted_unweighted", None),
            ("cohort_weighted", "sample_weight"),
            ("cohort_dm_weighted", "cohort_dm_sample_weight"),
            ("cohort_cl_dm_weighted", "cohort_cl_dm_sample_weight"),
        ]:
            ci = bootstrap_tvd_ci(train, clean_dist, method, weight_col, n_bootstrap=100, seed=7)
            ci.update({"attack_ratio": ratio, "attack_rows": attack_rows})
            ci_rows.append(ci)

        print(f"finished ratio={ratio:.2f} attack_rows={attack_rows}")

    results = pd.DataFrame(all_rows)
    ci = pd.DataFrame(ci_rows)
    results.to_csv(REPORTS / "attack_sweep_results.csv", index=False)
    ci.to_csv(REPORTS / "attack_sweep_bootstrap_ci.csv", index=False)

    best = (
        results.sort_values(["attack_ratio", "tvd_to_clean"])
        .groupby("attack_ratio")
        .head(3)[["attack_ratio", "method", "tvd_to_clean", "negative_abs_error"]]
    )
    summary = {
        "attack_ratios": attack_ratios,
        "rows": int(len(results)),
        "best_methods_by_ratio": best.to_dict(orient="records"),
    }
    (REPORTS / "attack_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(best.to_string(index=False))
    print(f"wrote {REPORTS / 'attack_sweep_results.csv'}")


if __name__ == "__main__":
    main()
