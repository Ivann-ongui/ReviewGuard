from pathlib import Path

import numpy as np
import pandas as pd

from experiment_lib import (
    add_weight_formula_columns,
    build_canonical_scored_train,
    label_distribution,
    total_variation_distance,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASETS = {
    "Uber": ROOT / "uber_reviews_without_reviewid.csv",
    "Google Apps": ROOT / "data" / "processed" / "google_play_app_reviews_standardized.csv",
}


def summarize_dataset(dataset: str, path: Path, attack_ratio: float = 0.20) -> list[dict]:
    raw = pd.read_csv(path)
    train, clean_dist, _ = build_canonical_scored_train(raw, attack_ratio)
    train = add_weight_formula_columns(train)

    cohort = train["sample_weight"].astype(float)
    dm = train["dm_sample_weight"].astype(float)
    y_attack = train["is_injected_attack"].astype(int)

    rows = [
        {
            "dataset": dataset,
            "attack_ratio": attack_ratio,
            "diagnostic": "pearson_corr_c_u",
            "value": float(cohort.corr(dm, method="pearson")),
        },
        {
            "dataset": dataset,
            "attack_ratio": attack_ratio,
            "diagnostic": "spearman_corr_c_u",
            "value": float(cohort.corr(dm, method="spearman")),
        },
        {
            "dataset": dataset,
            "attack_ratio": attack_ratio,
            "diagnostic": "mean_c_clean_minus_attack",
            "value": float(cohort[y_attack == 0].mean() - cohort[y_attack == 1].mean()),
        },
        {
            "dataset": dataset,
            "attack_ratio": attack_ratio,
            "diagnostic": "mean_u_clean_minus_attack",
            "value": float(dm[y_attack == 0].mean() - dm[y_attack == 1].mean()),
        },
    ]

    formulas = {
        "Cohort only": "formula_cohort",
        "Loss-mixture only": "formula_dm",
        "Linear average": "formula_linear_alpha",
        "Geometric mean": "formula_geomean",
        "Min": "formula_min",
        "Product CARD": "formula_product",
    }
    for name, column in formulas.items():
        dist = label_distribution(train["sentiment"], train[column])
        rows.append(
            {
                "dataset": dataset,
                "attack_ratio": attack_ratio,
                "diagnostic": f"tvd_{name}",
                "value": total_variation_distance(clean_dist, dist),
            }
        )
    return rows


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    for dataset, path in DATASETS.items():
        for attack_ratio in [0.10, 0.20, 0.30, 0.40]:
            rows.extend(summarize_dataset(dataset, path, attack_ratio))
            print(f"finished dataset={dataset} attack_ratio={attack_ratio}")

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "reliability_interaction_diagnostics.csv", index=False)

    formulas_long = out[out["diagnostic"].str.startswith("tvd_")].copy()
    formulas_long["formula"] = formulas_long["diagnostic"].str.replace("tvd_", "", regex=False)
    formulas_long[["dataset", "attack_ratio", "formula", "value"]].to_csv(
        REPORTS / "formula_multi_ratio_eval.csv", index=False
    )
    formula_means = (
        formulas_long.groupby(["formula", "dataset"], as_index=False)["value"]
        .mean()
        .pivot(index="formula", columns="dataset", values="value")
        .reset_index()
    )
    formula_means["Overall mean"] = formula_means[["Uber", "Google Apps"]].mean(axis=1)
    formula_means.to_csv(REPORTS / "table_formula_multi_ratio_mean_tvd.csv", index=False)

    out_20 = out[out["attack_ratio"] == 0.20].copy()

    corr = (
        out_20[out_20["diagnostic"].isin(["pearson_corr_c_u", "spearman_corr_c_u"])]
        .pivot(index="dataset", columns="diagnostic", values="value")
        .reset_index()
    )
    corr.to_csv(REPORTS / "table_reliability_correlation.csv", index=False)

    formulas = out_20[out_20["diagnostic"].str.startswith("tvd_")].copy()
    formulas["formula"] = formulas["diagnostic"].str.replace("tvd_", "", regex=False)
    formulas = formulas.pivot(index="formula", columns="dataset", values="value").reset_index()
    order = ["Cohort only", "Loss-mixture only", "Linear average", "Geometric mean", "Min", "Product CARD"]
    formulas["order"] = formulas["formula"].map({name: i for i, name in enumerate(order)})
    formulas = formulas.sort_values("order").drop(columns="order")
    formulas.to_csv(REPORTS / "table_formula_cross_dataset_tvd.csv", index=False)

    print(corr.to_string(index=False))
    print(formulas.to_string(index=False))


if __name__ == "__main__":
    main()
