from pathlib import Path

import numpy as np
import pandas as pd

from experiment_lib import (
    DEFAULT_COHORT_FEATURE_WEIGHTS,
    build_canonical_scored_train,
    label_distribution,
    sigmoid_weight,
    total_variation_distance,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASETS = {
    "Uber": ROOT / "uber_reviews_without_reviewid.csv",
    "Google Apps": ROOT / "data" / "processed" / "google_play_app_reviews_standardized.csv",
}
FEATURES = list(DEFAULT_COHORT_FEATURE_WEIGHTS)


def normalized(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def card_tvd(train: pd.DataFrame, clean_dist: dict[str, float], beta: np.ndarray) -> float:
    x = train[FEATURES].astype(float).to_numpy()
    risk = pd.Series(np.clip(x @ beta, 0.0, 1.0), index=train.index)
    cohort_weight = sigmoid_weight(risk, midpoint=0.45, sharpness=10.0)
    weights = (cohort_weight.astype(float) * train["dm_sample_weight"].astype(float)).clip(0.03, 1.0)
    dist = label_distribution(train["sentiment"], weights)
    return float(total_variation_distance(clean_dist, dist))


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(45000)
    rows = []
    summary_rows = []

    named_candidates = {
        "default_prior": normalized(np.array([DEFAULT_COHORT_FEATURE_WEIGHTS[f] for f in FEATURES], dtype=float)),
        "uniform": np.ones(len(FEATURES)) / len(FEATURES),
    }
    for feature_idx, feature in enumerate(FEATURES):
        one_hot = np.zeros(len(FEATURES))
        one_hot[feature_idx] = 1.0
        named_candidates[f"only_{feature}"] = one_hot

    random_candidates = []
    for concentration in [0.2, 0.5, 1.0, 2.0, 5.0]:
        random_candidates.extend(rng.dirichlet(np.ones(len(FEATURES)) * concentration, size=400))

    for dataset, path in DATASETS.items():
        raw = pd.read_csv(path)
        train, clean_dist, _ = build_canonical_scored_train(raw, 0.20)

        dataset_rows = []
        for name, beta in named_candidates.items():
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "candidate": name,
                    "tvd": card_tvd(train, clean_dist, beta),
                    **dict(zip(FEATURES, beta)),
                }
            )
        for idx, beta in enumerate(random_candidates):
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "candidate": f"dirichlet_{idx:04d}",
                    "tvd": card_tvd(train, clean_dist, beta),
                    **dict(zip(FEATURES, beta)),
                }
            )

        df = pd.DataFrame(dataset_rows).sort_values("tvd").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        rows.append(df)

        default_tvd = float(df.loc[df["candidate"] == "default_prior", "tvd"].iloc[0])
        best = df.iloc[0]
        summary_rows.append(
            {
                "dataset": dataset,
                "num_candidates": len(df),
                "default_tvd": default_tvd,
                "default_rank": int(df.loc[df["candidate"] == "default_prior", "rank"].iloc[0]),
                "default_percentile": float((df["tvd"] >= default_tvd).mean()),
                "best_candidate": best["candidate"],
                "best_tvd": float(best["tvd"]),
                "default_minus_best": default_tvd - float(best["tvd"]),
            }
        )
        print(f"finished {dataset}")

    all_rows = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    all_rows.to_csv(REPORTS / "cohort_weight_simplex_search.csv", index=False)
    summary.to_csv(REPORTS / "cohort_weight_simplex_search_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(all_rows.groupby("dataset").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
