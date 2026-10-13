from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from experiment_lib import (
    build_canonical_scored_train,
    calibrate_contribution_weights,
    label_distribution,
    make_char_pipeline,
    make_word_pipeline,
    total_variation_distance,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASETS = {
    "Uber": ROOT / "uber_reviews_without_reviewid.csv",
    "Google Apps": ROOT / "data" / "processed" / "google_play_app_reviews_standardized.csv",
}


def minmax(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def weights_from_loss(loss: np.ndarray, floor: float = 0.03) -> np.ndarray:
    score = 1.0 - minmax(loss)
    return np.clip(floor + (1.0 - floor) * score, floor, 1.0)


def oof_probabilities(train: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = train["sentiment"].astype(str).to_numpy()
    classes = np.array(sorted(pd.unique(labels)))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_idx = np.array([class_to_idx[label] for label in labels])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    word = cross_val_predict(make_word_pipeline(seed), train["content"], labels, cv=cv, method="predict_proba")
    char = cross_val_predict(make_char_pipeline(seed + 1), train["content"], labels, cv=cv, method="predict_proba")
    return word, char, classes, y_idx


def add_robust_baseline_weights(train: pd.DataFrame, attack_ratio: float, seed: int) -> pd.DataFrame:
    train = train.copy()
    word, char, classes, y_idx = oof_probabilities(train, seed)
    probs = 0.5 * word + 0.5 * char
    p_y = np.clip(probs[np.arange(len(train)), y_idx], 1e-6, 1.0)
    ce = -np.log(p_y)

    # Generalized Cross Entropy: L_q=(1-p_y^q)/q. Lower loss means more reliable.
    q = 0.7
    gce = (1.0 - np.power(p_y, q)) / q
    train["gce_weight"] = weights_from_loss(gce)

    # Symmetric Cross Entropy combines CE with reverse CE using clipped labels.
    clipped_label = np.full_like(probs, 1e-4)
    clipped_label[np.arange(len(train)), y_idx] = 1.0
    rce = -np.sum(probs * np.log(clipped_label), axis=1)
    sce = ce + 0.1 * rce
    train["sce_weight"] = weights_from_loss(sce)

    # Bootstrapping reliability uses model agreement with the observed label.
    strongest_other = np.max(np.where(np.eye(len(classes))[y_idx].astype(bool), -1.0, probs), axis=1)
    bootstrap_margin = p_y - strongest_other
    train["bootstrap_weight"] = np.clip(0.03 + 0.97 * ((bootstrap_margin + 1.0) / 2.0), 0.03, 1.0)

    # Co-teaching small-loss keeps examples ranked as small-loss by two views.
    word_loss = -np.log(np.clip(word[np.arange(len(train)), y_idx], 1e-6, 1.0))
    char_loss = -np.log(np.clip(char[np.arange(len(train)), y_idx], 1e-6, 1.0))
    remember_rate = max(0.50, min(0.95, 1.0 - attack_ratio))
    cutoff = int(round(len(train) * remember_rate))
    word_keep = np.zeros(len(train), dtype=float)
    char_keep = np.zeros(len(train), dtype=float)
    word_keep[np.argsort(word_loss)[:cutoff]] = 1.0
    char_keep[np.argsort(char_loss)[:cutoff]] = 1.0
    train["coteaching_weight"] = np.clip(0.03 + 0.97 * np.minimum(word_keep, char_keep), 0.03, 1.0)

    # ELR-style confidence rewards high label confidence and low predictive entropy.
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-6, 1.0)), axis=1) / np.log(len(classes))
    train["elr_style_weight"] = np.clip(0.03 + 0.97 * p_y * (1.0 - entropy), 0.03, 1.0)
    train["ce_log_loss"] = log_loss(train["sentiment"].astype(str), probs, labels=classes)
    return train


def evaluate_weights(dataset: str, train: pd.DataFrame, clean_dist: dict[str, float]) -> list[dict]:
    train = train.copy()
    train["card_t_weight"] = calibrate_contribution_weights(train, "cohort_dm_sample_weight")
    methods = [
        ("No defense", None),
        ("GCE", "gce_weight"),
        ("SCE", "sce_weight"),
        ("Bootstrapping", "bootstrap_weight"),
        ("Co-teaching", "coteaching_weight"),
        ("ELR-style", "elr_style_weight"),
        ("Loss-mixture only", "dm_sample_weight"),
        ("CARD", "cohort_dm_sample_weight"),
        ("CARD-T", "card_t_weight"),
    ]
    rows = []
    for method, column in methods:
        weights = None if column is None else train[column]
        dist = label_distribution(train["sentiment"], weights)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "tvd_to_clean": total_variation_distance(clean_dist, dist),
                "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                "mean_weight_clean": np.nan
                if column is None
                else float(train.loc[train["is_injected_attack"] == 0, column].mean()),
                "mean_weight_attack": np.nan
                if column is None
                else float(train.loc[train["is_injected_attack"] == 1, column].mean()),
            }
        )
    return rows


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rows = []
    attack_ratio = 0.20
    for dataset, path in DATASETS.items():
        raw = pd.read_csv(path)
        train, clean_dist, _ = build_canonical_scored_train(raw, attack_ratio)
        train = add_robust_baseline_weights(train, attack_ratio=attack_ratio, seed=42)
        rows.extend(evaluate_weights(dataset, train, clean_dist))
        print(f"finished strong baselines dataset={dataset}")

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "strong_noisy_label_baselines.csv", index=False)
    table = out.pivot_table(index="method", columns="dataset", values="tvd_to_clean")
    order = ["No defense", "GCE", "SCE", "Bootstrapping", "Co-teaching", "ELR-style", "Loss-mixture only", "CARD", "CARD-T"]
    table = table.reindex(order)
    table.to_csv(REPORTS / "table_strong_noisy_label_baselines_tvd.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
