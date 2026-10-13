from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoModel, AutoTokenizer

from experiment_lib import (
    DEFAULT_COHORT_FEATURE_WEIGHTS,
    FINAL_WEIGHT_CEILING,
    FINAL_WEIGHT_FLOOR,
    add_loss_mixture_scores,
    calibrate_contribution_weights,
    generate_attacks,
    label_distribution,
    minmax,
    prepare_real_reviews,
    sigmoid_weight,
    split_real_reviews,
    total_variation_distance,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"
MODEL_NAME = os.environ.get("CARD_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
DATASETS = {
    "uber": ROOT / "uber_reviews_without_reviewid.csv",
    "google_play_app_reviews": PROCESSED / "google_play_app_reviews_standardized.csv",
}


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts: list[str], cache_path: Path, batch_size: int = 64, max_length: int = 96) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    if "e5" in MODEL_NAME.lower():
        texts = [f"passage: {text}" for text in texts]
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype("float32"))
            if start and start % (batch_size * 20) == 0:
                print(f"encoded {start}/{len(texts)}")
    arr = np.vstack(vectors)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


def score_bert_cohorts(df: pd.DataFrame, embeddings: np.ndarray, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["content"] = df["content"].fillna("").astype(str)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["hour"] = df["at"].dt.floor("h")
    df["version_norm"] = df["reviewCreatedVersion"].fillna("UNKNOWN").astype(str)
    n_clusters = min(90, max(12, len(df) // 180))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=1024, n_init="auto")
    df["cohort_id"] = kmeans.fit_predict(embeddings)

    hourly = df.groupby("hour").size()
    hourly_z = (hourly - hourly.mean()) / (hourly.std(ddof=0) + 1e-9)
    df["burst_score"] = minmax(df["hour"].map(hourly_z).fillna(0).clip(lower=0))

    normalized_text = df["content"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    dup_counts = normalized_text.map(normalized_text.value_counts())
    df["duplicate_score"] = minmax(np.log1p(dup_counts))
    cluster_size = df.groupby("cohort_id")["row_id"].transform("count")
    cluster_low_rating = df.groupby("cohort_id")["score"].transform(lambda s: (s <= 2).mean())
    cluster_version_share = df.groupby("cohort_id")["version_norm"].transform(lambda s: s.value_counts(normalize=True).iloc[0])

    centers = kmeans.cluster_centers_
    centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9, None)
    centroid_sim = np.sum(embeddings * centers[df["cohort_id"].to_numpy()], axis=1)
    df["cluster_size_score"] = minmax(np.log1p(cluster_size))
    df["low_rating_cluster_score"] = cluster_low_rating.astype(float)
    df["version_concentration_score"] = cluster_version_share.astype(float)
    df["semantic_compactness_score"] = minmax(pd.Series(centroid_sim, index=df.index))
    df["contamination_score"] = sum(
        DEFAULT_COHORT_FEATURE_WEIGHTS[feature] * df[feature].astype(float)
        for feature in DEFAULT_COHORT_FEATURE_WEIGHTS
    ).clip(0, 1)
    df["sample_weight"] = sigmoid_weight(df["contamination_score"], midpoint=0.45, sharpness=10.0)

    y = df["is_injected_attack"].astype(int)
    threshold = float(df["contamination_score"].quantile(0.85))
    pred = (df["contamination_score"] >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    metrics = {
        "attack_detection_roc_auc": float(roc_auc_score(y, df["contamination_score"])),
        "attack_detection_average_precision": float(average_precision_score(y, df["contamination_score"])),
        "top_15_percent_precision": float(precision),
        "top_15_percent_recall": float(recall),
        "top_15_percent_f1": float(f1),
    }
    return df, metrics


def run_dataset(dataset_name: str, path: Path) -> list[dict]:
    raw = pd.read_csv(path)
    real = prepare_real_reviews(raw)
    split = split_real_reviews(real, seed=42)
    attack_rows = int(round(len(split.train_real) * 0.20))
    attack = generate_attacks(real, attack_rows, seed=4400)
    combined = pd.concat([real, attack], ignore_index=True)
    model_slug = MODEL_NAME.replace("/", "__").replace("-", "_")
    cache = PROCESSED / f"embedding_card_{model_slug}_{dataset_name}_canonical_20pct.npy"
    embeddings = encode_texts(combined["content"].fillna("").astype(str).tolist(), cache)
    scored, detection = score_bert_cohorts(combined, embeddings)
    train = pd.concat(
        [
            scored[(scored["is_injected_attack"] == 0) & (scored["row_id"].isin(split.train_real["row_id"]))],
            scored[scored["is_injected_attack"] == 1],
        ],
        ignore_index=True,
    )
    train = add_loss_mixture_scores(train, cv_splits=3, seed=42)
    train["bert_card_weight"] = (
        train["sample_weight"].astype(float) * train["dm_sample_weight"].astype(float)
    ).clip(FINAL_WEIGHT_FLOOR, FINAL_WEIGHT_CEILING)
    train["bert_card_t_weight"] = calibrate_contribution_weights(train, "bert_card_weight")
    clean_dist = label_distribution(split.train_real["sentiment"])
    rows = []
    for method, col in [
        ("No defense", None),
        ("Loss-mixture only", "dm_sample_weight"),
        ("Embedding-CARD", "bert_card_weight"),
        ("Embedding-CARD-T", "bert_card_t_weight"),
    ]:
        dist = label_distribution(train["sentiment"], train[col] if col else None)
        rows.append(
            {
                "dataset": dataset_name,
                "embedding_model": MODEL_NAME,
                "method": method,
                "attack_ratio": 0.20,
                "tvd_to_clean": total_variation_distance(clean_dist, dist),
                "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                **detection,
            }
        )
    return rows


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rows = []
    for dataset_name, path in DATASETS.items():
        rows.extend(run_dataset(dataset_name, path))
        print(f"finished Embedding-CARD dataset={dataset_name} model={MODEL_NAME}")
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "bert_card_eval.csv", index=False)
    pivot = out.pivot_table(index=["dataset", "method"], values="tvd_to_clean")
    pivot.to_csv(REPORTS / "table_bert_card_tvd.csv")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
