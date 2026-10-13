from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiment_lib import build_scored_train, label_distribution, total_variation_distance


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCES = [
    ROOT / "uber_reviews_without_reviewid.csv",
    ROOT / "data" / "processed" / "google_play_app_reviews_standardized.csv",
]


def make_stream(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [pd.read_csv(path) for path in SOURCES]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["content"].fillna("").astype(str).str.len() > 0].copy()
    sampled = df.sample(n=n_rows, replace=True, random_state=seed).reset_index(drop=True)
    start = pd.Timestamp("2024-01-01")
    offsets = np.sort(rng.integers(0, 90 * 24 * 3600, size=n_rows))
    sampled["at"] = [(start + pd.Timedelta(seconds=int(x))).strftime("%Y-%m-%d %H:%M:%S") for x in offsets]
    sampled["row_id"] = [f"stress_{n_rows}_{i}" for i in range(n_rows)]
    return sampled


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rows = []
    for n_rows in [25_000, 50_000, 100_000]:
        raw = make_stream(n_rows, seed=39000 + n_rows)
        attack_rows = int(round(n_rows * 0.20))
        start = perf_counter()
        train, clean_dist, detection = build_scored_train(
            raw,
            attack_rows,
            seed=39000 + n_rows,
            cv_splits=2,
        )
        elapsed = perf_counter() - start
        methods = {
            "No defense": None,
            "Cohort only": "sample_weight",
            "DivideMix only": "dm_sample_weight",
            "CARD": "cohort_dm_sample_weight",
        }
        for method, col in methods.items():
            dist = label_distribution(train["sentiment"], train[col] if col else None)
            rows.append(
                {
                    "stream_rows": n_rows,
                    "attack_rows": attack_rows,
                    "method": method,
                    "tvd_to_clean": total_variation_distance(clean_dist, dist),
                    "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                    "runtime_seconds": elapsed,
                    **detection,
                }
            )
        print(f"finished n_rows={n_rows} elapsed={elapsed:.1f}s")
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "scale_stress_eval.csv", index=False)
    table = out.pivot_table(index=["stream_rows", "method"], values=["tvd_to_clean", "runtime_seconds"])
    table.to_csv(REPORTS / "table_scale_stress_tvd.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
