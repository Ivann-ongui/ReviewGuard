from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "tmp" / "train_only_protocol_refresh"
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"
sys.path.insert(0, str(SCRIPTS))

import experiment_lib as el


_canonical_cache: dict[tuple, tuple[pd.DataFrame, dict[str, float], dict]] = {}
_explicit_cache: dict[tuple, tuple[pd.DataFrame, dict[str, float], dict]] = {}


def _dataset_key(raw: pd.DataFrame) -> tuple:
    return (len(raw), tuple(raw.columns), str(raw.iloc[0].get("content", ""))[:80])


def _weight_key(weights: dict[str, float] | None) -> tuple | None:
    return None if weights is None else tuple(sorted((key, float(value)) for key, value in weights.items()))


def _copy_result(result):
    train, clean_dist, detection = result
    return train.copy(), dict(clean_dist), dict(detection)


def build_canonical_train_only(
    raw: pd.DataFrame,
    attack_ratio: float,
    only_attack_type: str | None = None,
    feature_weights: dict[str, float] | None = None,
    sigmoid_midpoint: float = 0.45,
    sigmoid_sharpness: float = 10.0,
    cohort_weight_floor: float = 0.05,
    cv_splits: int = 3,
    weight_floor: float = 0.05,
):
    key = (
        _dataset_key(raw),
        float(attack_ratio),
        only_attack_type,
        _weight_key(feature_weights),
        float(sigmoid_midpoint),
        float(sigmoid_sharpness),
        float(cohort_weight_floor),
        int(cv_splits),
        float(weight_floor),
    )
    if key in _canonical_cache:
        return _copy_result(_canonical_cache[key])
    real = el.prepare_real_reviews(raw)
    split = el.split_real_reviews(real, seed=42)
    attack_rows = int(round(len(split.train_real) * attack_ratio))
    attack_seed = 4200 + int(round(attack_ratio * 1000))
    attack = el.generate_attacks(real, attack_rows, seed=attack_seed, only_attack_type=only_attack_type)
    analysis_stream = pd.concat([split.train_real, attack], ignore_index=True)
    train, detection = el.score_cohorts(
        analysis_stream,
        seed=42,
        feature_weights=feature_weights,
        sigmoid_midpoint=sigmoid_midpoint,
        sigmoid_sharpness=sigmoid_sharpness,
        sigmoid_floor=cohort_weight_floor,
    )
    train = el.add_loss_mixture_scores(train, cv_splits=cv_splits, seed=42, weight_floor=weight_floor)
    train = el.add_weight_formula_columns(train, floor=el.FINAL_WEIGHT_FLOOR)
    result = (train, el.label_distribution(split.train_real["sentiment"]), detection)
    _canonical_cache[key] = _copy_result(result)
    return _copy_result(result)


def build_explicit_train_only(
    raw: pd.DataFrame,
    attack_rows: int,
    seed: int = 42,
    only_attack_type: str | None = None,
    feature_weights: dict[str, float] | None = None,
    sigmoid_midpoint: float = 0.45,
    sigmoid_sharpness: float = 10.0,
    cv_splits: int = 3,
    weight_floor: float = 0.05,
):
    key = (
        _dataset_key(raw),
        int(attack_rows),
        int(seed),
        only_attack_type,
        _weight_key(feature_weights),
        float(sigmoid_midpoint),
        float(sigmoid_sharpness),
        int(cv_splits),
        float(weight_floor),
    )
    if key in _explicit_cache:
        return _copy_result(_explicit_cache[key])
    real = el.prepare_real_reviews(raw)
    split = el.split_real_reviews(real, seed=42)
    attack = el.generate_attacks(real, attack_rows, seed=seed, only_attack_type=only_attack_type)
    analysis_stream = pd.concat([split.train_real, attack], ignore_index=True)
    train, detection = el.score_cohorts(
        analysis_stream,
        seed=seed,
        feature_weights=feature_weights,
        sigmoid_midpoint=sigmoid_midpoint,
        sigmoid_sharpness=sigmoid_sharpness,
    )
    train = el.add_loss_mixture_scores(train, cv_splits=cv_splits, seed=seed, weight_floor=weight_floor)
    train = el.add_weight_formula_columns(train, floor=max(0.01, weight_floor * 0.6))
    result = (train, el.label_distribution(split.train_real["sentiment"]), detection)
    _explicit_cache[key] = _copy_result(result)
    return _copy_result(result)


def load_script(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_patched(filename: str, module_name: str, canonical: bool = False, explicit: bool = False):
    module = load_script(module_name, filename)
    if hasattr(module, "REPORTS"):
        module.REPORTS = OUT
    if canonical and hasattr(module, "build_canonical_scored_train"):
        module.build_canonical_scored_train = build_canonical_train_only
    if explicit and hasattr(module, "build_scored_train"):
        module.build_scored_train = build_explicit_train_only
    print(f"\n=== running {filename} ===", flush=True)
    module.main()
    return module


def run_attack_types() -> None:
    raw = pd.read_csv(ROOT / "uber_reviews_without_reviewid.csv")
    real = el.prepare_real_reviews(raw)
    split = el.split_real_reviews(real, seed=42)
    clean_dist = el.label_distribution(split.train_real["sentiment"])
    attack_rows = int(round(len(split.train_real) * 0.20))
    rows = []
    for attack_type in el.ATTACK_PROPORTIONS:
        train, _, detection = build_explicit_train_only(
            raw,
            attack_rows,
            seed=9000 + attack_rows,
            only_attack_type=attack_type,
        )
        for row in el.method_distributions(train, clean_dist):
            row.update(
                {
                    "attack_type": attack_type,
                    "attack_rows": attack_rows,
                    **detection,
                    "mean_weight_clean": float(
                        train.loc[train["is_injected_attack"] == 0, "cohort_dm_sample_weight"].mean()
                    ),
                    "mean_weight_attack": float(
                        train.loc[train["is_injected_attack"] == 1, "cohort_dm_sample_weight"].mean()
                    ),
                }
            )
            rows.append(row)
        print(f"finished attack type={attack_type}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ablation_results.csv", index=False)


def run_review_examples() -> None:
    raw = pd.read_csv(ROOT / "uber_reviews_without_reviewid.csv")
    train, _, _ = build_canonical_train_only(raw, 0.20)
    selected = train[train["row_id"].isin(["attack_279", "real_6402"])].copy()
    columns = [
        "row_id",
        "attack_type",
        "is_injected_attack",
        "content",
        "at",
        "reviewCreatedVersion",
        "sample_weight",
        "dm_sample_weight",
        "cohort_dm_sample_weight",
    ]
    selected[columns].to_csv(OUT / "review_level_target_examples.csv", index=False)


def run_e5_corrected() -> None:
    embedding_module = load_script("embedding_corrected", "34_bert_card_eval.py")
    datasets = {
        "uber": ROOT / "uber_reviews_without_reviewid.csv",
        "google_play_app_reviews": PROCESSED / "google_play_app_reviews_standardized.csv",
    }
    rows = []
    model_slug = embedding_module.MODEL_NAME.replace("/", "__").replace("-", "_")
    for dataset_name, path in datasets.items():
        raw = pd.read_csv(path)
        real = el.prepare_real_reviews(raw)
        split = el.split_real_reviews(real, seed=42)
        attack_rows = int(round(len(split.train_real) * 0.20))
        attack = el.generate_attacks(real, attack_rows, seed=4400)
        full_stream = pd.concat([real, attack], ignore_index=True)
        cache = PROCESSED / f"embedding_card_{model_slug}_{dataset_name}_canonical_20pct.npy"
        full_embeddings = np.load(cache)
        if len(full_embeddings) != len(full_stream):
            raise RuntimeError(f"stale embedding cache {cache}: {len(full_embeddings)} != {len(full_stream)}")
        keep = full_stream["row_id"].isin(set(split.train_real["row_id"])) | (full_stream["is_injected_attack"] == 1)
        analysis_stream = full_stream.loc[keep].reset_index(drop=True)
        embeddings = full_embeddings[keep.to_numpy()]
        scored, detection = embedding_module.score_bert_cohorts(analysis_stream, embeddings, seed=42)
        tfidf_train, clean_dist, _ = build_canonical_train_only(raw, 0.20)
        dm_by_id = tfidf_train.set_index("row_id")["dm_sample_weight"]
        scored["dm_sample_weight"] = scored["row_id"].map(dm_by_id)
        if scored["dm_sample_weight"].isna().any():
            raise RuntimeError("E5 and TF-IDF corrected streams differ")
        scored["bert_card_weight"] = (
            scored["sample_weight"].astype(float) * scored["dm_sample_weight"].astype(float)
        ).clip(el.FINAL_WEIGHT_FLOOR, el.FINAL_WEIGHT_CEILING)
        scored["bert_card_t_weight"] = el.calibrate_contribution_weights(scored, "bert_card_weight")
        for method, column in [
            ("No defense", None),
            ("Loss-mixture only", "dm_sample_weight"),
            ("Embedding-CARD", "bert_card_weight"),
            ("Embedding-CARD-T", "bert_card_t_weight"),
        ]:
            dist = el.label_distribution(scored["sentiment"], scored[column] if column else None)
            rows.append(
                {
                    "dataset": dataset_name,
                    "embedding_model": embedding_module.MODEL_NAME,
                    "method": method,
                    "attack_ratio": 0.20,
                    "tvd_to_clean": el.total_variation_distance(clean_dist, dist),
                    "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                    **detection,
                }
            )
        print(f"finished corrected E5 dataset={dataset_name}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "bert_card_eval.csv", index=False)


def build_figures() -> None:
    intro = load_script("intro_corrected", "46_plot_intro_problem_example.py")
    intro.RESULTS = OUT / "attack_sweep_results.csv"
    intro.OUTPUT = OUT / "intro_problem_baseline_example.png"
    intro.main()

    tables = load_script("embedding_tables_corrected", "35_build_cross_dataset_embedding_tables.py")
    tables.REPORTS = OUT
    tables.main()

    cross_plot = load_script("cross_plot_corrected", "47_plot_cross_dataset_representation.py")
    cross_plot.INPUT = OUT / "table_representation_comparison_20pct_tvd.csv"
    cross_plot.OUTPUT = OUT / "paper_cross_dataset_representation_tvd.png"
    cross_plot.main()

    parameter_plot = load_script("parameter_plot_corrected", "48_plot_parameter_check.py")
    parameter_plot.INPUT = OUT / "hyperparameter_sensitivity.csv"
    parameter_plot.OUTPUT = OUT / "paper_parameter_sensitivity_tvd.png"
    parameter_plot.main()


def summarize() -> None:
    summary: dict[str, object] = {}
    sweep = pd.read_csv(OUT / "attack_sweep_results.csv")
    trusted = pd.read_csv(OUT / "trusted_calibrated_card_eval.csv")
    method_names = {
        "No defense": "polluted_unweighted",
        "Group only": "cohort_weighted",
        "Loss only": "dm_weighted",
        "ReviewGuard": "cohort_dm_weighted",
    }
    attack_table = {}
    for label, method in method_names.items():
        values = sweep[sweep["method"] == method].sort_values("attack_ratio")["tvd_to_clean"].tolist()
        attack_table[label] = values
    attack_table["ReviewGuard calibrated"] = (
        trusted[trusted["method"] == "CARD-T"].sort_values("attack_ratio")["tvd_to_clean"].tolist()
    )
    summary["attack_sweep"] = attack_table
    summary["attack_sweep_means"] = {key: float(np.mean(values)) for key, values in attack_table.items()}

    seed = pd.read_csv(OUT / "seed_stability.csv")
    summary["seed_stability"] = {
        "mean": float(seed["tvd_to_clean"].mean()),
        "sample_std": float(seed["tvd_to_clean"].std(ddof=1)),
        "min": float(seed["tvd_to_clean"].min()),
        "max": float(seed["tvd_to_clean"].max()),
    }

    strong = pd.read_csv(OUT / "strong_noisy_label_baselines.csv")
    summary["strong_baselines"] = {
        f"{row.dataset}:{row.method}": float(row.tvd_to_clean)
        for row in strong.itertuples()
        if row.method in {"No defense", "Loss-mixture only", "CARD", "CARD-T"}
    }

    representation = pd.read_csv(OUT / "representation_comparison_20pct.csv")
    summary["representation"] = {
        f"{row.dataset}:{row.representation}:{row.method}": float(row.tvd_to_clean)
        for row in representation.itertuples()
    }

    attack_types = pd.read_csv(OUT / "ablation_results.csv")
    summary["attack_types_reviewguard"] = {
        attack_type: float(
            attack_types[
                (attack_types["attack_type"] == attack_type)
                & (attack_types["method"] == "cohort_dm_weighted")
            ]["tvd_to_clean"].iloc[0]
        )
        for attack_type in el.ATTACK_PROPORTIONS
    }

    scale = pd.read_csv(OUT / "scale_stress_eval.csv")
    summary["scale_100k"] = {
        row.method: float(row.tvd_to_clean)
        for row in scale[scale["stream_rows"] == 100000].itertuples()
    }

    formula = pd.read_csv(OUT / "table_formula_multi_ratio_mean_tvd.csv")
    summary["formula_means"] = {
        str(row["formula"]): {
            "Uber": float(row["Uber"]),
            "Google Apps": float(row["Google Apps"]),
            "Overall": float(row["Overall mean"]),
        }
        for _, row in formula.iterrows()
    }

    corr = pd.read_csv(OUT / "table_reliability_correlation.csv")
    summary["correlations"] = corr.to_dict(orient="records")

    feature = pd.read_csv(OUT / "cohort_feature_ablation.csv")
    summary["feature_ablation"] = {
        row.variant: float(row.tvd_to_clean) for row in feature.itertuples()
    }

    simplex_summary = pd.read_csv(OUT / "cohort_weight_simplex_search_summary.csv")
    summary["simplex"] = simplex_summary.to_dict(orient="records")

    sensitivity = pd.read_csv(OUT / "hyperparameter_sensitivity.csv")
    summary["sensitivity"] = sensitivity[
        ["cohort_weight_floor", "cv_splits", "midpoint", "sharpness", "tvd_to_clean"]
    ].to_dict(orient="records")

    examples = pd.read_csv(OUT / "review_level_target_examples.csv")
    summary["review_examples"] = examples.to_dict(orient="records")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = []
    for name, path in {
        "Uber": ROOT / "uber_reviews_without_reviewid.csv",
        "Google Apps": PROCESSED / "google_play_app_reviews_standardized.csv",
    }.items():
        raw = pd.read_csv(path)
        train, _, _ = build_canonical_train_only(raw, 0.20)
        metadata.append(
            {
                "dataset": name,
                "raw_rows": len(raw),
                "analysis_rows": len(train),
                "clean_train_rows": int((train["is_injected_attack"] == 0).sum()),
                "attack_rows": int(train["is_injected_attack"].sum()),
            }
        )
    pd.DataFrame(metadata).to_csv(OUT / "protocol_metadata.csv", index=False)
    print(pd.DataFrame(metadata).to_string(index=False), flush=True)

    run_patched("07_attack_sweep.py", "attack_sweep_corrected", canonical=True)
    run_patched("31_trusted_calibrated_card_eval.py", "trusted_corrected", canonical=True)
    run_patched("20_seed_stability.py", "seed_corrected", explicit=True)
    run_review_examples()

    run_patched("42_strong_noisy_label_baselines.py", "strong_corrected", canonical=True)
    run_patched("33_cross_dataset_card_eval.py", "cross_corrected", canonical=True)
    run_e5_corrected()
    run_attack_types()
    run_patched("39_scale_stress_eval.py", "scale_corrected", explicit=True)

    run_patched("40_reliability_interaction_diagnostics.py", "interaction_corrected", canonical=True)
    run_patched("45_cohort_weight_simplex_search.py", "simplex_corrected", canonical=True)
    run_patched("19_cohort_feature_ablation.py", "feature_corrected", canonical=True)
    run_patched("18_hyperparameter_sensitivity.py", "sensitivity_corrected", canonical=True)

    build_figures()
    summarize()


if __name__ == "__main__":
    main()
