from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    cross = pd.read_csv(REPORTS / "cross_dataset_card_eval.csv")
    embed = pd.read_csv(REPORTS / "bert_card_eval.csv")

    cross_table = cross.pivot_table(index=["dataset", "method"], columns="attack_setting", values="tvd_to_clean")
    cross_table.to_csv(REPORTS / "table_cross_dataset_card_methods_tvd.csv")

    embed_table = embed.pivot_table(index=["dataset", "method"], values="tvd_to_clean")
    embed_table.to_csv(REPORTS / "table_embedding_card_tvd.csv")

    merged_rows = []
    for _, row in cross.iterrows():
        if row["attack_setting"] == "20pct" and row["method"] in {"No defense", "CARD", "CARD-T"}:
            merged_rows.append(
                {
                    "dataset": row["dataset"],
                    "representation": "TF-IDF cohort",
                    "method": row["method"],
                    "tvd_to_clean": row["tvd_to_clean"],
                    "negative_abs_error": row["negative_abs_error"],
                }
            )
    for _, row in embed.iterrows():
        method_map = {
            "No defense": "No defense",
            "Embedding-CARD": "CARD",
            "Embedding-CARD-T": "CARD-T",
        }
        if row["method"] in method_map:
            merged_rows.append(
                {
                    "dataset": row["dataset"],
                    "representation": row["embedding_model"],
                    "method": method_map[row["method"]],
                    "tvd_to_clean": row["tvd_to_clean"],
                    "negative_abs_error": row["negative_abs_error"],
                }
            )
    merged = pd.DataFrame(merged_rows)
    merged.to_csv(REPORTS / "representation_comparison_20pct.csv", index=False)
    rep_table = merged.pivot_table(index=["dataset", "representation", "method"], values="tvd_to_clean")
    rep_table.to_csv(REPORTS / "table_representation_comparison_20pct_tvd.csv")
    print(rep_table.to_string())


if __name__ == "__main__":
    main()
