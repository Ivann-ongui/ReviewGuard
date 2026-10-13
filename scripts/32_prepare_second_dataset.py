from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reviews.csv"
OUT = ROOT / "data" / "processed" / "google_play_app_reviews_standardized.csv"


def main() -> None:
    df = pd.read_csv(SOURCE)
    df["content"] = df["content"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df = df[df["content"].str.len() > 0].copy()
    df = df[df["score"].between(1, 5)].copy()
    df = df[df["at"].notna()].copy()

    df["score"] = df["score"].astype(int)
    if "appVersion" not in df.columns:
        df["appVersion"] = df.get("reviewCreatedVersion")
    df["appVersion"] = df["appVersion"].fillna(df["reviewCreatedVersion"])
    df["source_dataset"] = "Google Play App Reviews Dataset"
    df["synthetic_metadata"] = False

    columns = [
        "reviewId",
        "userName",
        "userImage",
        "content",
        "score",
        "thumbsUpCount",
        "reviewCreatedVersion",
        "at",
        "replyContent",
        "repliedAt",
        "appVersion",
        "appId",
        "source_dataset",
        "synthetic_metadata",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    df["at"] = df["at"].dt.strftime("%Y-%m-%d %H:%M:%S")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(f"rows={len(df)}")
    print("score counts")
    print(df["score"].value_counts().sort_index().to_string())
    print("top appId counts")
    print(df["appId"].value_counts().head(15).to_string())
    print(f"time_span={df['at'].min()} to {df['at'].max()}")


if __name__ == "__main__":
    main()
