from __future__ import annotations

import random
import re
from itertools import product
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


ATTACK_PROPORTIONS = {
    "template_review_bomb": 900 / 2550,
    "paraphrase_review_bomb": 650 / 2550,
    "slow_drift_attack": 550 / 2550,
    "mixed_rating_camouflage": 450 / 2550,
}

FINAL_WEIGHT_FLOOR = 0.03
FINAL_WEIGHT_CEILING = 1.0
VERIFIED_SAMPLE_RATIO = 0.05
VERIFIED_SAMPLE_SEED = 4300

EXTENDED_ATTACK_TYPES = [
    "template_review_bomb",
    "paraphrase_review_bomb",
    "slow_drift_attack",
    "mixed_rating_camouflage",
    "llm_style_camouflage",
    "partially_adaptive_mixed",
]

NEGATIVE_TEMPLATES = [
    "App keeps charging me for trips I never took. Support ignores every message.",
    "Uber is unusable after the old update. Driver cancels and I still get charged.",
    "Fake prices, hidden fees, and no refund. Worst ride app experience.",
    "The old version broke payments and support refuses to help.",
    "Booked a ride, got cancelled, charged anyway. This app is a scam.",
    "Terrible update. Payment failed twice and customer service did nothing.",
    "Prices jump after booking and the app crashes on the payment screen.",
    "Charged for a cancelled ride. I want my money back now.",
]

PARAPHRASE_FRAGMENTS = {
    "opening": [
        "I used to rely on Uber, but",
        "After the latest trip,",
        "For the past few rides,",
        "This app has become frustrating because",
        "I cannot trust the service now because",
    ],
    "issue": [
        "the fare changed after booking",
        "support would not resolve a wrong charge",
        "drivers cancelled and the app still billed me",
        "the payment screen failed repeatedly",
        "the app froze right when I needed a ride",
        "the old version keeps breaking checkout",
    ],
    "closing": [
        "I am switching to another app.",
        "This needs a real fix.",
        "Please refund affected users.",
        "It makes the service feel unreliable.",
        "I cannot recommend it anymore.",
    ],
}


# This attack deliberately varies observable fields without inspecting ReviewGuard
# scores or parameters.  It is a realistic partially adaptive campaign, not a
# white-box evasion construction.
PARTIALLY_ADAPTIVE_TOPICS = {
    "cancellation": [
        "my driver cancelled after I had already been waiting",
        "the ride was cancelled after the app showed that the driver was close",
        "several drivers accepted and then cancelled the same trip",
        "the cancellation notice arrived after the pickup estimate had changed",
        "the trip disappeared after a long wait and I still had to rebook",
        "the app marked the driver as arriving and then cancelled the ride",
    ],
    "charge_refund": [
        "the final charge was higher than the estimate I accepted",
        "a cancelled trip still left a charge on my card",
        "the refund request kept moving between automated support replies",
        "the app charged a fee even though the ride never started",
        "the receipt showed an amount that did not match the trip",
        "support did not explain an unexpected payment adjustment",
    ],
    "crash_login": [
        "the app froze while I was trying to confirm the ride",
        "the login screen looped and would not let me request a car",
        "the payment screen crashed before the booking was completed",
        "the latest app session closed when I tried to change the destination",
        "the app kept returning to the sign-in page during checkout",
        "the booking screen stopped responding just before confirmation",
    ],
    "map_pickup": [
        "the pickup pin moved to a different street from my actual location",
        "the map sent the driver to the wrong side of the station",
        "the pickup point changed after I confirmed the ride",
        "the driver could not find me because the location pin was inaccurate",
        "the route preview placed the pickup marker several blocks away",
        "the map instructions made the driver circle a nearby intersection",
    ],
}

PARTIALLY_ADAPTIVE_OPENINGS = [
    "I usually rely on this service, but",
    "This week I had a frustrating experience because",
    "I tried to use the app for a normal trip and",
    "The ride itself was not the only issue;",
    "I expected a routine booking, yet",
    "After using the app again today,",
    "I do not normally leave reviews, but",
    "The recent experience was disappointing because",
]

PARTIALLY_ADAPTIVE_CONTEXTS = [
    "during an early morning trip",
    "when I was heading home from work",
    "while I was trying to get to an appointment",
    "on a short trip across town",
    "when I needed a ride after a late shift",
    "during a weekend booking",
    "while travelling with luggage",
    "when I was already running behind schedule",
]

PARTIALLY_ADAPTIVE_CLOSINGS = [
    "That should be fixed before I can trust the service again.",
    "It made a simple trip much harder than it needed to be.",
    "I hope the next update addresses this reliability problem.",
    "The support flow did not make the situation any clearer.",
    "This is why I am lowering my rating for now.",
    "A more reliable app experience would make a real difference.",
    "I would like to see a clear resolution instead of another generic reply.",
    "The service needs a more dependable way to handle this issue.",
]


@dataclass
class SplitData:
    real: pd.DataFrame
    train_real: pd.DataFrame
    test_real: pd.DataFrame


def sentiment_from_score(score: int) -> str:
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def prepare_real_reviews(df: pd.DataFrame) -> pd.DataFrame:
    real = df.copy()
    if "row_id" not in real.columns:
        real.insert(0, "row_id", [f"real_{i}" for i in range(len(real))])
    real["is_injected_attack"] = 0
    real["attack_type"] = "clean_real"
    real["sentiment"] = real["score"].map(sentiment_from_score)
    return real


def split_real_reviews(real: pd.DataFrame, seed: int = 42) -> SplitData:
    train_real, test_real = train_test_split(
        real,
        test_size=0.25,
        random_state=seed,
        stratify=real["sentiment"],
    )
    return SplitData(real=real, train_real=train_real.copy(), test_real=test_real.copy())


def random_time(np_rng: np.random.Generator, start: str, end: str) -> pd.Timestamp:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    seconds = int((end_ts - start_ts).total_seconds())
    return start_ts + pd.Timedelta(seconds=int(np_rng.integers(0, seconds)))


def make_row(
    base_columns: list[str],
    user_name: str,
    content: str,
    score: int,
    at: pd.Timestamp,
    version: str,
    thumbs_up: int,
) -> dict:
    row = {col: None for col in base_columns}
    row.update(
        {
            "userName": user_name,
            "userImage": np.nan,
            "content": content,
            "score": score,
            "thumbsUpCount": thumbs_up,
            "reviewCreatedVersion": np.nan if version == "UNKNOWN" else version,
            "at": at.strftime("%Y-%m-%d %H:%M:%S"),
            "replyContent": np.nan,
            "repliedAt": np.nan,
            "appVersion": np.nan if version == "UNKNOWN" else version,
        }
    )
    return row


def generate_attack_type(
    base_columns: list[str],
    attack_type: str,
    n_rows: int,
    start_idx: int,
    seed: int,
    version_choices: list[str] | None = None,
    version_weights: list[float] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    rows = []

    partially_adaptive_rows: list[dict] | None = None
    if attack_type == "partially_adaptive_mixed":
        # Allocate topics and ratings deterministically, then vary wording via
        # unique combinations.  The text generator never calls model scoring.
        topic_names = list(PARTIALLY_ADAPTIVE_TOPICS)
        topic_counts = allocate_counts(n_rows, {topic: 1.0 / len(topic_names) for topic in topic_names})
        combinations = []
        for topic, count in topic_counts.items():
            candidates = list(
                product(
                    PARTIALLY_ADAPTIVE_OPENINGS,
                    PARTIALLY_ADAPTIVE_TOPICS[topic],
                    PARTIALLY_ADAPTIVE_CONTEXTS,
                    PARTIALLY_ADAPTIVE_CLOSINGS,
                )
            )
            if count > len(candidates):
                raise ValueError("not enough unique partially_adaptive_mixed text combinations")
            combinations.extend((topic, *parts) for parts in rng.sample(candidates, count))
        rng.shuffle(combinations)

        rating_counts = allocate_counts(n_rows, {"1": 0.50, "2": 0.30, "3": 0.10, "4": 0.10})
        ratings = [int(rating) for rating, count in rating_counts.items() for _ in range(count)]
        rng.shuffle(ratings)
        if version_choices is None or not version_choices:
            version_choices = ["4.554.10001", "4.555.10003", "4.556.10005"]
            version_weights = [1.0, 1.0, 1.0]
        normalized_versions = version_weights or [1.0] * len(version_choices)
        partially_adaptive_rows = []
        for i, (_, opening, issue, context, closing) in enumerate(combinations):
            content = f"{opening} {issue} {context}. {closing}"
            partially_adaptive_rows.append(
                make_row(
                    base_columns,
                    f"InjectedUser_{start_idx + i}",
                    content,
                    ratings[i],
                    random_time(np_rng, "2024-12-06 00:00:00", "2024-12-12 00:00:00"),
                    rng.choices(version_choices, weights=normalized_versions, k=1)[0],
                    int(np_rng.integers(0, 18)),
                )
            )
        normalized = [re.sub(r"\s+", " ", row["content"].lower()).strip() for row in partially_adaptive_rows]
        if len(normalized) != len(set(normalized)):
            raise AssertionError("partially_adaptive_mixed must not contain duplicate normalized text")

    for i in range(n_rows):
        if partially_adaptive_rows is not None:
            row = partially_adaptive_rows[i]
        elif attack_type == "template_review_bomb":
            template = rng.choice(NEGATIVE_TEMPLATES)
            suffix = "" if rng.random() < 0.55 else rng.choice([" Please fix this.", " Refund me.", " Never using it again."])
            row = make_row(
                base_columns,
                f"InjectedUser_{start_idx + i}",
                f"{template}{suffix}",
                1,
                random_time(np_rng, "2024-12-10 08:00:00", "2024-12-11 20:00:00"),
                rng.choice(["4.532.10001", "4.533.10000", "4.532.10001", "UNKNOWN"]),
                int(np_rng.integers(0, 4)),
            )
        elif attack_type == "paraphrase_review_bomb":
            content = " ".join(
                [
                    rng.choice(PARAPHRASE_FRAGMENTS["opening"]),
                    rng.choice(PARAPHRASE_FRAGMENTS["issue"]),
                    rng.choice(PARAPHRASE_FRAGMENTS["closing"]),
                ]
            )
            row = make_row(
                base_columns,
                f"InjectedUser_{start_idx + i}",
                content,
                rng.choice([1, 1, 1, 2]),
                random_time(np_rng, "2024-12-07 00:00:00", "2024-12-12 23:59:00"),
                rng.choice(["4.548.10001", "4.549.10000", "4.550.10001", "4.551.10000"]),
                int(np_rng.integers(0, 9)),
            )
        elif attack_type == "slow_drift_attack":
            row = make_row(
                base_columns,
                f"InjectedUser_{start_idx + i}",
                rng.choice(
                    [
                        "Long pickup times and repeated cancellations made this ride stressful.",
                        "The fare estimate looked fine, then the final price was much higher.",
                        "Support gave a generic answer and did not solve the trip problem.",
                        "The app location pin was wrong and the driver went elsewhere.",
                        "Payment authorization failed even though my card works everywhere else.",
                    ]
                ),
                rng.choice([1, 2, 2, 3]),
                random_time(np_rng, "2024-11-28 00:00:00", "2024-12-17 23:59:00"),
                rng.choice(["4.552.10000", "4.553.10000", "4.554.10001", "4.555.10003", "UNKNOWN"]),
                int(np_rng.integers(0, 15)),
            )
        elif attack_type == "mixed_rating_camouflage":
            row = make_row(
                base_columns,
                f"InjectedUser_{start_idx + i}",
                rng.choice(
                    [
                        "Good drivers sometimes, but hidden fees and refund issues are unacceptable.",
                        "Useful app overall, yet the cancellation charge made me angry.",
                        "The ride was okay, but support ignored the payment problem.",
                        "Convenient when it works, however the new update made checkout unreliable.",
                        "Fast pickup, but the app overcharged me and support did not help.",
                    ]
                ),
                rng.choice([2, 3, 3, 4]),
                random_time(np_rng, "2024-12-13 06:00:00", "2024-12-16 23:00:00"),
                rng.choice(["4.554.10001", "4.555.10003", "4.556.10005", "UNKNOWN"]),
                int(np_rng.integers(2, 25)),
            )
        elif attack_type == "llm_style_camouflage":
            openings = [
                "I know every app has bad days, but",
                "This is not a one star rant;",
                "The driver was polite, yet",
                "I still use Uber sometimes, although",
                "The trip itself was fine, but",
                "I do not want to exaggerate, however",
            ]
            issues = [
                "the checkout flow quietly changed the final charge after I confirmed the ride",
                "the refund workflow kept sending me in circles without resolving the incorrect fee",
                "the pickup pin drifted across the map and support treated it like my mistake",
                "the app showed one fare before the ride and a very different fare afterwards",
                "the cancellation policy was unclear and I was charged for a ride that never happened",
                "the payment screen failed twice, then the app recorded the trip as completed",
            ]
            closings = [
                "so the rating should reflect that reliability problem.",
                "which makes the service hard to trust for work trips.",
                "and a generic support reply does not fix the actual issue.",
                "even though the rest of the experience was not terrible.",
                "so I am leaving a lower rating until this is fixed.",
                "and this has happened enough times to be a pattern.",
            ]
            row = make_row(
                base_columns,
                f"InjectedUser_{start_idx + i}",
                " ".join([rng.choice(openings), rng.choice(issues), rng.choice(closings)]),
                rng.choice([1, 2, 2, 3, 3, 4]),
                random_time(np_rng, "2024-11-30 00:00:00", "2024-12-18 12:00:00"),
                rng.choice(["4.552.10000", "4.553.10000", "4.554.10001", "4.555.10003", "4.556.10005", "UNKNOWN"]),
                int(np_rng.integers(0, 30)),
            )
        else:
            raise ValueError(f"unknown attack_type={attack_type}")
        rows.append(row)
    return rows


def allocate_counts(total_rows: int, mix: dict[str, float]) -> dict[str, int]:
    raw = {k: total_rows * v for k, v in mix.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total_rows - sum(counts.values())
    for key, _ in sorted(raw.items(), key=lambda item: item[1] - np.floor(item[1]), reverse=True)[:remainder]:
        counts[key] += 1
    return counts


def generate_attacks(
    base_df: pd.DataFrame,
    total_rows: int,
    seed: int = 42,
    attack_mix: dict[str, float] | None = None,
    only_attack_type: str | None = None,
) -> pd.DataFrame:
    base_columns = [c for c in base_df.columns if c not in {"row_id", "is_injected_attack", "attack_type", "sentiment"}]
    if only_attack_type:
        counts = {only_attack_type: total_rows}
    else:
        counts = allocate_counts(total_rows, attack_mix or ATTACK_PROPORTIONS)

    rows = []
    next_idx = 0
    row_attack_types = []
    observed_versions = base_df.get("reviewCreatedVersion", pd.Series(dtype="object")).dropna().astype(str)
    observed_versions = observed_versions[observed_versions.str.strip().ne("") & observed_versions.ne("UNKNOWN")]
    top_versions = observed_versions.value_counts().head(3)
    version_choices = top_versions.index.tolist()
    version_weights = top_versions.astype(float).tolist()
    for offset, (attack_type, n_rows) in enumerate(counts.items()):
        built = generate_attack_type(
            base_columns,
            attack_type,
            n_rows,
            next_idx,
            seed + 101 * (offset + 1),
            version_choices=version_choices,
            version_weights=version_weights,
        )
        rows.extend(built)
        row_attack_types.extend([attack_type] * len(built))
        next_idx += len(built)

    attack = pd.DataFrame(rows)
    attack.insert(0, "row_id", [f"attack_{i}" for i in range(len(attack))])
    attack["is_injected_attack"] = 1
    attack["attack_type"] = row_attack_types
    attack["sentiment"] = attack["score"].map(sentiment_from_score)
    return attack


def minmax(series: pd.Series) -> pd.Series:
    series = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    lo = series.min()
    hi = series.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def sigmoid_weight(
    score: pd.Series,
    midpoint: float = 0.45,
    sharpness: float = 10.0,
    floor: float = 0.05,
) -> pd.Series:
    weights = 1.0 / (1.0 + np.exp(sharpness * (score.astype(float) - midpoint)))
    return pd.Series(floor + (1.0 - floor) * weights, index=score.index).clip(floor, 1.0)


def stratified_verified_sample(clean: pd.DataFrame, ratio: float = VERIFIED_SAMPLE_RATIO, seed: int = VERIFIED_SAMPLE_SEED) -> pd.DataFrame:
    parts = []
    for _, group in clean.groupby("sentiment"):
        n = max(1, int(round(len(group) * ratio)))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed, replace=False))
    return pd.concat(parts, ignore_index=False)


def calibrate_contribution_weights(
    train: pd.DataFrame,
    base_col: str,
    ratio: float = VERIFIED_SAMPLE_RATIO,
    seed: int = VERIFIED_SAMPLE_SEED,
    epsilon: float = 1e-6,
) -> pd.Series:
    trusted = stratified_verified_sample(
        train[train["is_injected_attack"] == 0], ratio=ratio, seed=seed
    )
    target = label_distribution(trusted["sentiment"])
    current = label_distribution(train["sentiment"], train[base_col])
    factors = {
        label: target.get(label, 0.0) / (current.get(label, 0.0) + epsilon)
        for label in target
    }
    return (
        train[base_col].astype(float)
        * train["sentiment"].map(factors).fillna(1.0).astype(float)
    ).clip(FINAL_WEIGHT_FLOOR, FINAL_WEIGHT_CEILING)


def make_word_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    min_df=2,
                    max_df=0.9,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]
    )


def make_char_pipeline(seed: int = 43) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]
    )


DEFAULT_COHORT_FEATURE_WEIGHTS = {
    "burst_score": 0.25,
    "duplicate_score": 0.20,
    "low_rating_cluster_score": 0.20,
    "version_concentration_score": 0.15,
    "semantic_compactness_score": 0.10,
    "cluster_size_score": 0.10,
}


def score_cohorts(
    df: pd.DataFrame,
    seed: int = 42,
    feature_weights: dict[str, float] | None = None,
    sigmoid_midpoint: float = 0.45,
    sigmoid_sharpness: float = 10.0,
    sigmoid_floor: float = 0.05,
    n_clusters: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["content"] = df["content"].fillna("").astype(str)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["hour"] = df["at"].dt.floor("h")
    df["date"] = df["at"].dt.date.astype(str)
    df["version_norm"] = df["reviewCreatedVersion"].fillna("UNKNOWN").astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    x = vectorizer.fit_transform(df["content"])
    n_clusters = n_clusters or min(90, max(12, len(df) // 180))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=1024, n_init="auto")
    df["cohort_id"] = kmeans.fit_predict(x)

    hourly = df.groupby("hour").size()
    hourly_z = (hourly - hourly.mean()) / (hourly.std(ddof=0) + 1e-9)
    df["burst_score"] = minmax(df["hour"].map(hourly_z).fillna(0).clip(lower=0))

    normalized_text = df["content"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    dup_counts = normalized_text.map(normalized_text.value_counts())
    df["duplicate_score"] = minmax(np.log1p(dup_counts))

    cluster_size = df.groupby("cohort_id")["row_id"].transform("count")
    cluster_low_rating = df.groupby("cohort_id")["score"].transform(lambda s: (s <= 2).mean())
    cluster_version_share = df.groupby("cohort_id")["version_norm"].transform(lambda s: s.value_counts(normalize=True).iloc[0])

    centroid_sim = np.zeros(len(df))
    centers = kmeans.cluster_centers_
    for cluster_id in range(n_clusters):
        idx = np.where(df["cohort_id"].to_numpy() == cluster_id)[0]
        if len(idx) == 0:
            continue
        centroid_sim[idx] = cosine_similarity(x[idx], centers[cluster_id].reshape(1, -1)).ravel()

    df["cluster_size_score"] = minmax(np.log1p(cluster_size))
    df["low_rating_cluster_score"] = cluster_low_rating.astype(float)
    df["version_concentration_score"] = cluster_version_share.astype(float)
    df["semantic_compactness_score"] = minmax(pd.Series(centroid_sim, index=df.index))
    weights = feature_weights or DEFAULT_COHORT_FEATURE_WEIGHTS
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        raise ValueError("feature_weights must contain at least one positive weight")
    contamination = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    for feature, weight in weights.items():
        if weight <= 0:
            continue
        contamination += (weight / total) * df[feature].astype(float)
    df["contamination_score"] = contamination.clip(0, 1)
    df["sample_weight"] = sigmoid_weight(
        df["contamination_score"],
        midpoint=sigmoid_midpoint,
        sharpness=sigmoid_sharpness,
        floor=sigmoid_floor,
    )

    metrics = {}
    if df["is_injected_attack"].nunique() == 2:
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
            "mean_score_clean": float(df.loc[y == 0, "contamination_score"].mean()),
            "mean_score_attack": float(df.loc[y == 1, "contamination_score"].mean()),
        }
    return df, metrics


def add_loss_mixture_scores(
    train: pd.DataFrame,
    cv_splits: int = 3,
    seed: int = 42,
    weight_floor: float = 0.05,
) -> pd.DataFrame:
    train = train.copy()
    labels = train["sentiment"].astype(str).to_numpy()
    classes = np.array(sorted(pd.unique(labels)))
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y_idx = np.array([class_to_idx[label] for label in labels])
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    word_probs = cross_val_predict(make_word_pipeline(seed), train["content"], labels, cv=cv, method="predict_proba")
    char_probs = cross_val_predict(make_char_pipeline(seed + 1), train["content"], labels, cv=cv, method="predict_proba")
    probs = 0.5 * word_probs + 0.5 * char_probs
    self_conf = probs[np.arange(len(train)), y_idx]
    masked = probs.copy()
    masked[np.arange(len(train)), y_idx] = -1.0
    strongest_other = masked.max(axis=1)
    predicted_idx = probs.argmax(axis=1)
    disagreement = (predicted_idx != y_idx).astype(float)

    normalized_margin = self_conf - strongest_other
    issue_score = np.clip((1.0 - normalized_margin) / 2.0, 0.0, 1.0)
    issue_score = np.maximum(issue_score, 0.75 * disagreement)
    nll = -np.log(np.clip(self_conf, 1e-6, 1.0))
    nll_norm = (nll - nll.min()) / (nll.max() - nll.min() + 1e-12)

    gmm = GaussianMixture(n_components=2, random_state=seed, covariance_type="full")
    gmm.fit(nll_norm.reshape(-1, 1))
    clean_component = int(np.argmin(gmm.means_.ravel()))
    clean_prob = gmm.predict_proba(nll_norm.reshape(-1, 1))[:, clean_component]

    train["cl_issue_score"] = issue_score
    train["cl_sample_weight"] = sigmoid_weight(train["cl_issue_score"], midpoint=0.52, sharpness=12.0)
    train["dm_clean_probability"] = clean_prob
    train["dm_sample_weight"] = pd.Series(weight_floor + (1.0 - weight_floor) * clean_prob, index=train.index).clip(weight_floor, 1.0)
    train["cohort_cl_sample_weight"] = (train["sample_weight"].astype(float) * train["cl_sample_weight"].astype(float)).clip(max(0.01, weight_floor * 0.6), 1.0)
    train["cohort_dm_sample_weight"] = (train["sample_weight"].astype(float) * train["dm_sample_weight"].astype(float)).clip(FINAL_WEIGHT_FLOOR, FINAL_WEIGHT_CEILING)
    train["cohort_cl_dm_sample_weight"] = (
        train["sample_weight"].astype(float)
        * (0.5 * train["cl_sample_weight"].astype(float) + 0.5 * train["dm_sample_weight"].astype(float))
    ).clip(max(0.01, weight_floor * 0.6), 1.0)
    train["cl_is_disagreement"] = disagreement
    train["cl_predicted_label"] = classes[predicted_idx]
    train["cl_cv_log_loss"] = log_loss(labels, probs, labels=classes)
    return train


def add_weight_formula_columns(train: pd.DataFrame, alpha: float = 0.5, floor: float = 0.03) -> pd.DataFrame:
    train = train.copy()
    cohort = train["sample_weight"].astype(float).clip(floor, 1.0)
    dm = train["dm_sample_weight"].astype(float).clip(floor, 1.0)
    train["formula_cohort"] = cohort
    train["formula_dm"] = dm
    train["formula_product"] = (cohort * dm).clip(floor, 1.0)
    train["formula_linear_alpha"] = (alpha * cohort + (1.0 - alpha) * dm).clip(floor, 1.0)
    train["formula_min"] = np.minimum(cohort, dm).clip(floor, 1.0)
    train["formula_geomean"] = np.sqrt(cohort * dm).clip(floor, 1.0)
    return train


def build_scored_train(
    raw: pd.DataFrame,
    attack_rows: int,
    seed: int = 42,
    only_attack_type: str | None = None,
    feature_weights: dict[str, float] | None = None,
    sigmoid_midpoint: float = 0.45,
    sigmoid_sharpness: float = 10.0,
    cv_splits: int = 3,
    weight_floor: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float], dict]:
    real = prepare_real_reviews(raw)
    split = split_real_reviews(real, seed=42)
    attack = generate_attacks(real, attack_rows, seed=seed, only_attack_type=only_attack_type)
    scored, detection = score_cohorts(
        pd.concat([split.train_real, attack], ignore_index=True),
        seed=seed,
        feature_weights=feature_weights,
        sigmoid_midpoint=sigmoid_midpoint,
        sigmoid_sharpness=sigmoid_sharpness,
    )
    train = pd.concat(
        [
            scored[(scored["is_injected_attack"] == 0) & (scored["row_id"].isin(split.train_real["row_id"]))],
            scored[scored["is_injected_attack"] == 1],
        ],
        ignore_index=True,
    )
    train = add_loss_mixture_scores(train, cv_splits=cv_splits, seed=seed, weight_floor=weight_floor)
    train = add_weight_formula_columns(train, floor=max(0.01, weight_floor * 0.6))
    return train, label_distribution(split.train_real["sentiment"]), detection


def build_canonical_scored_train(
    raw: pd.DataFrame,
    attack_ratio: float,
    only_attack_type: str | None = None,
    feature_weights: dict[str, float] | None = None,
    sigmoid_midpoint: float = 0.45,
    sigmoid_sharpness: float = 10.0,
    cohort_weight_floor: float = 0.05,
    cv_splits: int = 3,
    weight_floor: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float], dict]:
    real = prepare_real_reviews(raw)
    split = split_real_reviews(real, seed=42)
    attack_rows = int(round(len(split.train_real) * attack_ratio))
    attack_seed = 4200 + int(round(attack_ratio * 1000))
    attack = generate_attacks(real, attack_rows, seed=attack_seed, only_attack_type=only_attack_type)
    scored, detection = score_cohorts(
        pd.concat([split.train_real, attack], ignore_index=True),
        seed=42,
        feature_weights=feature_weights,
        sigmoid_midpoint=sigmoid_midpoint,
        sigmoid_sharpness=sigmoid_sharpness,
        sigmoid_floor=cohort_weight_floor,
    )
    train = pd.concat(
        [
            scored[(scored["is_injected_attack"] == 0) & (scored["row_id"].isin(split.train_real["row_id"]))],
            scored[scored["is_injected_attack"] == 1],
        ],
        ignore_index=True,
    )
    train = add_loss_mixture_scores(train, cv_splits=cv_splits, seed=42, weight_floor=weight_floor)
    train = add_weight_formula_columns(train, floor=FINAL_WEIGHT_FLOOR)
    return train, label_distribution(split.train_real["sentiment"]), detection


def label_distribution(labels: pd.Series, weights: pd.Series | None = None) -> dict[str, float]:
    labels = labels.astype(str)
    if weights is None:
        counts = labels.value_counts(normalize=True).sort_index()
        return {str(k): float(v) for k, v in counts.items()}
    tmp = pd.DataFrame({"label": labels, "weight": weights.astype(float)})
    weighted = tmp.groupby("label")["weight"].sum()
    weighted = weighted / weighted.sum()
    return {str(k): float(v) for k, v in weighted.sort_index().items()}


def total_variation_distance(reference: dict[str, float], candidate: dict[str, float]) -> float:
    keys = sorted(set(reference) | set(candidate))
    return float(0.5 * sum(abs(reference.get(k, 0.0) - candidate.get(k, 0.0)) for k in keys))


def method_distributions(train: pd.DataFrame, clean_dist: dict[str, float]) -> list[dict]:
    filter_threshold = float(train["contamination_score"].quantile(0.85))
    risk_filtered = train[train["contamination_score"] < filter_threshold]
    hybrid_threshold = float(train["cohort_cl_sample_weight"].quantile(0.15))
    hybrid_filtered = train[train["cohort_cl_sample_weight"] > hybrid_threshold]

    methods = {
        "polluted_unweighted": (train, None),
        "cohort_weighted": (train, "sample_weight"),
        "risk_filtered": (risk_filtered, None),
        "dm_weighted": (train, "dm_sample_weight"),
        "cl_weighted": (train, "cl_sample_weight"),
        "cohort_cl_weighted": (train, "cohort_cl_sample_weight"),
        "cohort_dm_weighted": (train, "cohort_dm_sample_weight"),
        "cohort_cl_dm_weighted": (train, "cohort_cl_dm_sample_weight"),
        "cohort_cl_filtered": (hybrid_filtered, "cohort_cl_sample_weight"),
    }
    if "policy_sample_weight" in train.columns:
        methods["adaptive_policy_weighted"] = (train, "policy_sample_weight")
    if "time_burst_weight" in train.columns:
        methods["time_burst_weighted"] = (train, "time_burst_weight")
    if "duplicate_weight" in train.columns:
        methods["duplicate_weighted"] = (train, "duplicate_weight")
    if "rating_cluster_weight" in train.columns:
        methods["rating_cluster_weighted"] = (train, "rating_cluster_weight")
    if "isolation_forest_weight" in train.columns:
        methods["isolation_forest_weighted"] = (train, "isolation_forest_weight")
    if "lof_weight" in train.columns:
        methods["lof_weighted"] = (train, "lof_weight")
    rows = []
    for method, (subset, weight_col) in methods.items():
        dist = label_distribution(subset["sentiment"], subset[weight_col] if weight_col else None)
        rows.append(
            {
                "method": method,
                "negative_rate": dist.get("negative", 0.0),
                "neutral_rate": dist.get("neutral", 0.0),
                "positive_rate": dist.get("positive", 0.0),
                "tvd_to_clean": total_variation_distance(clean_dist, dist),
                "negative_abs_error": abs(dist.get("negative", 0.0) - clean_dist.get("negative", 0.0)),
                "rows_used": int(len(subset)),
            }
        )
    return rows


def add_adaptive_policy_weights(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    duplicate_high = train["duplicate_score"] >= train["duplicate_score"].quantile(0.85)
    burst_high = train["burst_score"] >= train["burst_score"].quantile(0.85)
    cl_high = train["cl_issue_score"] >= train["cl_issue_score"].quantile(0.80)
    mixed_like = (train["low_rating_cluster_score"] < 0.70) & (train["version_concentration_score"] < 0.75)

    policy = pd.Series("cohort_dm", index=train.index, dtype="object")
    policy.loc[duplicate_high & burst_high] = "hard_filter"
    policy.loc[cl_high & ~duplicate_high] = "cohort_cl"
    policy.loc[mixed_like & ~cl_high] = "cohort_dm"

    weights = train["cohort_dm_sample_weight"].astype(float).copy()
    weights.loc[policy == "hard_filter"] = np.minimum(weights.loc[policy == "hard_filter"], train.loc[policy == "hard_filter", "sample_weight"])
    weights.loc[policy == "cohort_cl"] = train.loc[policy == "cohort_cl", "cohort_cl_sample_weight"].astype(float)
    weights.loc[policy == "cohort_dm"] = train.loc[policy == "cohort_dm", "cohort_dm_sample_weight"].astype(float)
    train["policy_strategy"] = policy
    train["policy_sample_weight"] = weights.clip(0.03, 1.0)
    return train


def add_baseline_anomaly_weights(scored: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    scored = scored.copy()
    scored["time_burst_weight"] = sigmoid_weight(scored["burst_score"], midpoint=0.50, sharpness=10.0)
    scored["duplicate_weight"] = sigmoid_weight(scored["duplicate_score"], midpoint=0.50, sharpness=10.0)
    scored["rating_cluster_weight"] = sigmoid_weight(scored["low_rating_cluster_score"], midpoint=0.65, sharpness=8.0)

    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_df=0.9, ngram_range=(1, 2), sublinear_tf=True)
    x = tfidf.fit_transform(scored["content"].fillna("").astype(str))
    n_components = min(40, max(2, x.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    dense = svd.fit_transform(x)
    meta = scored[
        [
            "burst_score",
            "duplicate_score",
            "low_rating_cluster_score",
            "version_concentration_score",
            "semantic_compactness_score",
            "cluster_size_score",
        ]
    ].fillna(0)
    features = np.hstack([dense, StandardScaler().fit_transform(meta)])

    iso = IsolationForest(n_estimators=150, contamination=0.15, random_state=seed)
    iso_score = -iso.fit(features).score_samples(features)
    iso_score = minmax(pd.Series(iso_score, index=scored.index))
    scored["isolation_forest_score"] = iso_score
    scored["isolation_forest_weight"] = sigmoid_weight(iso_score, midpoint=0.55, sharpness=10.0)

    lof = LocalOutlierFactor(n_neighbors=35, contamination=0.15)
    lof_pred_score = -lof.fit_predict(features)
    lof_score = -lof.negative_outlier_factor_
    lof_score = minmax(pd.Series(lof_score, index=scored.index))
    scored["lof_score"] = lof_score
    scored["lof_weight"] = sigmoid_weight(lof_score, midpoint=0.55, sharpness=10.0)
    scored["lof_pred_score"] = lof_pred_score
    return scored


def bootstrap_tvd_ci(
    train: pd.DataFrame,
    clean_dist: dict[str, float],
    method: str,
    weight_col: str | None,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(train), len(train))
        sample = train.iloc[idx]
        dist = label_distribution(sample["sentiment"], sample[weight_col] if weight_col else None)
        values.append(total_variation_distance(clean_dist, dist))
    arr = np.array(values)
    return {
        "method": method,
        "tvd_mean": float(arr.mean()),
        "tvd_std": float(arr.std(ddof=1)),
        "tvd_ci95_low": float(np.quantile(arr, 0.025)),
        "tvd_ci95_high": float(np.quantile(arr, 0.975)),
    }
