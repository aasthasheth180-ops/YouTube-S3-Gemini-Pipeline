import os
import sys
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# PROJECT IMPORTS
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.build_ml_dataset import prepare_ml_dataset


# =========================================================
# MODEL FEATURES
# =========================================================

NUMERIC_FEATURES = [
    "trending_rank",
    "previous_rank",
    "rank_movement",
    "observations_so_far",
    "hours_trending_so_far",
    "days_trending_so_far",
    "view_count",
    "view_growth",
    "view_velocity",
    "like_velocity",
    "comment_velocity",
    "engagement_rate",
    "engagement_movement",
    "channel_subscriber_count",
    "channel_view_count",
    "channel_video_count",
    "regions_trending_today",
    "cross_region_today"
]

CATEGORICAL_FEATURES = [
    "category_name",
    "region_code"
]

TARGET = "still_trending_tomorrow"


# =========================================================
# TEMPORAL TRAIN / TEST SPLIT
# =========================================================

def temporal_train_test_split(
    ml_df,
    test_fraction=0.20
):
    """
    Split chronologically instead of randomly.

    Training = older dates
    Testing  = newest dates

    We also purge the training date immediately before
    the test period because its target refers to the
    following calendar day.
    """

    df = ml_df.copy()

    df["fetch_date"] = pd.to_datetime(
        df["fetch_date"]
    )

    unique_dates = sorted(
        df["fetch_date"].dropna().unique()
    )

    split_index = int(
        len(unique_dates)
        * (1 - test_fraction)
    )

    test_start_date = pd.Timestamp(
        unique_dates[split_index]
    )

    # Target for date D tells us what happens on D+1.
    # Therefore training rows must have their target outcome
    # completely before the test period begins.

    latest_allowed_train_date = (
        test_start_date
        - pd.Timedelta(days=2)
    )

    train_df = df[
        df["fetch_date"]
        <= latest_allowed_train_date
    ].copy()

    test_df = df[
        df["fetch_date"]
        >= test_start_date
    ].copy()

    return (
        train_df,
        test_df,
        test_start_date
    )


# =========================================================
# PREPROCESSING
# =========================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# =========================================================
# EVALUATION
# =========================================================

def evaluate_predictions(
    model_name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# =========================================================
# TRAIN MODELS
# =========================================================

def train_models():

    print(
        "Building Phase 8 ML dataset..."
    )

    ml_df = prepare_ml_dataset()

    print(
        f"ML rows available: {len(ml_df)}"
    )

    train_df, test_df, test_start_date = (
        temporal_train_test_split(
            ml_df
        )
    )

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        TARGET
    ]

    print(
        "\nTemporal split:"
    )

    print(
        "Training:",
        train_df["fetch_date"].min(),
        "→",
        train_df["fetch_date"].max()
    )

    print(
        "Testing:",
        test_df["fetch_date"].min(),
        "→",
        test_df["fetch_date"].max()
    )

    print(
        "Test starts:",
        test_start_date
    )

    print(
        "\nTraining rows:",
        len(train_df)
    )

    print(
        "Testing rows:",
        len(test_df)
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.value_counts(
            normalize=True
        ).sort_index().round(4)
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test.value_counts(
            normalize=True
        ).sort_index().round(4)
    )

    # =====================================================
    # MODEL 1 — MAJORITY BASELINE
    # =====================================================

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(
        X_train,
        y_train
    )

    # =====================================================
    # MODEL 2 — LOGISTIC REGRESSION
    # =====================================================

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # MODEL 3 — RANDOM FOREST
    # =====================================================

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # COMPARE MODELS
    # =====================================================

    results = []

    results.append(
        evaluate_predictions(
            "Majority Baseline",
            baseline,
            X_test,
            y_test
        )
    )

    results.append(
        evaluate_predictions(
            "Logistic Regression",
            logistic_model,
            X_test,
            y_test
        )
    )

    results.append(
        evaluate_predictions(
            "Random Forest",
            random_forest_model,
            X_test,
            y_test
        )
    )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\nModel comparison:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    return {
        "ml_df": ml_df,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "baseline": baseline,
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
        "results": results_df
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_models()