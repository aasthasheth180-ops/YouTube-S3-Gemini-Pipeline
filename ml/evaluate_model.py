import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report
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


from ml.train_model import train_models


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model):

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importances = (
        classifier.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    return importance_df


# =========================================================
# MAIN EVALUATION
# =========================================================

def evaluate_best_model():

    training_output = train_models()

    results = training_output[
        "results"
    ]

    X_test = training_output[
        "X_test"
    ]

    y_test = training_output[
        "y_test"
    ]

    random_forest_model = training_output[
        "random_forest_model"
    ]

    # =====================================================
    # SELECT BEST MODEL
    # =====================================================

    model_results = results[
        results["model"]
        != "Majority Baseline"
    ].copy()

    best_result = model_results.sort_values(
        [
            "f1",
            "roc_auc"
        ],
        ascending=False
    ).iloc[0]

    print(
        "\nBest model:"
    )

    print(
        best_result.to_string()
    )

    # For our current candidates, Random Forest should win.
    # We explicitly use the corresponding trained pipeline.

    if (
        best_result["model"]
        == "Random Forest"
    ):
        best_model = random_forest_model

    else:
        best_model = training_output[
            "logistic_model"
        ]

    # =====================================================
    # DETAILED TEST EVALUATION
    # =====================================================

    predictions = best_model.predict(
        X_test
    )

    print(
        "\nConfusion matrix:"
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print(matrix)

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    if (
        best_result["model"]
        == "Random Forest"
    ):

        importance_df = (
            get_feature_importance(
                best_model
            )
        )

        print(
            "\nTop 20 model features:"
        )

        print(
            importance_df
            .head(20)
            .to_string(
                index=False
            )
        )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    models_directory = os.path.join(
        PROJECT_ROOT,
        "models"
    )

    os.makedirs(
        models_directory,
        exist_ok=True
    )

    model_path = os.path.join(
        models_directory,
        "trending_persistence_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        "\nSaved best model:"
    )

    print(
        model_path
    )


if __name__ == "__main__":

    evaluate_best_model()