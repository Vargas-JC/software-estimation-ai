import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.core.config import ARTIFACTS_DIR, DEFAULT_DATASET_PATH
from app.ml.dataset_gen import generate_training_dataset
from app.schemas import ClassificationModel, RegressionModel

CATEGORICAL_FEATURES = ["project_type"]
NUMERIC_FEATURES = [
    "base_effort_points",
    "total_effort_hours",
    "stories_total",
    "tasks_total",
    "total_sprints",
    "planned_days",
    "junior_ratio",
    "senior_ratio",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def train_and_evaluate(data_path: Path | None = None) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if data_path is not None:
        df = pd.read_csv(data_path)
    elif DEFAULT_DATASET_PATH.exists():
        df = pd.read_csv(DEFAULT_DATASET_PATH)
    else:
        df = generate_training_dataset()

    X = df[ALL_FEATURES]
    y_reg = df["contingency_margin"]
    y_clf = df["risk_level"]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    regression_candidates = {
        RegressionModel.LINEAR_REGRESSION.value: (
            Pipeline([("prep", _build_preprocessor()), ("model", LinearRegression())]),
            {},
        ),
        RegressionModel.RIDGE.value: (
            Pipeline([("prep", _build_preprocessor()), ("model", Ridge(random_state=42))]),
            {"model__alpha": [0.1, 1.0, 10.0]},
        ),
        RegressionModel.POLYNOMIAL_RIDGE.value: (
            Pipeline(
                [
                    ("prep", _build_preprocessor()),
                    ("poly", PolynomialFeatures(include_bias=False)),
                    ("model", Ridge(random_state=42)),
                ]
            ),
            {"poly__degree": [2, 3], "model__alpha": [1.0, 10.0]},
        ),
        RegressionModel.RANDOM_FOREST_REGRESSOR.value: (
            Pipeline(
                [
                    ("prep", _build_preprocessor()),
                    ("model", RandomForestRegressor(random_state=42)),
                ]
            ),
            {"model__n_estimators": [100, 200], "model__max_depth": [6, 12, None]},
        ),
    }

    regression_results: dict[str, dict] = {}
    best_reg_name = ""
    best_reg_model = None
    best_reg_rmse = float("inf")

    for name, (pipeline, param_grid) in regression_candidates.items():
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_reg_train)
        preds = search.best_estimator_.predict(X_test)
        rmse = float(root_mean_squared_error(y_reg_test, preds))
        mae = float(mean_absolute_error(y_reg_test, preds))
        r2 = float(r2_score(y_reg_test, preds))

        regression_results[name] = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "best_params": search.best_params_,
        }

        if rmse < best_reg_rmse:
            best_reg_rmse = rmse
            best_reg_name = name
            best_reg_model = search.best_estimator_

    classification_candidates = {
        ClassificationModel.LOGISTIC_REGRESSION.value: (
            Pipeline(
                [
                    ("prep", _build_preprocessor()),
                    ("model", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
            False,
        ),
        ClassificationModel.SVM.value: (
            Pipeline(
                [
                    ("prep", _build_preprocessor()),
                    ("model", SVC(probability=True, random_state=42)),
                ]
            ),
            {"model__C": [0.5, 1.0, 5.0], "model__kernel": ["linear", "rbf"]},
            False,
        ),
        ClassificationModel.KNN.value: (
            Pipeline([("prep", _build_preprocessor()), ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 7, 11], "model__weights": ["uniform", "distance"]},
            False,
        ),
        ClassificationModel.DECISION_TREE.value: (
            Pipeline(
                [
                    ("prep", _build_preprocessor()),
                    ("model", DecisionTreeClassifier(random_state=42)),
                ]
            ),
            {"model__max_depth": [4, 6, 10, None], "model__min_samples_split": [2, 5]},
            False,
        ),
        ClassificationModel.RANDOM_FOREST.value: (
            Pipeline(
                [
                    ("prep", _build_preprocessor()),
                    ("model", RandomForestClassifier(random_state=42)),
                ]
            ),
            {"model__n_estimators": [100, 200], "model__max_depth": [6, 12, None]},
            True,
        ),
        ClassificationModel.GRADIENT_BOOSTING.value: (
            Pipeline(
                [
                    ("prep", _build_preprocessor()),
                    ("model", GradientBoostingClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [100, 150],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
            },
            True,
        ),
    }

    classification_results: dict[str, dict] = {}
    best_clf_name = ""
    best_clf_model = None
    best_clf_f1 = -1.0

    for name, (pipeline, param_grid, is_ensemble) in classification_candidates.items():
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        search.fit(X_train, y_clf_train)
        estimator = search.best_estimator_

        preds = estimator.predict(X_test)
        probs = estimator.predict_proba(X_test)
        labels = list(estimator.classes_)

        acc = float(accuracy_score(y_clf_test, preds))
        prec = float(precision_score(y_clf_test, preds, average="weighted", zero_division=0))
        rec = float(recall_score(y_clf_test, preds, average="weighted", zero_division=0))
        f1 = float(f1_score(y_clf_test, preds, average="weighted", zero_division=0))
        roc_auc = float(roc_auc_score(y_clf_test, probs, multi_class="ovr", labels=labels))

        classification_results[name] = {
            "is_ensemble": is_ensemble,
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "f1_weighted": round(f1, 4),
            "roc_auc_ovr": round(roc_auc, 4),
            "best_params": search.best_params_,
            "labels": labels,
            "confusion_matrix": confusion_matrix(y_clf_test, preds, labels=labels).tolist(),
        }

        if f1 > best_clf_f1:
            best_clf_f1 = f1
            best_clf_name = name
            best_clf_model = estimator

    joblib.dump(best_reg_model, ARTIFACTS_DIR / "contingency_regressor.joblib")
    joblib.dump(best_clf_model, ARTIFACTS_DIR / "risk_classifier.joblib")

    report = {
        "selected_regression_model": best_reg_name,
        "selected_classification_model": best_clf_name,
        "regression_comparison": regression_results,
        "classification_comparison": classification_results,
        "dataset_rows": int(len(df)),
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    result = train_and_evaluate(args.data)
    print(json.dumps(result, indent=2))
