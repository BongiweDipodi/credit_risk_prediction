from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split

from src.logger import logger


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets for supervised learning."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    target = df[target_column]
    features = df.drop(columns=[target_column])

    if stratify is None:
        stratify = target.nunique() <= 2

    effective_test_size = test_size
    if stratify:
        min_required_test_size = target.nunique() / len(target)
        if min_required_test_size >= 1:
            raise ValueError("Cannot stratify split with fewer rows than classes")
        effective_test_size = max(test_size, min_required_test_size)

        try:
            train_features, test_features, train_target, test_target = train_test_split(
                features,
                target,
                test_size=effective_test_size,
                random_state=random_state,
                stratify=target,
            )
        except ValueError:
            train_features, test_features, train_target, test_target = train_test_split(
                features,
                target,
                test_size=effective_test_size,
                random_state=random_state,
            )
    else:
        train_features, test_features, train_target, test_target = train_test_split(
            features,
            target,
            test_size=effective_test_size,
            random_state=random_state,
        )

    logger.info(
        "Split data into %s train and %s test rows",
        len(train_features),
        len(test_features),
    )
    return train_features, test_features, train_target, test_target


def build_training_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create a training dataset bundle with train/test splits and metadata."""
    train_features, test_features, train_target, test_target = split_data(
        df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "train_features": train_features,
        "test_features": test_features,
        "train_target": train_target,
        "test_target": test_target,
        "target_column": target_column,
        "test_size": test_size,
        "random_state": random_state,
    }


def train_model_pipeline(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[bool] = None,
    persist_artifacts: bool = False,
    artifact_dir: Optional[str] = None,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Train a RandomForest model and return metrics plus the fitted model."""
    dataset = build_training_dataset(
        df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    base_model = RandomForestClassifier(random_state=random_state)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(
        n_splits=min(cv_folds, len(dataset["train_target"].unique())),
        shuffle=True,
        random_state=random_state,
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
    )
    grid_search.fit(dataset["train_features"], dataset["train_target"])

    model = grid_search.best_estimator_
    model.fit(dataset["train_features"], dataset["train_target"])

    predictions = model.predict(dataset["test_features"])
    metrics = {
        "accuracy": float(accuracy_score(dataset["test_target"], predictions)),
        "precision": float(precision_score(dataset["test_target"], predictions, zero_division=0)),
        "recall": float(recall_score(dataset["test_target"], predictions, zero_division=0)),
        "f1": float(f1_score(dataset["test_target"], predictions, zero_division=0)),
        "best_params": grid_search.best_params_,
    }

    cv_scores = cross_val_score(model, dataset["train_features"], dataset["train_target"], cv=cv, scoring="accuracy")
    metrics["cv_mean_accuracy"] = float(cv_scores.mean())
    metrics["cv_std_accuracy"] = float(cv_scores.std())

    artifact_paths: Dict[str, str] = {}
    if persist_artifacts:
        output_dir = Path(artifact_dir or "models")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "random_forest_model.joblib"
        metrics_path = output_dir / "training_metrics.json"
        joblib.dump(model, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        artifact_paths = {
            "model": str(model_path),
            "metrics": str(metrics_path),
        }

    logger.info("Trained RandomForest model with accuracy %.4f", metrics["accuracy"])
    return {
        "model": model,
        "metrics": metrics,
        "dataset": dataset,
        "artifact_paths": artifact_paths,
    }


def main() -> None:
    """Run the training pipeline on the repository's sample data if available."""
    data_path = Path("data") / "credit_risk.csv"
    if not data_path.exists():
        logger.info("No training dataset found at %s; skipping run", data_path)
        return

    df = pd.read_csv(data_path)
    train_model_pipeline(df, target_column="risk", persist_artifacts=True, artifact_dir="models")


if __name__ == "__main__":
    main()

