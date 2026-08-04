from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.logger import logger


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    method: str = "one_hot",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode categorical columns using one-hot or label encoding."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    if method not in {"one_hot", "label"}:
        raise ValueError("method must be either 'one_hot' or 'label'")

    working_df = df.copy()
    if categorical_columns is None:
        categorical_columns = [
            col for col in working_df.select_dtypes(include=["object", "category"]).columns
        ]

    missing_columns = [col for col in categorical_columns if col not in working_df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    if not categorical_columns:
        logger.info("No categorical columns found for encoding")
        return working_df, {"method": method, "encoded_columns": []}

    metadata: Dict[str, Any] = {
        "method": method,
        "encoded_columns": list(categorical_columns),
        "categories": {},
    }

    if method == "one_hot":
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_array = encoder.fit_transform(working_df[categorical_columns])
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoded_frame = pd.DataFrame(encoded_array, columns=encoded_columns, index=working_df.index)
        metadata["categories"] = {
            col: encoder.categories_[i].tolist() for i, col in enumerate(categorical_columns)
        }
        working_df = pd.concat([working_df.drop(columns=categorical_columns), encoded_frame], axis=1)
        logger.info("Applied one-hot encoding to columns: %s", categorical_columns)
    else:
        for column in categorical_columns:
            encoder = LabelEncoder()
            encoded_values = encoder.fit_transform(working_df[column].astype(str))
            working_df[column] = encoded_values
            metadata["categories"][column] = encoder.classes_.tolist()
        logger.info("Applied label encoding to columns: %s", categorical_columns)

    return working_df, metadata


def scale_numeric_features(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Scale selected numeric features with StandardScaler."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    working_df = df.copy()
    if numeric_columns is None:
        numeric_columns = [
            col for col in working_df.select_dtypes(include=["number"]).columns
        ]

    missing_columns = [col for col in numeric_columns if col not in working_df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    if not numeric_columns:
        logger.info("No numeric columns found for scaling")
        return working_df, {"scaled_columns": []}

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(working_df[numeric_columns])
    scaled_frame = pd.DataFrame(scaled_values, columns=numeric_columns, index=working_df.index)
    scaled_df = pd.concat([working_df.drop(columns=numeric_columns), scaled_frame], axis=1)

    metadata = {
        "scaled_columns": list(numeric_columns),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    logger.info("Applied StandardScaler to columns: %s", numeric_columns)
    return scaled_df, metadata


def select_features(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    top_k: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select the strongest predictive features using mutual information."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    working_df = df.copy()
    if feature_columns is None:
        feature_columns = [col for col in working_df.columns if col != target_column]

    missing_columns = [col for col in feature_columns if col not in working_df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    if not feature_columns:
        logger.info("No feature columns available for selection")
        return working_df[[target_column]], {"selected_features": [], "scores": {}}

    X = working_df[feature_columns].copy()
    y = working_df[target_column]

    for column in X.columns:
        if X[column].dtype == "object" or pd.api.types.is_categorical_dtype(X[column]):
            X[column] = X[column].astype("string").fillna("missing")

    scores = mutual_info_classif(X, y, random_state=42)
    ranked = sorted(zip(feature_columns, scores), key=lambda item: item[1], reverse=True)
    selected = [name for name, _ in ranked[: max(1, min(top_k, len(ranked)))] ]

    metadata = {
        "selected_features": selected,
        "scores": {name: float(score) for name, score in ranked},
    }
    logger.info("Selected features: %s", selected)
    return working_df[[*selected, target_column]], metadata


def build_feature_pipeline(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    feature_columns: Optional[List[str]] = None,
    top_k: int = 5,
    encoding_method: str = "one_hot",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a full feature transformation pipeline: encode → scale → select."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    working_df = df.copy()
    pipeline_metadata: Dict[str, Any] = {}

    encoded_df, encoding_metadata = encode_categorical_features(
        working_df,
        categorical_columns=categorical_columns,
        method=encoding_method,
    )
    pipeline_metadata["encoding"] = encoding_metadata
    working_df = encoded_df

    numeric_candidates = numeric_columns or [
        col for col in working_df.select_dtypes(include=["number"]).columns if col != target_column
    ]
    scaled_df, scaling_metadata = scale_numeric_features(working_df, numeric_columns=numeric_candidates)
    pipeline_metadata["scaling"] = scaling_metadata
    working_df = scaled_df

    selected_df, selection_metadata = select_features(
        working_df,
        target_column=target_column,
        feature_columns=feature_columns or [col for col in working_df.columns if col != target_column],
        top_k=top_k,
    )
    pipeline_metadata["selection"] = selection_metadata

    logger.info("Built feature transformation pipeline for target '%s'", target_column)
    return selected_df, pipeline_metadata


def analyze_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Analyze feature importance using a RandomForest model."""
    if X is None or not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if y is None or not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series")

    model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    ranked = importances.to_dict()
    top_features = list(importances.head(max(1, min(top_k, len(importances)))).index)

    metadata = {
        "top_features": top_features,
        "importance_scores": ranked,
        "model": "RandomForestClassifier",
    }
    logger.info("Computed feature importance for %s features", len(importances))
    return metadata

