from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

