import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.logger import logger


def encode_categorical(
    df: pd.DataFrame,
    categorical_columns: list[str],
    method: str = 'onehot',
    handle_unknown: str = 'ignore',
) -> tuple[pd.DataFrame, object]:
    """Encode categorical columns using either one-hot or label encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    categorical_columns : list[str]
        Columns to encode.
    method : str, optional
        Encoding method: 'onehot' or 'label'. Default is 'onehot'.
    handle_unknown : str, optional
        Behavior for unknown categories in one-hot encoding. Default is 'ignore'.

    Returns
    -------
    tuple[pd.DataFrame, object]
        Encoded DataFrame and encoder object.
    """
    df = df.copy()

    missing_columns = [col for col in categorical_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Categorical columns not found in DataFrame: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if method == 'label':
        encoders = {}
        for column in categorical_columns:
            encoder = LabelEncoder()
            logger.info(f"Label encoding column: {column}")
            df[column] = encoder.fit_transform(df[column].astype(str))
            encoders[column] = encoder
        return df, encoders

    if method == 'onehot':
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown=handle_unknown)

        logger.info(f"One-hot encoding columns: {categorical_columns}")
        encoded_array = encoder.fit_transform(df[categorical_columns].astype(str))
        encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df.index)
        df = df.drop(columns=categorical_columns)
        df = pd.concat([df, encoded_df], axis=1)
        return df, encoder

    error_msg = f"Unsupported encoding method: {method}. Use 'onehot' or 'label'."
    logger.error(error_msg)
    raise ValueError(error_msg)


def scale_numerical(
    df: pd.DataFrame,
    numerical_columns: list[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical columns using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame.
    numerical_columns : list[str] | None
        Columns to scale. If None, all numeric columns are selected.
    scaler : StandardScaler | None
        Existing scaler instance to use. If None, a new StandardScaler is created.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        Scaled DataFrame and fitted scaler.
    """
    df = df.copy()

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        logger.info(f"Auto-selected numerical columns for scaling: {numerical_columns}")

    missing_columns = [col for col in numerical_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Numerical columns not found in DataFrame: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not numerical_columns:
        error_msg = "No numerical columns available for scaling."
        logger.error(error_msg)
        raise ValueError(error_msg)

    scaler = scaler or StandardScaler()
    logger.info(f"Scaling numerical columns: {numerical_columns}")
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

