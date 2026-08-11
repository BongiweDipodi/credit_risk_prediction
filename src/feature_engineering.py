import pandas as pd
from typing import Literal
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier

from src.logger import logger


def encode_categorical(
    df: pd.DataFrame,
    categorical_columns: list[str],
    method: str = 'onehot',
    handle_unknown: Literal['error', 'ignore', 'infrequent_if_exist'] = 'ignore',
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
            encoded_series = pd.Series(
                encoder.fit_transform(df[column].astype(str)),
                index=df.index,
            )
            df[column] = encoded_series
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


def select_features(
    df: pd.DataFrame,
    target: pd.Series,
    method: str = 'f_classif',
    k: int | str = 'all',
) -> tuple[pd.DataFrame, list[str]]:
    """Select features using statistical or model-based methods.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data frame.
    target : pd.Series
        Target variable.
    method : str, optional
        Feature selection method: 'f_classif', 'chi2', or 'model_importance'. Default is 'f_classif'.
    k : int | str, optional
        Number of features to select. If 'all', all features are selected. Default is 'all'.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Selected features DataFrame and list of selected feature names.
    """
    if len(df) == 0:
        error_msg = "Input DataFrame is empty."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(target) != len(df):
        error_msg = "Target length does not match DataFrame length."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if k == 'all':
        k = df.shape[1]
    elif not isinstance(k, int) or k <= 0:
        error_msg = f"k must be a positive integer or 'all', got {k}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    k = min(k, df.shape[1])

    if method == 'f_classif':
        logger.info(f"Selecting top {k} features using f_classif.")
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(df, target)
        selected_indices = selector.get_support(indices=True)
        selected_columns = df.columns[selected_indices].tolist()
        return df.iloc[:, selected_indices], selected_columns

    if method == 'chi2':
        if (df < 0).any().any():
            error_msg = "chi2 requires non-negative features."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Selecting top {k} features using chi2.")
        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(df, target)
        selected_indices = selector.get_support(indices=True)
        selected_columns = df.columns[selected_indices].tolist()
        return df.iloc[:, selected_indices], selected_columns

    if method == 'model_importance':
        logger.info(f"Selecting top {k} features using RandomForest importance.")
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(df, target)
        importance_df = pd.DataFrame(
            {'feature': df.columns, 'importance': clf.feature_importances_}
        )
        importance_df = importance_df.sort_values('importance', ascending=False)
        selected_columns = importance_df.head(k)['feature'].tolist()
        return df[selected_columns], selected_columns

    error_msg = f"Unsupported feature selection method: {method}. Use 'f_classif', 'chi2', or 'model_importance'."
    logger.error(error_msg)
    raise ValueError(error_msg)


def compute_feature_importance(
    df: pd.DataFrame,
    target: pd.Series,
    method: str = 'random_forest',
    top_k: int | None = None,
) -> pd.DataFrame:
    """Compute feature importance scores using model-based or statistical methods.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data frame.
    target : pd.Series
        Target variable.
    method : str, optional
        Importance extraction method: 'random_forest' or 'f_classif'. Default is 'random_forest'.
    top_k : int | None, optional
        If given, returns only the top k features. Default is None (all features).

    Returns
    -------
    pd.DataFrame
        Feature importance scores sorted descending.
    """
    if len(df) == 0:
        error_msg = "Input DataFrame is empty."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if len(target) != len(df):
        error_msg = "Target length does not match DataFrame length."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        error_msg = f"top_k must be a positive integer or None, got {top_k}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if method == 'random_forest':
        logger.info("Computing feature importance using RandomForestClassifier.")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(df, target)
        importance = model.feature_importances_
    elif method == 'f_classif':
        logger.info("Computing feature importance using f_classif scores.")
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(df, target)
        importance = selector.scores_
    else:
        error_msg = f"Unsupported importance method: {method}. Use 'random_forest' or 'f_classif'."
        logger.error(error_msg)
        raise ValueError(error_msg)

    importance_df = pd.DataFrame(
        {'feature': df.columns, 'importance': importance}
    ).sort_values('importance', ascending=False).reset_index(drop=True)

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    return importance_df


def build_preprocessing_pipeline(
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
    encoder_method: str = 'onehot',
) -> Pipeline:
    """Build a preprocessing pipeline combining imputation, encoding, and scaling.

    Parameters
    ----------
    categorical_columns : list[str] | None
        Categorical columns to encode. If None, categorical features are skipped.
    numerical_columns : list[str] | None
        Numerical columns to scale. If None, numeric columns are selected automatically.
    encoder_method : str, optional
        'onehot' or 'ordinal' encoding strategy. Default is 'onehot'.

    Returns
    -------
    Pipeline
        A sklearn Pipeline containing a ColumnTransformer preprocessor.
    """
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    if encoder_method == 'onehot':
        try:
            cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    elif encoder_method == 'ordinal':
        cat_encoder = OrdinalEncoder()
    else:
        error_msg = f"Unsupported encoder_method: {encoder_method}. Use 'onehot' or 'ordinal'."
        logger.error(error_msg)
        raise ValueError(error_msg)

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', cat_encoder),
        ]
    )

    transformers = []
    if numerical_columns is None:
        transformers.append(('num', numeric_transformer, make_column_selector(dtype_include=np.number)))
    else:
        transformers.append(('num', numeric_transformer, numerical_columns))

    if categorical_columns is not None:
        transformers.append(('cat', categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    logger.info('Preprocessing pipeline constructed')
    return pipeline

