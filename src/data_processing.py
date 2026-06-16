import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame with comprehensive error handling.
    """
    
    # Convert to Path object for better path handling
    file_path_obj = Path(file_path)
    
    # Validate file existence
    if not file_path_obj.exists():
        error_msg = f"File not found: {file_path}. Please check the file path."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Validate file extension
    if file_path_obj.suffix.lower() != '.csv':
        error_msg = f"Invalid file type: {file_path_obj.suffix}. Expected .csv file."
        logger.warning(f"Warning: File extension is {file_path_obj.suffix}, not .csv")
    
    # Check file permissions
    if not os.access(file_path_obj, os.R_OK):
        error_msg = f"Permission denied: Cannot read file {file_path}."
        logger.error(error_msg)
        raise PermissionError(error_msg)
    
    # Check if file is empty
    if file_path_obj.stat().st_size == 0:
        error_msg = f"File is empty: {file_path}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Load the CSV file
        logger.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path, encoding=encoding)
        
        # Validate that data was loaded
        if df.empty:
            error_msg = f"File {file_path} contains no data rows."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log success information
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types:\n{df.dtypes}")
        
        return df
    
    except UnicodeDecodeError as e:
        error_msg = f"Encoding error: Could not decode file {file_path} with {encoding} encoding. Try a different encoding."
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    except pd.errors.ParserError as e:
        error_msg = f"CSV parsing error: {str(e)}. The file may be corrupted or have an invalid format."
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    except Exception as e:
        error_msg = f"Unexpected error while loading {file_path}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def clean_data(
    df: pd.DataFrame,
    drop_columns: Optional[list[str]] = None,
    missing_strategy: str = 'drop',
    outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    """Clean a DataFrame by removing duplicates, handling missing values, and removing outliers.
    """
    df = df.copy()

    if drop_columns is None:
        drop_columns = ['clientid']

    existing_drop_columns = [col for col in drop_columns if col in df.columns]
    if existing_drop_columns:
        logger.info(f"Dropping columns: {existing_drop_columns}")
        df = df.drop(columns=existing_drop_columns)

    initial_shape = df.shape
    df = df.drop_duplicates()
    if df.shape != initial_shape:
        logger.info(f"Dropped duplicates: {initial_shape[0] - df.shape[0]} rows removed")

    if missing_strategy == 'drop':
        before_missing = df.shape[0]
        df = df.dropna()
        logger.info(f"Dropped rows with missing values: {before_missing - df.shape[0]} rows removed")
    elif missing_strategy == 'fill':
        numeric_columns = df.select_dtypes(include=['number']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        for column in numeric_columns:
            if df[column].isna().any():
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                logger.info(f"Filled missing numeric values in {column} with median={median_value}")

        for column in categorical_columns:
            if df[column].isna().any():
                mode_value = df[column].mode()
                if not mode_value.empty:
                    df[column] = df[column].fillna(mode_value.iloc[0])
                    logger.info(f"Filled missing categorical values in {column} with mode={mode_value.iloc[0]}")
                else:
                    df[column] = df[column].fillna('unknown')
                    logger.info(f"Filled missing categorical values in {column} with 'unknown'")
    else:
        error_msg = f"Unknown missing_strategy '{missing_strategy}'. Use 'drop' or 'fill'."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if outlier_threshold and outlier_threshold > 0:
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            z_scores = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std(ddof=0)
            outlier_mask = z_scores.abs().gt(outlier_threshold).any(axis=1)
            removed_count = int(outlier_mask.sum())
            if removed_count > 0:
                df = df.loc[~outlier_mask].reset_index(drop=True)
                logger.info(f"Removed outliers: {removed_count} rows removed using z-score threshold {outlier_threshold}")

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def validate_data(
    df: pd.DataFrame,
    expected_columns: Optional[list[str]] = None,
    required_columns: Optional[list[str]] = None,
    missing_threshold: float = 0.1,
    numeric_ranges: Optional[Dict[str, tuple[float, float]]] = None,
    categorical_values: Optional[Dict[str, list[Any]]] = None,
) -> Dict[str, Any]:
    """Validate dataset quality and return a summary of issues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    expected_columns : Optional[list[str]], optional
        List of expected columns. Missing or extra columns are reported.
    required_columns : Optional[list[str]], optional
        Columns that must not contain missing values.
    missing_threshold : float, optional
        Maximum allowed proportion of missing values per column.
    numeric_ranges : Optional[Dict[str, tuple[float, float]]], optional
        Acceptable ranges for numeric columns.
    categorical_values : Optional[Dict[str, list[Any]]], optional
        Allowed values for categorical columns.

    Returns
    -------
    Dict[str, Any]
        Summary of validation checks.

    Raises
    ------
    ValueError
        If validation fails for required schema or critical data quality issues.
    """
    summary: Dict[str, Any] = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_values': {},
        'duplicate_rows': 0,
        'unexpected_columns': [],
        'missing_columns': [],
        'out_of_range': {},
        'invalid_categories': {},
    }

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    summary['duplicate_rows'] = duplicate_count
    if duplicate_count > 0:
        logger.warning(f"Data contains {duplicate_count} duplicate rows")

    # Missing values
    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / len(df)).to_dict()
    summary['missing_values'] = {col: {
        'count': int(missing_counts[col]),
        'percent': float(missing_percent[col]),
    } for col in df.columns}

    # Missing threshold check
    columns_over_threshold = [
        col for col, percent in missing_percent.items() if percent > missing_threshold
    ]
    if columns_over_threshold:
        logger.warning(
            f"Columns exceed missing value threshold ({missing_threshold}): {columns_over_threshold}"
        )
        summary['columns_over_missing_threshold'] = columns_over_threshold

    # Schema checks
    if expected_columns is not None:
        missing_cols = [col for col in expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_columns]
        summary['missing_columns'] = missing_cols
        summary['unexpected_columns'] = extra_cols

        if missing_cols:
            error_msg = f"Missing expected columns: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    if required_columns is not None:
        missing_required = [col for col in required_columns if df[col].isna().any()]
        if missing_required:
            error_msg = f"Required columns contain missing values: {missing_required}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Numeric range validation
    if numeric_ranges is not None:
        for column, (low, high) in numeric_ranges.items():
            if column not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                summary['out_of_range'][column] = 'non-numeric column'
                continue
            invalid_mask = df[column].lt(low) | df[column].gt(high)
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                summary['out_of_range'][column] = {
                    'invalid_count': invalid_count,
                    'allowed_range': (low, high),
                }
                logger.warning(
                    f"Column '{column}' has {invalid_count} values outside range {low} to {high}"
                )

    # Categorical values validation
    if categorical_values is not None:
        for column, allowed_values in categorical_values.items():
            if column not in df.columns:
                continue
            invalid_values = df.loc[~df[column].isin(allowed_values), column].unique().tolist()
            if invalid_values:
                summary['invalid_categories'][column] = invalid_values
                logger.warning(
                    f"Column '{column}' contains invalid categories: {invalid_values}"
                )

    logger.info("Data validation completed")
    return summary


if __name__ == "__main__":
    df = load_data("../data/original.csv")
    df = clean_data(df)
    print(df.head())
