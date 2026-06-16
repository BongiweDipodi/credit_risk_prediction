import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional

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

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    drop_columns : Optional[list[str]], optional
        Columns to drop before cleaning. Defaults to ['clientid'] if present.
    missing_strategy : str, optional
        Strategy for missing values: 'drop' or 'fill'. Default is 'drop'.
    outlier_threshold : float, optional
        Z-score threshold for numeric outlier removal. Set to 0 to disable. Default is 3.0.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
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


if __name__ == "__main__":
    df = load_data("../data/original.csv")
    df = clean_data(df)
    print(df.head())
