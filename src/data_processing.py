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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic data cleaning"""
    df = df.drop('clientid', axis=1)
    df = df.dropna()
    return df


if __name__ == "__main__":
    df = load_data("../data/original.csv")
    df = clean_data(df)
    print(df.head())
