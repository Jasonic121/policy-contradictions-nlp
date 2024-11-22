"""
Utilities for saving and loading policy documents and pipeline results.
"""

import pickle
from pathlib import Path
from typing import Dict, Union
import string
import random
import os

import pandas as pd
from haystack import Document
from pandas import DataFrame


def extract_raw_text_from_pdf(url: str) -> str:
    raise NotImplementedError


def load_dataset_from_markdown(directory: Union[str, Path]) -> pd.DataFrame:
    """Load markdown files from a directory into a pandas DataFrame."""
    directory = Path(directory)
    data = []
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    for filepath in directory.glob('*.md'):
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            data.append({
                'filename': filepath.name,
                'text_by_page': [content],  # Wrap in list to match expected format
                'filepath': str(filepath),
                'url': str(filepath)  # Added to match expected schema
            })
    
    if not data:
        raise ValueError(f"No markdown files found in directory: {directory}")
        
    return pd.DataFrame(data)


def load_dataset_from_json(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load dataset from either JSON file or directory of markdown files."""
    filepath = Path(filepath)
    
    # Check if input is a directory
    if filepath.is_dir():
        try:
            return load_dataset_from_markdown(filepath)
        except Exception as e:
            print(f"Error loading markdown files: {e}")
            raise
    
    # If not a directory, try loading as JSON
    try:
        df = pd.read_json(str(filepath))
        return df
    except ValueError as e:
        print(f"Error loading JSON file: {e}")
        print("Attempting to load as directory of markdown files...")
        return load_dataset_from_markdown(filepath)


def load_dataset_from_pyspark(table_name: str) -> pd.DataFrame:
    from pyspark.sql.context import SparkSession
    spark = SparkSession.getActiveSession()
    sql_cmd = spark.sql(f"SELECT * FROM {table_name}")
    df = sql_cmd.toPandas()
    return df


def save_chunks_pickle(
    chunks: Dict[str, Document], filepath: Path, overwrite: bool = False
) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and Path(filepath).exists():
        rand_id = _get_random_id()
        old = Path(filepath)
        filepath = Path(old.parent, old.stem + '_' + rand_id + old.suffix)
        print(f"[!] WARNING: File {old} already exists! Writing to {filepath} instead.")
    with open(filepath, 'wb') as f:
        pickle.dump(chunks, f)


def load_chunks_pickle(filepath: Path) -> Dict[str, Document]:
    with open(filepath, 'rb') as f:
        chunks = pickle.load(f)
    return chunks


def save_candidates_csv(
    candidates: DataFrame, filepath: Path, overwrite: bool = False
) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and Path(filepath).exists():
        rand_id = _get_random_id()
        old = Path(filepath)
        filepath = Path(old.parent, old.stem + '_' + rand_id + old.suffix)
        print(f"[!] WARNING: File {old} already exists! Writing to {filepath} instead.")
    candidates.to_csv(filepath, index=False)


def load_candidates_csv(filepath: Path) -> DataFrame:
    candidates = pd.read_csv(filepath)
    return candidates


def _get_random_id(size=8) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=size))