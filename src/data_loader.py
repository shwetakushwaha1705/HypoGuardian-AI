"""HypoGuardian AI: Data Ingestion Module.

Loads and validates blood glucose data from CSV files and OhioT1DM research XML files.
Provides standardized DataFrame output with columns: ['patient_id', 'timestamp', 'glucose'].
"""

import os
from pathlib import Path
from typing import BinaryIO, TextIO, Union
import xml.etree.ElementTree as ET
import pandas as pd


class DataIngestionError(Exception):
    """Base exception for data loading errors."""
    pass


class MissingColumnsError(DataIngestionError):
    """Raised when uploaded or loaded data is missing mandatory columns."""
    pass


class EmptyDatasetError(DataIngestionError):
    """Raised when the data file contains zero valid rows."""
    pass


def load_glucose_csv(
    source: Union[str, Path, TextIO, BinaryIO, pd.DataFrame],
    default_patient_id: str = "PATIENT_01",
) -> pd.DataFrame:
    """Loads and standardizes a blood glucose CSV data source.

    Supports file paths, Streamlit UploadedFile buffers, and existing DataFrames.

    Args:
        source: File path, file-like object, or existing pandas DataFrame.
        default_patient_id: Fallback identifier if 'patient_id' column is absent.

    Returns:
        Standardized pandas DataFrame with columns: ['patient_id', 'timestamp', 'glucose'].

    Raises:
        EmptyDatasetError: If the data source is empty.
        MissingColumnsError: If required columns ('timestamp', 'glucose') are absent.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        try:
            df = pd.read_csv(source)
        except Exception as e:
            raise DataIngestionError(f"Failed to parse CSV file: {str(e)}") from e

    if df.empty:
        raise EmptyDatasetError("The provided dataset is empty (0 rows).")

    # Normalize column names: strip whitespace and lowercase
    column_mapping = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=column_mapping)

    # Check for alias columns (e.g., 'time', 'date', 'datetime' -> 'timestamp')
    alias_map = {
        "time": "timestamp",
        "datetime": "timestamp",
        "date_time": "timestamp",
        "ts": "timestamp",
        "bg": "glucose",
        "blood_glucose": "glucose",
        "cgm": "glucose",
        "glucose_value": "glucose",
        "value": "glucose",
        "patient": "patient_id",
        "id": "patient_id",
        "user_id": "patient_id",
    }
    for alias, target in alias_map.items():
        if alias in df.columns and target not in df.columns:
            df = df.rename(columns={alias: target})

    # Validate mandatory columns
    required_cols = {"timestamp", "glucose"}
    missing = required_cols - set(df.columns)
    if missing:
        raise MissingColumnsError(
            f"Missing mandatory column(s): {sorted(list(missing))}. "
            f"Required columns are 'timestamp' and 'glucose'."
        )

    # If patient_id is not provided, populate with default ID
    if "patient_id" not in df.columns:
        df["patient_id"] = default_patient_id

    # Retain standard columns plus optional source column if present
    cols_to_keep = ["patient_id", "timestamp", "glucose"]
    if "source" in df.columns:
        cols_to_keep.append("source")

    return df[cols_to_keep].copy()


def parse_ohiot1dm_xml(xml_path: Union[str, Path]) -> pd.DataFrame:
    """Parses official OhioT1DM research XML files.

    OhioT1DM schema structure:
    <patient id="559">
        <glucose_level>
            <event ts="21-05-2021 00:00:00" value="130"/>
            ...
        </glucose_level>
    </patient>

    Args:
        xml_path: Path to the OhioT1DM XML file.

    Returns:
        Standardized DataFrame with columns: ['patient_id', 'timestamp', 'glucose', 'source'].
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"OhioT1DM XML file not found at: {path}")

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        raise DataIngestionError(f"Error parsing XML file {path.name}: {str(e)}") from e

    patient_id = root.attrib.get("id", path.stem)
    records = []

    glucose_level_elem = root.find("glucose_level")
    if glucose_level_elem is not None:
        for event in glucose_level_elem.findall("event"):
            ts_str = event.attrib.get("ts")
            val_str = event.attrib.get("value")
            if ts_str and val_str:
                try:
                    records.append({
                        "patient_id": patient_id,
                        "timestamp": ts_str,
                        "glucose": float(val_str),
                        "source": "OhioT1DM_Research_XML",
                    })
                except ValueError:
                    continue  # Ignore corrupt numeric value

    if not records:
        raise EmptyDatasetError(f"No glucose event records found in XML file: {path.name}")

    return pd.DataFrame(records)
