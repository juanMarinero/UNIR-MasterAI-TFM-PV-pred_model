#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=6 foldmethod=indent :

import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from functools import reduce
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta


def get_general_config(
    BUCKET="oedi-data-lake",
    BASE_KEY="pvdaq/2023-solar-data-prize/9068_OEDI/data",
    START_DATE="2025-01-01",  # Initial MVP time window
    END_DATE="2025-02-01",  # ...just January 2025
    OUTPUT_DIR=None,
):
    """
    Get general configuration parameters for the data processing pipeline.

    Args:
        BUCKET: Name of the S3 bucket containing the data (default: "oedi-data-lake")
        BASE_KEY: Base S3 key path to the data files
        START_DATE: Start date for data filtering (format: YYYY-MM-DD)
        END_DATE: End date for data filtering (format: YYYY-MM-DD, exclusive)
        OUTPUT_DIR: Directory path where output files will be saved
                    If None, defaults to '../dataset' relative to this script.

    Returns:
        Dictionary containing all configuration parameters
    """

    # If no output directory provided, use '../dataset' relative to this script
    if OUTPUT_DIR is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.resolve()
        # Go up one level to project root, then into 'dataset'
        OUTPUT_DIR = script_dir.parent / "dataset"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return dict(
        BUCKET=BUCKET,
        BASE_KEY=BASE_KEY,
        START_DATE=START_DATE,
        END_DATE=END_DATE,
        OUTPUT_DIR=OUTPUT_DIR,
    )


def get_csv_suffix(START_DATE: str, END_DATE: str):

    datasets_link = "https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=pvdaq%2F2023-solar-data-prize%2F9068_OEDI%2Fdata%2F&limit=50"
    metadata_link = "https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=pvdaq%2F2023-solar-data-prize%2F9068_OEDI%2Fmetadata%2F&limit=50"
    links_msg = f"\nCheck the time window: {START_DATE} to {END_DATE}\n\nDatasets: {datasets_link}\nMetadata: {metadata_link}"

    if START_DATE >= END_DATE:
        raise ValueError(
            "The initial time window must be shorter than the final one." + links_msg
        )
    elif START_DATE >= "2017-08-23" and END_DATE <= "2023-11-16":
        # head 9068_ac_power_data.csv # 2017-08-23 18:15:00
        # tail 9068_ac_power_data.csv # 2023-11-16 13:40:00
        return "data"
    elif START_DATE >= "2017-08-23" and END_DATE <= "2024-01-01":
        print("Blank data from 2023-11-16 to 2024-01-01")
        return "data"
    elif START_DATE >= "2024-01-01" and END_DATE <= "2025-05-01":
        return "data_20240101_20250430"
    else:
        raise ValueError(
            "Not valid time window. Pass it directly to the function or fix dates."
            + links_msg,
        )


def get_files_config(BASE_KEY: str, START_DATE: str, END_DATE: str, csv_suffix=""):
    """
    Returns a dictionary with the configuration of the selected files and columns.
    """

    if csv_suffix == "":
        # Get CSV basename based on timestamp
        csv_suffix = get_csv_suffix(START_DATE, END_DATE)

    csv_suffix_meter = csv_suffix
    if csv_suffix == "data":  # i.e. for date < 2024-01-01
        # Add Power Factor prefix
        csv_suffix_meter = "pf_" + csv_suffix_meter

    meter_ac_power = "meter_ac_power_(kw)_meter_150150"
    files_config = {
        "meter": {
            "key": f"{BASE_KEY}/9068_meter_{csv_suffix_meter}.csv",
            "usecols": [
                "measured_on",
                meter_ac_power,  # in "meter"-file cause not in "ac_power"-file if date >= 2024-01-01
            ],
        },
        "ac_power": {
            "key": f"{BASE_KEY}/9068_ac_power_{csv_suffix}.csv",
            "usecols": [
                "measured_on",
                "inverter_1_ac_power_(kw)_inv_150143",
                "inverter_2_ac_power_(kw)_inv_150144",
                meter_ac_power,  # in "ac_power"-file cause not in "meter"-file if date < 2024-01-01
            ],
        },
        "irradiance": {
            "key": f"{BASE_KEY}/9068_irradiance_{csv_suffix}.csv",
            "usecols": [
                "measured_on",
                "pyranometer_(class_a)_pad_1_poa_irradiance_temp_compensated_(w/m2)_o_149724",
                "pyranometer_(class_a)_pad_2_poa_irradiance_temp_compensated_(w/m2)_o_149726",
            ],
        },
        "environment": {
            "key": f"{BASE_KEY}/9068_environment_{csv_suffix}.csv",
            "usecols": [
                "measured_on",
                "weather_station_ambient_temperature_(c)_o_149727",
                "thermocouple_pad_1_back-of-module_temperature_1_(c)_o_149728",
                "thermocouple_pad_1_back-of-module_temperature_2_(c)_o_149729",
                "thermocouple_pad_2_back-of-module_temperature_1_(c)_o_149730",
                "thermocouple_pad_2_back-of-module_temperature_2_(c)_o_149731",
            ],
        },
    }

    # Adjust based on time period
    if csv_suffix == "data":  # 2017-2023 period
        # Keep meter column in ac_power (it exists in old files)
        # Remove the separate meter file (not needed for old period)
        files_config.pop("meter")
    else:  # 2024-2025 period
        # Remove meter column from ac_power (it doesn't exist in new files)
        if meter_ac_power in files_config["ac_power"]["usecols"]:
            files_config["ac_power"]["usecols"].remove(meter_ac_power)
        # Keep the separate meter file for new period

    return files_config


def read_filtered_csv_from_s3(
    BUCKET: str,
    key: str,
    usecols: list,
    label: str,
    START_DATE: str,
    END_DATE: str,
    chunksize=100_000,
    fallback=None,
) -> pd.DataFrame:
    """
    Read a public CSV file from S3, select specific columns, and filter by the defined time window.
    """

    print(f"\nReading file: {label}")
    print(f"S3 route: s3://{BUCKET}/{key}")

    # Anonymous S3 client to read the OEDI public bucket
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    obj = s3.get_object(Bucket=BUCKET, Key=key)

    chunks = []
    chunk_iter = pd.read_csv(obj["Body"], usecols=usecols, chunksize=chunksize)
    # No total known, but tqdm will show iteration count
    for chunk in tqdm(chunk_iter, desc=f"Reading {label}", unit="chunk"):
        chunk["measured_on"] = pd.to_datetime(chunk["measured_on"])
        mask = (chunk["measured_on"] >= pd.Timestamp(START_DATE)) & (
            chunk["measured_on"] < pd.Timestamp(END_DATE)
        )
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        if fallback is not None:
            print(f"  No data in {label}, using fallback...")
            return fallback(START_DATE, END_DATE)
        raise ValueError(
            f"No data found for label '{label}' between {START_DATE} and {END_DATE}."
        )

    df = pd.concat(chunks, ignore_index=True)

    print(f"{label}: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def ensure_local_file(s3_key: str, local_dir: Path = None) -> Path:
    """
    Download a file from public S3 bucket to local cache if not already present.

    Args:
        s3_key: Full S3 key (e.g., "pvdaq/.../9068_ac_power_data_20240101_20250430.csv")
        local_dir: Directory to store cached files. If None, uses '../dataset/noGit' relative to script.

    Returns:
        Local file path.
    """
    if local_dir is None:
        script_dir = Path(__file__).parent.resolve()
        local_dir = script_dir.parent / "dataset" / "noGit"
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / Path(s3_key).name
    if local_path.exists():
        print(f"Using cached file: {local_path}")
        return local_path

    print(f"Downloading {s3_key} to {local_path} ...")
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file(Bucket="oedi-data-lake", Key=s3_key, Filename=str(local_path))
    print("Download complete.")
    return local_path


def concatenate_dataframes(
    general_config: dict,
    files_config: dict,
    chunksize=100_000,
    fallbacks=None,
) -> list[pd.DataFrame]:
    """
    Concatenate dataframes from S3 by reading all configured files.

    Args:
        general_config: Dictionary with general configuration parameters
        files_config: Dictionary with file-specific configuration (keys, usecols)
        chunksize: Number of rows to read per chunk (default: 100,000)

    Returns:
        List of DataFrames, one for each file type (meter, ac_power, irradiance, environment)

    Raises:
        Various exceptions from read_filtered_csv_from_s3 if file reading fails
    """

    dataframes = []
    for label, cfg in files_config.items():
        fallback_fn = fallbacks.get(label) if fallbacks else None
        df_part = read_filtered_csv_from_s3(
            BUCKET=general_config["BUCKET"],
            key=cfg["key"],
            usecols=cfg["usecols"],
            label=label,
            START_DATE=general_config["START_DATE"],
            END_DATE=general_config["END_DATE"],
            chunksize=chunksize,
            fallback=fallback_fn,
        )
        dataframes.append(df_part)

    return dataframes


def meter_fallback_from_inverters(
    base_key: str,
    start_date: str,
    end_date: str,
    suffix: str = None,
) -> pd.DataFrame:
    """Build a meter DataFrame by summing inverter powers from the ac_power file."""
    if suffix is None:
        # Determine suffix from date range (reuse the same logic as get_files_config)
        if start_date >= "2024-01-01" and end_date <= "2025-05-01":
            suffix = "data_20240101_20250430"
        else:
            suffix = "data"

    ac_key = f"{base_key}/9068_ac_power_{suffix}.csv"
    local_file = ensure_local_file(ac_key)  # downloads to dataset/noGit if needed

    df = pd.read_csv(
        local_file,
        usecols=[
            "measured_on",
            "inverter_1_ac_power_(kw)_inv_150143",
            "inverter_2_ac_power_(kw)_inv_150144",
        ],
    )
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    mask = (df["measured_on"] >= pd.Timestamp(start_date)) & (
        df["measured_on"] < pd.Timestamp(end_date)
    )
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No ac_power data for {start_date}–{end_date}")
    df["meter_ac_power_(kw)_meter_150150"] = (
        df["inverter_1_ac_power_(kw)_inv_150143"]
        + df["inverter_2_ac_power_(kw)_inv_150144"]
    )
    return df[["measured_on", "meter_ac_power_(kw)_meter_150150"]]


def merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple dataframes by their timestamp column ('measured_on').

    Uses an inner join to keep only timestamps that exist in all dataframes.

    Args:
        dataframes: List of DataFrames to merge, each must have a 'measured_on' column

    Returns:
        Merged DataFrame with columns from all input DataFrames

    Raises:
        ValueError: If the dataframes list is empty
        RuntimeError: If the merge operation fails for any reason
    """
    print("\nMerging dataframes by measured_on...")

    if not dataframes:
        raise ValueError("No dataframes to merge.")

    try:
        df = reduce(
            lambda left, right: pd.merge(left, right, on="measured_on", how="inner"),
            dataframes,
        )
        print(f"Datasets are merged: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise RuntimeError(f"Error merging dataframes: {e}")


def rename_columns(df: pd.DataFrame, mapping=None) -> pd.DataFrame:
    """Rename columns of a DataFrame based on a mapping."""

    if mapping is None:
        mapping = {
            "measured_on": "timestamp",
            "meter_ac_power_(kw)_meter_150150": "plant_ac_power_kw",
            "inverter_1_ac_power_(kw)_inv_150143": "inverter_1_ac_power_kw",
            "inverter_2_ac_power_(kw)_inv_150144": "inverter_2_ac_power_kw",
            "pyranometer_(class_a)_pad_1_poa_irradiance_temp_compensated_(w/m2)_o_149724": "poa_irradiance_pad_1_wm2",
            "pyranometer_(class_a)_pad_2_poa_irradiance_temp_compensated_(w/m2)_o_149726": "poa_irradiance_pad_2_wm2",
            "weather_station_ambient_temperature_(c)_o_149727": "ambient_temperature_c",
            "thermocouple_pad_1_back-of-module_temperature_1_(c)_o_149728": "module_temperature_pad_1_1_c",
            "thermocouple_pad_1_back-of-module_temperature_2_(c)_o_149729": "module_temperature_pad_1_2_c",
            "thermocouple_pad_2_back-of-module_temperature_1_(c)_o_149730": "module_temperature_pad_2_1_c",
            "thermocouple_pad_2_back-of-module_temperature_2_(c)_o_149731": "module_temperature_pad_2_2_c",
        }

    # Check if all columns are present in the DataFrame
    missing = [col for col in mapping.keys() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Rename columns
    return df.rename(columns=mapping)


def apply_basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic data cleaning rules to the solar PV dataset.

    Cleaning steps:
    1. Clip negative power values to 0
    2. Clip negative irradiance values to 0
    3. Calculate mean irradiance from both pyranometers
    4. Remove out-of-range module temperatures (-50°C to 100°C)
    5. Calculate mean module temperature from all sensors

    Args:
        df: Input DataFrame with raw measurements

    Returns:
        Cleaned DataFrame with additional calculated columns

    Raises:
        ValueError: If required columns are missing
    """

    print("\nApplying basic cleaning rules...")

    # Required columns check
    required_power_cols = [
        "plant_ac_power_kw",
        "inverter_1_ac_power_kw",
        "inverter_2_ac_power_kw",
    ]
    required_irradiance_cols = ["poa_irradiance_pad_1_wm2", "poa_irradiance_pad_2_wm2"]
    required_module_temp_cols = [
        "module_temperature_pad_1_1_c",
        "module_temperature_pad_1_2_c",
        "module_temperature_pad_2_1_c",
        "module_temperature_pad_2_2_c",
    ]

    missing_cols = []
    for col in required_power_cols + required_irradiance_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing required columns for cleaning: {missing_cols}")

    # 1. Clip negative power values to 0
    print("  - Clipping negative power values to 0...")
    df["plant_ac_power_kw"] = df["plant_ac_power_kw"].clip(lower=0)
    df["inverter_1_ac_power_kw"] = df["inverter_1_ac_power_kw"].clip(lower=0)
    df["inverter_2_ac_power_kw"] = df["inverter_2_ac_power_kw"].clip(lower=0)

    # 2. Clip negative irradiance values to 0
    print("  - Clipping negative irradiance values to 0...")
    df["poa_irradiance_pad_1_wm2"] = df["poa_irradiance_pad_1_wm2"].clip(lower=0)
    df["poa_irradiance_pad_2_wm2"] = df["poa_irradiance_pad_2_wm2"].clip(lower=0)

    # 3. Calculate mean irradiance
    print("  - Calculating mean irradiance...")
    df["poa_irradiance_wm2"] = df[
        ["poa_irradiance_pad_1_wm2", "poa_irradiance_pad_2_wm2"]
    ].mean(axis=1)

    # 4. Clean module temperatures (remove out-of-range values)
    print("  - Cleaning module temperatures (-50°C to 100°C range)...")
    module_temp_cols = required_module_temp_cols

    # Check which temperature columns actually exist
    existing_temp_cols = [col for col in module_temp_cols if col in df.columns]
    if not existing_temp_cols:
        print(
            "  Warning: No module temperature columns found, skipping temperature cleaning"
        )
    else:
        for col in existing_temp_cols:
            out_of_range = (df[col] < -50) | (df[col] > 100)
            if out_of_range.any():
                print(
                    f"    - Replacing {out_of_range.sum():,} out-of-range values in {col}"
                )
                df.loc[out_of_range, col] = pd.NA

    # 5. Calculate mean module temperature
    if existing_temp_cols:
        print("  - Calculating mean module temperature...")
        df["module_temperature_c"] = df[existing_temp_cols].mean(axis=1)
    else:
        print("  Warning: No module temperature columns available for mean calculation")
        df["module_temperature_c"] = pd.NA

    print("Basic cleaning completed.")

    return df


def add_temporal_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Add temporal features (hour, dayofyear, month) from timestamp column for modeling.

    Temporal features created:
    - hour: Decimal hour (e.g., 14.5 for 2:30 PM) - useful for capturing daily patterns
    - dayofyear: Day number from 1 to 365/366 - captures seasonal patterns
    - month: Month number from 1 to 12 - captures monthly seasonality

    Args:
        df: Input DataFrame with timestamp column
        timestamp_col: Name of the timestamp column (default: "timestamp")

    Returns:
        DataFrame with added temporal feature columns

    Raises:
        ValueError: If timestamp column is missing or not datetime type
    """

    print("\nAdding temporal features for modeling...")

    # Check if timestamp column exists
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")

    # Check if timestamp is datetime type, convert if needed
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        print(f"  Converting '{timestamp_col}' to datetime type...")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Calculate temporal features
    print("  - Calculating decimal hour (for daily patterns)...")
    df["hour"] = df[timestamp_col].dt.hour + df[timestamp_col].dt.minute / 60

    print("  - Calculating day of year (for seasonal patterns)...")
    df["dayofyear"] = df[timestamp_col].dt.dayofyear

    print("  - Calculating month (for monthly patterns)...")
    df["month"] = df[timestamp_col].dt.month

    # Optional: Add more temporal features if needed (commented out)
    # df["dayofweek"] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
    # df["weekend"] = (df[timestamp_col].dt.dayofweek >= 5).astype(int)  # 1 for weekend
    # df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)  # Cyclical encoding
    # df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    print(f"Temporal features added: hour, dayofyear, month")

    return df


def select_final_columns(df: pd.DataFrame, selected_cols: list = None) -> pd.DataFrame:
    """
    Select and organize final columns for the curated dataset.

    Args:
        df: Input DataFrame with all processed columns
        selected_cols: List of columns to keep. If None, uses default selection.

    Returns:
        DataFrame with only the selected columns

    Raises:
        ValueError: If any selected column is missing from the DataFrame
    """

    print("\nSelecting final columns for curated dataset...")

    # Default column selection if none provided
    if selected_cols is None:
        selected_cols = [
            "timestamp",
            "plant_ac_power_kw",
            "inverter_1_ac_power_kw",
            "inverter_2_ac_power_kw",
            "poa_irradiance_wm2",
            "ambient_temperature_c",
            "module_temperature_c",
            "hour",
            "dayofyear",
            "month",
        ]

    # Check if all selected columns exist
    missing_cols = [col for col in selected_cols if col not in df.columns]

    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
        available_cols = [col for col in selected_cols if col in df.columns]
        if not available_cols:
            raise ValueError(
                f"None of the selected columns found in DataFrame. Available columns: {list(df.columns)}"
            )

        print(f"  Using only available columns: {available_cols}")
        selected_cols = available_cols

    # Create final DataFrame with selected columns
    df_final = df[selected_cols].copy()

    print(f"  Selected {df_final.shape[1]} columns: {list(df_final.columns)}")
    print(
        f"  Final dataset shape: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns"
    )

    return df_final


def create_training_dataset(
    df: pd.DataFrame, min_irradiance_wm2: float = 50
) -> pd.DataFrame:
    """
    Create training dataset with daytime periods and complete data.

    Args:
        df: Input DataFrame (curated full dataset)
        min_irradiance_wm2: Minimum irradiance threshold in W/m²

    Returns:
        Training dataset filtered for daytime with non-null values
    """

    print(f"\nCreating training dataset (irradiance > {min_irradiance_wm2} W/m²)...")

    # Required columns check
    required_cols = [
        "poa_irradiance_wm2",
        "plant_ac_power_kw",
        "ambient_temperature_c",
        "module_temperature_c",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Apply filters
    df_training = df[
        (df["poa_irradiance_wm2"] > min_irradiance_wm2)
        & (df["plant_ac_power_kw"].notna())
        & (df["ambient_temperature_c"].notna())
        & (df["module_temperature_c"].notna())
    ].copy()

    print(
        f"  Training dataset: {df_training.shape[0]:,} rows, {df_training.shape[1]} columns"
    )
    print(
        f"  Filtered out {df.shape[0] - df_training.shape[0]:,} rows "
        f"({(1 - df_training.shape[0]/df.shape[0])*100:.1f}% of data)"
    )

    return df_training


def print_dataset_summary(
    df_curated_all_cols: pd.DataFrame,
    df_curated_full: pd.DataFrame,
    df_training: pd.DataFrame,
    curated_all_cols_path: Path = None,
    curated_full_path: Path = None,
    training_path: Path = None,
) -> None:
    """
    Print comprehensive summary statistics for all three datasets.

    Args:
        df_curated_all_cols: Curated dataset with all columns (before column selection)
        df_curated_full: Curated dataset with only selected columns
        df_training: Training dataset (filtered for daytime with complete data)
        curated_all_cols_path: Path where curated_all_cols dataset was saved (optional)
        curated_full_path: Path where curated_full dataset was saved (optional)
        training_path: Path where training dataset was saved (optional)
    """

    print("\n" + "=" * 60)
    print("STEP 3 RESULTS - DATASETS SUMMARY")
    print("=" * 60)

    # Dataset 1: Curated All Columns
    print("\n" + "-" * 50)
    print("1. CURATED DATASET (ALL COLUMNS)")
    print("-" * 50)
    if curated_all_cols_path:
        print(f"Path: {curated_all_cols_path}")
    print(f"Rows: {df_curated_all_cols.shape[0]:,}")
    print(f"Columns: {df_curated_all_cols.shape[1]}")
    print(df_curated_all_cols.info())

    # Dataset 2: Curated Full (Selected Columns)
    print("\n" + "-" * 50)
    print("2. CURATED DATASET (SELECTED COLUMNS)")
    print("-" * 50)
    if curated_full_path:
        print(f"Path: {curated_full_path}")
    print(f"Rows: {df_curated_full.shape[0]:,}")
    print(f"Columns: {df_curated_full.shape[1]}")
    print(df_curated_full.info())

    # Dataset 3: Training Dataset
    print("\n" + "-" * 50)
    print("3. TRAINING DATASET (DAYTIME + COMPLETE DATA)")
    print("-" * 50)
    if training_path:
        print(f"Path: {training_path}")
    print(f"Rows: {df_training.shape[0]:,}")
    print(f"Columns: {df_training.shape[1]}")
    print(df_training.info())

    # Show first rows of curated_full dataset
    print("\n" + "-" * 50)
    print("PREVIEW OF CURATED FULL DATASET (first 5 rows)")
    print("-" * 50)
    print(df_curated_full.head())

    # Null values summary for curated_full dataset
    print("\n" + "-" * 50)
    print("NULL VALUES SUMMARY (curated full dataset)")
    print("-" * 50)
    null_counts = df_curated_full.isna().sum()
    if null_counts.any():
        print(null_counts[null_counts > 0])
    else:
        print("No null values found")

    # Statistics for curated_full dataset
    print("\n" + "-" * 50)
    print("STATISTICS - CURATED FULL DATASET")
    print("-" * 50)
    print(df_curated_full.describe())

    # Statistics for training dataset
    print("\n" + "-" * 50)
    print("STATISTICS - TRAINING DATASET")
    print("-" * 50)
    print(df_training.describe())

    # Summary of data reduction
    print("\n" + "-" * 50)
    print("DATA REDUCTION SUMMARY")
    print("-" * 50)
    print(f"Curated full dataset: {df_curated_full.shape[0]:,} rows")
    print(f"Training dataset:     {df_training.shape[0]:,} rows")
    reduction_pct = (1 - df_training.shape[0] / df_curated_full.shape[0]) * 100
    print(f"Training dataset filters out {reduction_pct:.1f}% of rows")

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETED SUCCESSFULLY")
    print("=" * 60)


def list_available_files(BUCKET, key):
    """List all CSV files in the bucket for system 9068."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=key)

    print(f"\nFiles in s3://{BUCKET}/{key}")
    print("-" * 60)
    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".csv"):
                print(key)


def validate_files_exist(BUCKET: str, files_config: dict) -> bool:
    """Check if all required files exist in S3 before processing."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    missing_files = []

    for label, cfg in files_config.items():
        key = cfg["key"]
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            print(f"✓ {label:20s} file exists:  {key}")
        except Exception:
            missing_files.append(key)
            print(f"✗ {label:20s} file MISSING: {key}")

    if missing_files:
        print("\nMissing files:")
        for f in missing_files:
            print(f"  - {f}")

        # Show available files in the directory to help debug
        prefix = "/".join(missing_files[0].split("/")[:-1]) + "/"
        print(f"\nListing available files in: s3://{BUCKET}/{prefix}")
        list_available_files(BUCKET, prefix)
        return False

    return True


def build_pvdt_dataset(
    start_date="2025-01-01",
    end_date="2025-02-01",
    csv_prefix="pvdt_9068_2025_01",
    save_to_csv=False,
    DBG=False,
):
    """
    Main function to build the PVDT dataset for Step 3.

    Args:
        start_date: Start date for data filtering (default: "2025-01-01")
        end_date: End date for data filtering (default: "2025-02-01")
        csv_prefix: Prefix for CSV file names (default: "pvdt_9068_2025_01")
        save_to_csv: If True, saves the datasets to CSV files (default: False)
        DBG: If True, prints debug information including head() and info() (default: False)

    Returns:
        Tuple of three DataFrames:
        - df_curated_all_cols: Curated dataset with all columns (before column selection)
        - df_curated_full: Curated dataset with only selected columns
        - df_training: Training dataset filtered for daytime with complete data

    Example:
        >>> # Run with saving and debug output
        >>> all_cols, full, training = build_pvdt_dataset(save_to_csv=True, DBG=True)

        >>> # Run without saving, just return dataframes
        >>> all_cols, full, training = build_pvdt_dataset()
    """

    general_config = get_general_config(START_DATE=start_date, END_DATE=end_date)
    files_config = get_files_config(
        BASE_KEY=general_config["BASE_KEY"],
        START_DATE=general_config["START_DATE"],
        END_DATE=general_config["END_DATE"],
        csv_suffix="",
    )

    # Determine suffix based on date range (instead of extracting from ac_key)
    suffix = get_csv_suffix(start_date, end_date)
    if start_date >= "2024-01-01" and end_date <= "2025-05-01":
        suffix = "data_20240101_20250430"
    else:
        suffix = "data"

    # Define fallback for meter (only for 2024‑2025 period, because old period always has meter data)
    def meter_fallback(start, end):
        return meter_fallback_from_inverters(
            base_key=general_config["BASE_KEY"],
            start_date=start,
            end_date=end,
            suffix=suffix,  # pass the correct suffix
        )

    fallbacks = {}
    # Only use fallback for the period where meter file might be missing
    if suffix != "data":  # i.e., 2024‑2025
        fallbacks["meter"] = meter_fallback

    # Now read all dataframes with fallback support
    dataframes = concatenate_dataframes(
        general_config=general_config,
        files_config=files_config,
        chunksize=100_000,
        fallbacks=fallbacks,
    )

    # Merge dfs & rename cols
    df = merge_dataframes(dataframes)
    df = rename_columns(df)

    # Apply basic cleaning
    df = apply_basic_cleaning(df)

    # Add temporal features for modeling
    df = add_temporal_features(df, timestamp_col="timestamp")

    # Save whole curated dataset (not to confuse with 'curated full' dataset)
    df_curated_all_cols = df.copy()

    # Select final columns for curated dataset
    df_curated_full = select_final_columns(df)

    # Create training dataset (filtered for daytime with complete data)
    df_training = create_training_dataset(
        df_curated_full,
        min_irradiance_wm2=50,  # Only keep irradiance > 50 W/m²
    )

    output_curated_all_cols = None
    output_curated_full = None
    output_training = None

    if save_to_csv:
        # Save all three datasets to CSV
        output_curated_all_cols = (
            general_config["OUTPUT_DIR"] / f"{csv_prefix}_curated_all_cols.csv"
        )
        output_curated_full = (
            general_config["OUTPUT_DIR"] / f"{csv_prefix}_curated_full.csv"
        )
        output_training = general_config["OUTPUT_DIR"] / f"{csv_prefix}_training.csv"

        df_curated_all_cols.to_csv(output_curated_all_cols, index=False)
        df_curated_full.to_csv(output_curated_full, index=False)
        df_training.to_csv(output_training, index=False)

    if DBG:
        # Print summary
        print_dataset_summary(
            df_curated_all_cols,
            df_curated_full,
            df_training,
            output_curated_all_cols,
            output_curated_full,
            output_training,
        )

    return df_curated_all_cols, df_curated_full, df_training


def main_2025_01():
    build_pvdt_dataset(
        start_date="2025-01-01",
        end_date="2025-02-01",
        csv_prefix="pvdt_9068_2025_01",
        save_to_csv=True,
        DBG=True,
    )


def main_2024_03():
    build_pvdt_dataset(
        start_date="2024-03-01",
        end_date="2024-04-01",
        csv_prefix="pvdt_9068_2024_03",
        save_to_csv=True,
        DBG=True,
    )


def main_2023_01():
    build_pvdt_dataset(
        start_date="2023-01-01",
        end_date="2023-02-01",
        csv_prefix="pvdt_9068_2023_01",
        save_to_csv=True,
        DBG=True,
    )


def main_2023_12():
    """
    Special month with no data

    Output:

    Blank data from 2023-11-16 to 2024-01-01
    ✓ ac_power             file exists:  pvdaq/2023-solar-data-prize/9068_OEDI/data/9068_ac_power_data.csv
    ✓ irradiance           file exists:  pvdaq/2023-solar-data-prize/9068_OEDI/data/9068_irradiance_data.csv
    ✓ environment          file exists:  pvdaq/2023-solar-data-prize/9068_OEDI/data/9068_environment_data.csv

    Reading file: ac_power
    S3 route: s3://oedi-data-lake/pvdaq/2023-solar-data-prize/9068_OEDI/data/9068_ac_power_data.csv
    Reading ac_power: 7chunk [00:40,  5.79s/chunk]
    Error: No data was found for label 'ac_power' between 2023-12-01 and 2024-01-01.
    """
    try:
        build_pvdt_dataset(
            start_date="2023-12-01",
            end_date="2024-01-01",
            csv_prefix="pvdt_9068_2024_12",
            save_to_csv=True,
            DBG=True,
        )
    except Exception as e:
        print(f"Error: {e}")


from datetime import datetime, timedelta


def process_all_months(save_to_csv=True, DBG=False):
    """
    Process all months from 2017-01-01 to 2025-04-30.

    Args:
        save_to_csv: If True, saves the datasets to CSV files
        DBG: If True, prints debug information

    Output:
    Total months processed: 100
    Successful: 90
    Failed: 10

    Failed months:
      - pvdt_9068_2017_01
      - pvdt_9068_2017_02
      - pvdt_9068_2017_03
      - pvdt_9068_2017_04
      - pvdt_9068_2017_05
      - pvdt_9068_2017_06
      - pvdt_9068_2017_07
      - pvdt_9068_2017_08
      - pvdt_9068_2023_12
      - pvdt_9068_2024_03
    """

    # Define the date range
    start_date_total = datetime(2017, 1, 1)
    end_date_total = datetime(2025, 4, 30)

    # Start with the first month
    current_date = start_date_total

    print("\n" + "=" * 60)
    print("PROCESSING ALL MONTHS FROM 2017-01-01 TO 2025-04-30")
    print("=" * 60)

    month_count = 0
    successful_months = 0
    failed_months = []

    while current_date <= end_date_total:
        # Calculate start and end dates for this month
        start_date = current_date.strftime("%Y-%m-%d")

        # Calculate end date as first day of next month
        if current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1)

        # Stop if next_month exceeds our total range
        if next_month > end_date_total + timedelta(days=1):
            break

        end_date_str = next_month.strftime("%Y-%m-%d")

        # Create csv_prefix for this month
        csv_prefix = f"pvdt_9068_{current_date.year}_{current_date.month:02d}"

        # Print progress
        print(f"\n{'='*50}")
        print(f"Processing: {csv_prefix}")
        print(f"Date range: {start_date} to {end_date_str}")
        print(f"{'='*50}")

        try:
            # Call the build function for this month
            build_pvdt_dataset(
                start_date=start_date,
                end_date=end_date_str,
                csv_prefix=csv_prefix,
                save_to_csv=save_to_csv,
                DBG=DBG,
            )
            successful_months += 1
            print(f"✓ Successfully processed {csv_prefix}")

        except Exception as e:
            print(f"✗ Failed to process {csv_prefix}: {e}")
            failed_months.append(csv_prefix)

        month_count += 1

        # Move to next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total months processed: {month_count}")
    print(f"Successful: {successful_months}")
    print(f"Failed: {len(failed_months)}")

    if failed_months:
        print("\nFailed months:")
        for month in failed_months:
            print(f"  - {month}")

    return successful_months, failed_months


if __name__ == "__main__":

    # Example month < 2024
    if False:
        main_2023_01()

    # Example month > 2024
    if False:
        main_2025_01()

    # Special month with no data
    if True:
        main_2023_12()

    # Special month
    #   Reading file: meter
    #   S3 route: s3://oedi-data-lake/pvdaq/2023-solar-data-prize/9068_OEDI/data/9068_meter_data_20240101_20250430.csv
    #   Reading meter: 2chunk [00:04,  2.05s/chunk]
    #     No data in meter, using fallback...
    #   Using cached file: dataset/noGit/9068_ac_power_data_20240101_20250430.csv
    if False:
        main_2024_03()

    # Process all months from 2017-01-01 to 2025-04-30
    if False:
        process_all_months(
            save_to_csv=True, DBG=False
        )  # Set DBG=False to avoid too much output
