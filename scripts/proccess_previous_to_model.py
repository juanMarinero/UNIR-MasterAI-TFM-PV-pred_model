#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=4 foldmethod=indent :

import numpy as np
import pandas as pd

TARGET = "total_inverter_ac_power_kw"

RANDOM_STATE = 42

# Capacidad aproximada DC de la planta PVDAQ 9068.
# No se usa como variable del modelo, solo como control físico razonable.
PLANT_DC_CAPACITY_KW = 4738.0

# Umbrales físicos de limpieza.
MIN_POA_WM2 = 50
MAX_POWER_KW = 5500
MIN_AMBIENT_TEMP_C = -40
MAX_AMBIENT_TEMP_C = 60
MIN_MODULE_TEMP_C = -40
MAX_MODULE_TEMP_C = 95


def filter_max_power_and_ambient_temp_range(df):
    return df[
        (df[TARGET] <= MAX_POWER_KW)
        & (df["ambient_temperature_c"] >= MIN_AMBIENT_TEMP_C)
        & (df["ambient_temperature_c"] <= MAX_AMBIENT_TEMP_C)
        & (df["module_temperature_c"] >= MIN_MODULE_TEMP_C)
        & (df["module_temperature_c"] <= MAX_MODULE_TEMP_C)
    ]


def get_power_per_irradiance(df):
    # For dashboard ONLY
    # Variable diagnóstica, no necesariamente feature principal.
    df["power_per_irradiance"] = np.where(
        df["poa_irradiance_wm2"] == 0, 0, (df[TARGET] / df["poa_irradiance_wm2"])
    )
    return df


def get_module_vs_ambient_diff_c(df):
    # For dashboard ONLY
    # Variable diagnóstica, no necesariamente feature principal.
    # Diferencia térmica ambient and module
    df["module_vs_ambient_diff_c"] = (
        df["module_temperature_c"] - df["ambient_temperature_c"]
    )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade variables temporales cíclicas para capturar patrones diarios y estacionales.
    """
    df = df.copy()

    # Convert timestamp column if it exists and isn't already datetime
    if "timestamp" in df.columns and df["timestamp"].dtype != "datetime64[ns]":
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2024 es año bisiesto.
    # 00:00 hours close to 23:59 hours
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # doy stands for Date Of Year
    # 31st Dec close to 1st Jan
    df["doy_sin"] = np.sin(
        2
        * np.pi
        * (df["dayofyear"] - 1)
        / df["timestamp"].dt.is_leap_year.apply(lambda x: 366 if x else 365)
    )
    df["doy_cos"] = np.cos(
        2
        * np.pi
        * (df["dayofyear"] - 1)
        / df["timestamp"].dt.is_leap_year.apply(lambda x: 366 if x else 365)
    )

    return df


def get_df_for_model(df):
    # Target: 'total_inverter_ac_power_kw'
    keep_cols = [
        TARGET,
        "poa_irradiance_wm2",
        "ambient_temperature_c",
        "module_temperature_c",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    return df[keep_cols]


def month_stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Split train/test estratificado por mes.

    La idea es que tanto train como test tengan ejemplos de todas las estaciones,
    evitando que el test quede concentrado solo en invierno o verano.
    """
    test_parts = []

    for month, group in df.groupby("month"):
        n_test = max(1, int(len(group) * test_size))
        test_parts.append(group.sample(n=n_test, random_state=random_state))

    test_df = pd.concat(test_parts).sort_values("timestamp").reset_index(drop=True)
    train_df = df.drop(index=test_df.index, errors="ignore")

    # El drop por índice puede fallar si hemos reseteado. Hacemos forma robusta por timestamp.
    test_keys = set(test_df["timestamp"].astype(str))
    train_df = df[~df["timestamp"].astype(str).isin(test_keys)].copy()

    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    return train_df, test_df


def save_dataset_bundle(
    train_df, test_df, prefix: str, df: pd.DataFrame, features: list
):
    """
    Guarda dataset completo, train, test y metadatos de features.
    """
    full_path = OUTPUT_DIR / f"{prefix}_full.csv"
    train_path = OUTPUT_DIR / f"{prefix}_train.csv"
    test_path = OUTPUT_DIR / f"{prefix}_test.csv"
    metadata_path = OUTPUT_DIR / f"{prefix}_metadata.json"

    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "target": TARGET,
        "features": features,
        "rows_full": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "random_state": RANDOM_STATE,
        "split_strategy": "month_stratified_80_20",
        "filters": {
            "poa_irradiance_wm2_min": MIN_POA_WM2,
            TARGET + "_min": 0,
            TARGET + "_max": MAX_POWER_KW,
            "ambient_temperature_c_min": MIN_AMBIENT_TEMP_C,
            "ambient_temperature_c_max": MAX_AMBIENT_TEMP_C,
            "module_temperature_c_min": MIN_MODULE_TEMP_C,
            "module_temperature_c_max": MAX_MODULE_TEMP_C,
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nArchivos generados:")
    print(full_path)
    print(train_path)
    print(test_path)
    print(metadata_path)

    return train_df, test_df


def proccess_previous_to_model_fit(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_time_features(df)

    df[TARGET] = df["inverter_1_ac_power_kw"] + df["inverter_2_ac_power_kw"]

    # Drop rows with NaN
    df.dropna(subset=[TARGET], inplace=True)

    return df


def proccess_previous_to_model(df, scaler=None):

    # str to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = add_time_features(df)

    df = filter_max_power_and_ambient_temp_range(df)

    df = get_power_per_irradiance(df)

    df = get_module_vs_ambient_diff_c(df)

    # Escalar todas las columnas excepto "timestamp"
    if scaler is not None:
        df, scaler = apply_scaler(df, scaler)

    train_df, test_df = get_train_test_dfs(df)

    return df, train_df, test_df


def apply_scaler(df, scaler, columns_to_scale=None):
    if scaler is not None:
        if columns_to_scale is None:
            columns_to_scale = df.columns
        # Filter columns
        columns_to_scale = [
            col for col in columns_to_scale if col != "timestamp" and col != TARGET
        ]

        if hasattr(scaler, "fit_transform") and not hasattr(scaler, "mean_"):
            # 1st time: fit & transform
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        else:
            # If already trained: transform
            df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    else:
        print("No scaler passed!")
    return df, scaler


def get_train_test_dfs(df, test_size=0.2):
    train_df, test_df = month_stratified_split(
        df=df, test_size=0.2, random_state=RANDOM_STATE
    )

    train_df = get_df_for_model(train_df)
    test_df = get_df_for_model(test_df)

    return train_df, test_df
