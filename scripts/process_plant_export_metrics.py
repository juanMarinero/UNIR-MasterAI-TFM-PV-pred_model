#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=4 foldmethod=indent :

"""
Production note:
This scripts in what build_pvdt_dataset_step3_v2.py achieves that was not done in build_pvdt_dataset_step3.py
"""

import numpy as np
import pandas as pd


def process_plant_export_metrics(
    df,
    target="total_inverter_ac_power_kw",
    min_poa_wm2=0,
    max_power_kw=None,
    min_ambient_temp_c=None,
    max_ambient_temp_c=None,
):
    """Process plant export metrics by comparing meter readings with inverter power.

    Args:
        df: DataFrame containing columns: 'inverter_1_ac_power_kw',
            'inverter_2_ac_power_kw', 'plant_ac_power_kw', 'poa_irradiance_wm2',
            'ambient_temperature_c'
        target: Target column name for total inverter power (default: "total_inverter_ac_power_kw")
        min_poa_wm2: Minimum POA irradiance threshold (default: 0)
        max_power_kw: Maximum power threshold (default: None = no filter)
        min_ambient_temp_c: Minimum ambient temperature threshold (default: None = no filter)
        max_ambient_temp_c: Maximum ambient temperature threshold (default: None = no filter)

    Returns:
        DataFrame with added columns for inverter totals, export power, and differences,
        filtered according to specified criteria

    Raises:
        KeyError: If any required columns are missing
    """

    # Required columns
    required_columns = [
        "inverter_1_ac_power_kw",
        "inverter_2_ac_power_kw",
        "plant_ac_power_kw",
        "poa_irradiance_wm2",
        "ambient_temperature_c",
    ]

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {', '.join(missing_columns)}. "
            f"Available columns: {list(df.columns)}"
        )

    # Potencia total de inversores
    df["total_inverter_ac_power_kw"] = (
        df["inverter_1_ac_power_kw"] + df["inverter_2_ac_power_kw"]
    )

    # El contador usa signo negativo para exportación
    # Se conserva la señal original y se crea una variable positiva de potencia exportada
    df["plant_export_power_kw"] = (-df["plant_ac_power_kw"]).clip(lower=0)

    # Diferencia entre potencia exportada de contador y potencia total de inversores
    df["meter_vs_inverter_diff_kw"] = (
        df["plant_export_power_kw"] - df["total_inverter_ac_power_kw"]
    )

    df["meter_vs_inverter_diff_pct"] = np.where(
        df["total_inverter_ac_power_kw"] == 0,
        0,
        (df["meter_vs_inverter_diff_kw"] / df["total_inverter_ac_power_kw"]) * 100,
    )

    # Apply filtering criteria
    filter_conditions = (
        (df["poa_irradiance_wm2"].notna())
        & (df["poa_irradiance_wm2"] >= min_poa_wm2)
        & (df[target].notna())
        & (df[target] >= 0)
    )

    # Add max power filter if specified
    if max_power_kw is not None:
        filter_conditions &= df[target] <= max_power_kw

    # Add temperature filters if specified
    filter_conditions &= df["ambient_temperature_c"].notna()

    if min_ambient_temp_c is not None:
        filter_conditions &= df["ambient_temperature_c"] >= min_ambient_temp_c

    if max_ambient_temp_c is not None:
        filter_conditions &= df["ambient_temperature_c"] <= max_ambient_temp_c

    # Apply all filters
    df_filtered = df[filter_conditions].copy()

    return df_filtered


if __name__ == "__main__":
    process_plant_export_metrics(df)
