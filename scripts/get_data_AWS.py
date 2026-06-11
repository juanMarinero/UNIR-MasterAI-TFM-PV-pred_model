#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=4 foldmethod=indent :

import numpy as np
import pandas as pd

TARGET = "total_inverter_ac_power_kw"
MIN_POA_WM2 = 50


def load_model(model_path="models/random_forest_model.pkl", dbg=True):

    import pickle

    # Cargar el modelo
    with open(model_path, "rb") as file:
        loaded_model = pickle.load(file)

    if dbg:
        print("Modelo cargado exitosamente")

        # Verificar que funciona
        print(f"Tipo de modelo: {type(loaded_model)}")
        print(f"Features esperadas: {loaded_model.feature_names_in_}")

    return loaded_model


def get_data_AWS(
    dataset="dataset/pvdt_9068_2025_04_curated_all_cols.csv",
    model_path="models/random_forest_model.pkl",
    plot_bool=False,
    row_start=0,
    row_end=None,
):

    # Load data
    df = pd.read_csv(dataset)

    # Imports
    from scripts.proccess_previous_to_model import (
        proccess_previous_to_model_fit,
        get_df_for_model,
        get_power_per_irradiance,
        get_module_vs_ambient_diff_c,
    )

    # Preprocess
    df = proccess_previous_to_model_fit(df)

    # Filter rows
    if row_end and row_end <= len(df):
        df = df[int(row_start) : int(row_end)]

    # Filter columns
    df_to_fit = get_df_for_model(df)

    # Get predictions
    y_true = df_to_fit[TARGET]
    n_nans = y_true.isna().sum()
    if n_nans > 0:
        print(
            f"Se encontraron {n_nans} valores nulos en la columna {TARGET}. Se reemplazan con 0.0."
        )
        y_true = y_true.fillna(0.0)  # Reemplazar NaN con 0.0

    # Get predictions
    model = load_model(model_path, dbg=False)
    df_to_fit.drop(columns=TARGET, inplace=True)
    df_to_fit = df_to_fit[model.feature_names_in_]
    y_pred = model.predict(df_to_fit)

    # Clip negative predictions
    y_pred = np.clip(y_pred, 0, None)

    # Clip if low irradiance
    mask_irradiance = df["poa_irradiance_wm2"] < MIN_POA_WM2
    y_pred[mask_irradiance] = 0.0

    # Dashboard fields
    df = get_power_per_irradiance(df)
    df = get_module_vs_ambient_diff_c(df)

    # Plot
    MAE, MAPE = np.nan, np.nan
    # Get timestamps start and end
    timestamps_range = [df["timestamp"].iloc[0], df["timestamp"].iloc[-1]]
    if plot_bool:
        from scripts.utils import plot_df_pred_true

        fig, ax1, *_, MAE, MAPE, timestamps_range = plot_df_pred_true(
            df, y_true, y_pred, all_points_bool=True
        )
        # ax1.set_title(dataset)

    return df, y_true, y_pred, MAE, MAPE, timestamps_range
