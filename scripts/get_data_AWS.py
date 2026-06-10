#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=6 foldmethod=indent :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_model(model_path="models/random_forest_model.pkl"):

    import pickle

    # Cargar el modelo
    with open(model_path, "rb") as file:
        loaded_model = pickle.load(file)

    print("Modelo cargado exitosamente")

    # Verificar que funciona
    print(f"Tipo de modelo: {type(loaded_model)}")
    print(f"Features esperadas: {loaded_model.feature_names_in_}")

    return loaded_model


TARGET = "total_inverter_ac_power_kw"


def get_data_AWS(
    dataset="dataset/pvdt_9068_2025_04_curated_all_cols.csv", plot_bool=False
):

    model = load_model()

    df = pd.read_csv(dataset)

    from scripts.proccess_previous_to_model import (
        proccess_previous_to_model_fit,
        get_df_for_model,
        get_power_per_irradiance,
        get_module_vs_ambient_diff_c,
    )

    df = proccess_previous_to_model_fit(df)
    df_to_fit = get_df_for_model(df)

    y_true = df_to_fit[TARGET]
    df_to_fit.drop(columns=TARGET, inplace=True)
    df_to_fit = df_to_fit[model.feature_names_in_]
    y_pred = model.predict(df_to_fit)

    # Dashboard adicionales
    df = get_power_per_irradiance(df)
    df = get_module_vs_ambient_diff_c(df)

    # Plot
    fig, ax = plot_df_pred_true(df, y_true, y_pred, n_points_all=True)
    ax.set_title(dataset)

    return df, y_true, y_pred


def plot_df_pred_true(df, y_true, y_pred, n_points_all=False):

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Usar timestamps como x
    timestamps = df["timestamp"] if "timestamp" in df.columns else df.index
    x_vals = timestamps if isinstance(timestamps, pd.Series) else np.arange(len(y_true))

    # Calcular errores
    errors = np.abs(y_true - y_pred)
    error_percentage = (errors / (y_true + 1e-6)) * 100

    # Muestreo
    n_points = len(y_true)
    if not n_points_all:
        sample_rate = max(1, n_points // 500)
    else:
        sample_rate = 1

    x_sample = x_vals[::sample_rate]
    y_true_sample = y_true[::sample_rate]
    y_pred_sample = y_pred[::sample_rate]
    errors_sample = errors[::sample_rate]
    error_percentage_sample = error_percentage[::sample_rate]

    # Eje primario: Valores reales y predicciones CON barras de error
    ax1.errorbar(
        x_sample,
        y_pred_sample,
        yerr=errors_sample,
        fmt="o",
        capsize=2,
        alpha=0.6,
        markersize=3,
        label="Predicción ± error",
        color="blue",
        ecolor="lightgray",
    )

    ax1.plot(
        x_sample,
        y_true_sample,
        "r-",
        alpha=0.7,
        linewidth=1.5,
        label="Valor Real",
        marker="o",
        markersize=2,
    )

    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Potencia (kW)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Eje secundario: Error porcentual
    ax2 = ax1.twinx()
    ax2.plot(
        x_sample,
        error_percentage_sample,
        "g-",
        alpha=0.6,
        linewidth=1,
        marker="^",
        markersize=2,
        label="Error %",
    )
    ax2.set_ylabel("Error Porcentual (%)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="upper right")

    # Línea de referencia en 0% para el error
    ax2.axhline(y=0, color="green", linestyle="--", alpha=0.3)

    ax1.set_title(
        "Predicciones vs Valores Reales con Error Porcentual en Eje Secundario"
    )

    if "timestamp" in str(type(x_vals)):
        plt.xticks(rotation=45)

    plt.tight_layout()

    # Estadísticas del error
    print(f"Error absoluto medio: {np.mean(errors):.3f} kW")
    print(f"Error porcentual medio: {np.mean(error_percentage):.2f}%")
    print(f"Desviación estándar del error: {np.std(errors):.3f} kW")

    return fig, ax1
