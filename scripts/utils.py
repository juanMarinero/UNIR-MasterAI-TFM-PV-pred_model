#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=4 foldmethod=indent :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PLANT_DC_CAPACITY_KW = 4738.0
TARGET = "total_inverter_ac_power_kw"
MIN_POA_WM2 = 50


def evaluate_model(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    # Error porcentual solo cuando la potencia real es suficientemente alta.
    mask = y_true > 100
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    metrics = {
        "mae_kw": float(mae),
        "rmse_kw": float(rmse),
        "r2": float(r2),
        "mape_pct_when_actual_gt_100kw": float(mape),
        "mae_pct_of_dc_capacity": float(mae / PLANT_DC_CAPACITY_KW * 100),
        "rmse_pct_of_dc_capacity": float(rmse / PLANT_DC_CAPACITY_KW * 100),
        "median_abs_error_kw": float(np.median(abs_errors)),
        "p90_abs_error_kw": float(np.percentile(abs_errors, 90)),
        "p95_abs_error_kw": float(np.percentile(abs_errors, 95)),
        "p99_abs_error_kw": float(np.percentile(abs_errors, 99)),
    }

    return metrics


def plot_compare_test_pred(y_test, y_pred):

    # Ordenar por valores reales para una mejor visualización
    sorted_indices = np.argsort(y_test)
    y_test_sorted = (
        y_test.iloc[sorted_indices]
        if hasattr(y_test, "iloc")
        else y_test[sorted_indices]
    )
    y_pred_sorted = y_pred[sorted_indices]

    # Calcular errores
    errors = y_test_sorted - y_pred_sorted
    absolute_errors = np.abs(errors)
    percentage_errors = (
        absolute_errors / (y_test_sorted + 1e-6)
    ) * 100  # Evitar división por cero

    # Crear figura con 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Análisis de Predicciones vs Valores Reales", fontsize=16)

    # 1. Gráfico de dispersión con línea ideal
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax1.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Predicción perfecta",
    )
    ax1.set_xlabel("Valores Reales (kW)")
    ax1.set_ylabel("Predicciones (kW)")
    ax1.set_title("Predicciones vs Reales")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Gráfico con barras de error (primeros 200 puntos para visibilidad)
    ax2 = axes[0, 1]
    n_points = min(200, len(y_test_sorted))
    x_range = range(n_points)
    ax2.errorbar(
        x_range,
        y_test_sorted[:n_points],
        yerr=absolute_errors[:n_points],
        fmt="o",
        capsize=2,
        alpha=0.6,
        markersize=3,
        label="Predicción con error",
    )
    ax2.plot(
        x_range,
        y_test_sorted[:n_points],
        "r-",
        alpha=0.7,
        linewidth=1,
        label="Valor Real",
    )
    ax2.set_xlabel("Muestra")
    ax2.set_ylabel("Potencia (kW)")
    ax2.set_title(f"Primeros {n_points} puntos: Reales vs Predicciones")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distribución de errores absolutos
    ax3 = axes[1, 0]
    ax3.hist(absolute_errors, bins=50, alpha=0.7, edgecolor="black")
    ax3.axvline(
        absolute_errors.mean(),
        color="r",
        linestyle="--",
        label=f"Error medio: {absolute_errors.mean():.3f} kW",
    )
    ax3.axvline(
        absolute_errors.median(),
        color="g",
        linestyle="--",
        label=f"Error mediano: {absolute_errors.median():.3f} kW",
    )
    ax3.set_xlabel("Error Absoluto (kW)")
    ax3.set_ylabel("Frecuencia")
    ax3.set_title("Distribución de Errores Absolutos")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Error porcentual vs valores reales
    ax4 = axes[1, 1]
    ax4.scatter(y_test_sorted, percentage_errors, alpha=0.5, s=10)
    ax4.set_xlabel("Valores Reales (kW)")
    ax4.set_ylabel("Error Porcentual (%)")
    ax4.set_title("Error Porcentual vs Valores Reales")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(
        percentage_errors.mean(),
        color="r",
        linestyle="--",
        label=f"Error % medio: {percentage_errors.mean():.2f}%",
    )
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # Métricas de error
    print("\n=== MÉTRICAS DE ERROR ===")
    print(f"MAE (Error Absoluto Medio): {np.mean(absolute_errors):.4f} kW")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.4f} kW")
    print(f"MAPE (Error Porcentual Absoluto Medio): {np.mean(percentage_errors):.2f}%")
    print(f"R²: {1 - np.sum(errors**2) / np.sum((y_test - np.mean(y_test))**2):.4f}")


def plot_df_pred_true(df, y_true, y_pred, all_points_bool=False):

    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Usar timestamps como x
    timestamps = df["timestamp"] if "timestamp" in df.columns else df.index
    x_vals = timestamps if isinstance(timestamps, pd.Series) else np.arange(len(y_true))

    # Calcular errores
    errors = np.abs(y_true - y_pred)
    error_percentage = np.where(errors < 1e-6, 0, (errors / (y_true + 1e-6)) * 100)

    # Obtener irradiancia
    poa_irradiance = (
        df["poa_irradiance_wm2"].values
        if "poa_irradiance_wm2" in df.columns
        else np.zeros_like(y_true)
    )

    # Muestreo
    n_points = len(y_true)
    if not all_points_bool:
        sample_rate = max(1, n_points // 500)
    else:
        sample_rate = 1

    x_sample = x_vals[::sample_rate]
    y_true_sample = y_true[::sample_rate]
    y_pred_sample = y_pred[::sample_rate]
    errors_sample = errors[::sample_rate]
    error_percentage_sample = error_percentage[::sample_rate]
    poa_sample = poa_irradiance[::sample_rate]

    # Eje primario: Valores reales y predicciones
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

    ax1.plot(
        x_sample,
        y_pred_sample,
        "b-",
        alpha=0.7,
        linewidth=1.5,
        label="Predicción",
        marker="s",
        markersize=2,
    )

    # Barras de error
    ax1.errorbar(
        x_sample,
        y_pred_sample,
        yerr=errors_sample,
        fmt="none",
        capsize=2,
        alpha=0.3,
        ecolor="lightgray",
        label="Margen de error",
    )

    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Potencia (kW)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Eje secundario: Error porcentual
    ax2 = ax1.twinx()
    ax2.plot(
        x_sample,
        np.log10(1 + error_percentage_sample),
        "g-",
        alpha=0.6,
        linewidth=0,
        marker="^",
        markersize=2,
        label="Log10 Error %",
    )
    ax2.set_ylabel("Log10 Error Porcentual (%)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.axhline(y=0, color="green", linestyle="--", alpha=0.3)

    # Tercer eje: Error absoluto
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(
        ("outward", 60)
    )  # Desplazar para evitar solapamiento
    ax3.plot(
        x_sample,
        np.log10(1 + errors_sample),
        "orange",
        alpha=0.3,
        linewidth=0,
        marker="s",
        markersize=2,
        label="Log10 Error Absoluto (kW)",
    )
    ax3.set_ylabel("Log10 Error Absoluto (kW)", color="orange")
    ax3.tick_params(axis="y", labelcolor="orange")
    ax3.axhline(y=0, color="orange", linestyle="--", alpha=0.3)

    # Cuarto eje: Irradiancia POA
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("outward", 120))  # Desplazar aún más a la derecha
    ax4.plot(
        x_sample,
        poa_sample,
        "purple",
        alpha=0.5,
        linewidth=0,
        marker="o",
        markersize=3,
        label="POA Irradiance",
    )
    ax4.set_ylabel("POA Irradiance (W/m²)", color="purple")
    ax4.tick_params(axis="y", labelcolor="purple")

    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2 + lines3 + lines4,
        labels1 + labels2 + labels3 + labels4,
        loc="upper left",
        fontsize=8,
    )
    mask_irradiance = df["poa_irradiance_wm2"] < MIN_POA_WM2
    MAE = np.mean(errors[~mask_irradiance])
    MAPE = np.mean(error_percentage[~mask_irradiance])
    timestamps_range = timestamps.iloc[0], timestamps.iloc[-1]
    ax1.set_title(
        "Predicciones vs Valores Reales con Errores e Irradiancia"
        + f"\nDesde {timestamps_range[0]}\nhasta {timestamps_range[1]}"
        + f"\nMAE: {MAE:-4.1f} kW, MAPE: {MAPE:-3.1e}%",
        fontdict={"family": "monospace", "size": 12},
    )

    if "timestamp" in str(type(x_vals)):
        plt.xticks(rotation=45)

    plt.tight_layout()

    # Estadísticas
    print(
        f"Estadísticas para predicciones con irradiance mayor a {MIN_POA_WM2} W/m2"
        + f"\ndesde {timestamps_range[0]}\nhasta {timestamps_range[1]}"
    )
    print(f"\tError absoluto medio:   {MAE:.3f} kW")
    print(f"\tError porcentual medio: {MAPE:.2f}%")
    print(f"\tDesv. estándar error:   {np.std(errors[~mask_irradiance]):.3f} kW")
    print(
        f"\tPercentil 95 error:     {np.percentile(errors[~mask_irradiance], 95):.3f} kW"
    )

    return fig, ax1, ax2, ax3, ax4, MAE, MAPE, timestamps_range


def plot_hist_and_boxplot_error(y_true, y_pred, min_kW=10, bins=100, log=True):
    # Get abs errors
    errors = np.abs(y_true - y_pred)

    # Filter
    mask = y_true > min_kW  # some kW
    errors = errors[mask]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Error histogram and boxplot\nFiltered for 'y_true' > {min_kW} kW")

    # Hist
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=bins, log=log)
    plt.xlabel("|Error| (kW)")
    plt.ylabel("Frequency")

    # Boxplot
    plt.subplot(1, 2, 2)
    plt.boxplot(
        errors,
        vert=True,
        patch_artist=True,
        showmeans=True,
        boxprops=dict(facecolor="steelblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    plt.yscale("log")  # log scale
    plt.ylabel("|Error| (kW)")
    plt.xticks([])  # Empty list removes ticks
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()
