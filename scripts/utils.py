#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=4 foldmethod=indent :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
