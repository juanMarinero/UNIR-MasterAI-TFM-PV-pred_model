#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=6 foldmethod=indent :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_corr(df, cols, annotbool=True, figsize=(5, 5)):
    corr = df[cols].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annotbool,
        mask=np.triu(corr),
        fmt=".2f",
        cmap="RdBu",
        vmin=-1,
        vmax=1,
    )
    plt.show()
    return corr


def plot_cumulative_variance(pca, threshold=0.95):
    # 1. Calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # 2. Find the optimal number of components
    # np.argmax returns the first index where the condition is True
    optimal_idx = np.argmax(cumulative_variance >= threshold)
    n_components = optimal_idx + 1  # Convert 0-based index to 1-based count
    actual_variance = cumulative_variance[optimal_idx]

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    # Plot the cumulative variance curve
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        "b-o",
        label="Cumulative Explained Variance",
    )

    # Add a horizontal dashed line for the threshold
    plt.axhline(
        y=threshold, color="r", linestyle="--", label=f"Threshold ({threshold*100}%)"
    )

    # Add a vertical dashed line for the selected n_components
    plt.axvline(
        x=n_components,
        color="g",
        linestyle="--",
        label=f"Selected Components (n={n_components})",
    )

    # Mark the specific point where the threshold is crossed
    plt.plot(n_components, actual_variance, "ro", markersize=10, zorder=5)

    # Annotations
    plt.annotate(
        f"{actual_variance:.4f}",
        xy=(n_components, actual_variance),
        xytext=(n_components - 0.5, actual_variance + 0.02),
        arrowprops=dict(facecolor="black", shrink=0.05),
        fontsize=10,
    )

    # Labels and Title
    plt.title("Cumulative Explained Variance by Number of Components")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.xticks(range(1, len(cumulative_variance) + 1))
    plt.legend(loc="lower center")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()

    return cumulative_variance, n_components, actual_variance


def plot_PCA_coef_portion(
    pca,
    cols,
    n_components,
    legendbool=True,
    fontsize=20,
    figsize=(15, 10),
    min_percentage_to_show=2,
    show_col_name=True,
):
    """
    Visualiza la contribución de cada característica a las componentes principales.

    Parámetros:
    - pca: objeto PCA ya ajustado
    - cols: nombres de las columnas/características
    - n_components: número de componentes a mostrar
    - legendbool: si mostrar leyenda
    """
    pca_xticks = [f"PC-{k+1}" for k in range(n_components)]

    # Matriz de coeficientes (valor absoluto)
    df_coef = pd.DataFrame(
        np.abs(pca.components_[:n_components, :]),
        index=pca_xticks,
        columns=cols,
    )

    # Convertir a porcentajes por fila
    df_coef_portion = df_coef.div(df_coef.sum(axis=1), axis=0) * 100

    # Graficar
    fig, ax = plt.subplots(figsize=figsize)
    df_coef_portion.plot(kind="bar", stacked=True, ax=ax, width=0.8)

    # Configurar leyenda
    if legendbool:
        ax.legend(
            bbox_to_anchor=(1.02, 1.0), ncol=1, fontsize=fontsize, loc="upper left"
        )
    else:
        ax.get_legend().remove()

    # Añadir etiquetas de porcentaje en las barras
    y_offset = 0
    for bar_idx, (idx, row) in enumerate(df_coef_portion.iterrows()):
        cumsum = 0
        for col_idx, (col, val) in enumerate(row.items()):
            cumsum += val
            if (
                val > min_percentage_to_show
            ):  # Solo mostrar si el porcentaje es significativo
                ax.text(
                    bar_idx,
                    cumsum - val / 2 + y_offset,
                    f"{val:.1f}%" + f" {col}" if show_col_name else "",
                    ha="center",
                    va="center",
                    fontsize=7,
                    weight="bold",
                )

    # Configurar ejes
    ax.set_xlabel("Componentes Principales", fontsize=16)
    ax.set_ylabel("Contribución (%)", fontsize=16)
    ax.tick_params(labelsize=14, rotation=45)
    ax.set_title(
        "Contribución de características a las Componentes Principales", fontsize=18
    )

    # Añadir línea de varianza acumulada en segundo eje
    ax2 = ax.twinx()
    cumulative_var = 100 * np.cumsum(pca.explained_variance_ratio_[:n_components])
    ax2.plot(
        range(len(cumulative_var)),
        cumulative_var,
        "ro-",
        linewidth=2,
        markersize=8,
        label="Varianza acumulada",
    )
    ax2.set_ylabel("Varianza explicada acumulada (%)", fontsize=14, color="red")
    ax2.tick_params(labelsize=12, colors="red")
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()

    return df_coef


def get_pca(X_train_norm, threshold=0.95):
    if False:
        corr = plot_corr(X_train_norm, X_train_norm.columns)

    pca = PCA().fit(X_train_norm)
    cumulative_variance, n_components, actual_variance = plot_cumulative_variance(
        pca, threshold=threshold
    )
    print(
        f"Selected {n_components} components explaining {actual_variance:.4%} of variance."
    )

    df_coef = plot_PCA_coef_portion(
        pca, X_train_norm.columns, n_components=n_components
    )

    print(
        f"\n\nEl DataFrame 'df_coef' contiene los pesos de cada columna original en cada PCA:"
    )
    print(f"\tdf_coef.columns: {df_coef.columns}")
    for _df in [df_coef, df_coef.T]:
        display(
            _df.style.set_caption("pca.components_")
            .background_gradient(axis=1, cmap="Greens")
            .format("{:.2f}")
        )

    # Crear un DataFrame con los resultados de PCA
    pca = PCA(n_components=n_components)  # Usa el número óptimo de componentes
    X_pca = pca.fit_transform(X_train_norm)  # Transformar los datos

    # 2. Crear DataFrame con los componentes principales
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])

    print(f"\n\nEl DataFrame 'df_pca' contiene los resultados de PCA:")
    display(df_pca.sample(3))

    return pca
