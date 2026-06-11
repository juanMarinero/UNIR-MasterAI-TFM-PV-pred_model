#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=4 foldmethod=indent :

import numpy as np
import pandas as pd
from scipy import stats


def remove_outliers_high_threshold(df, columns=None, threshold=5, min_std=1e-10):
    """
    Elimina filas con outliers basado en z-score.
    Excluye automáticamente columnas con desviación estándar muy pequeña.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if "timestamp" in columns:
            columns.remove("timestamp")

    # Filtrar columnas con varianza significativa
    variable_columns = []
    skipped_columns = []
    for col in columns:
        std = df[col].std()
        if std > min_std:
            variable_columns.append(col)
        else:
            skipped_columns.append(col)

    if skipped_columns:
        print(f"Columnas excluidas (std ≤ {min_std}): {skipped_columns}")

    if not variable_columns:
        print("No hay columnas variables para evaluar outliers")
        return df

    print(f"Evaluando outliers en {len(variable_columns)} columnas variables")

    # Calcular z-scores solo en columnas variables
    z_scores = np.abs(stats.zscore(df[variable_columns], nan_policy="omit"))

    # Manejar posibles NaN
    z_scores = np.nan_to_num(z_scores, nan=0.0)

    # Mantener filas donde TODAS las columnas variables tienen z-score < threshold
    keep = (z_scores < threshold).all(axis=1)

    removed = (~keep).sum()
    print(f"Eliminadas {removed} filas de {len(df)} ({removed/len(df)*100:.2f}%)")
    print(f"Filas restantes: {keep.sum()}")

    return df[keep].copy()
