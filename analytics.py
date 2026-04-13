"""
analytics.py — Motor analítico puro para el Rappi Intelligent Analysis System.

Regla crítica: el LLM nunca calcula. Todas las operaciones sobre los datos
viven aquí. Cada función retorna un DataFrame o dict listo para pasarle
al LLM como contexto.
"""

import numpy as np
import pandas as pd

from data_loader import (
    METRICS_WEEK_COLS,
    ORDERS_WEEK_COLS,
    load_data,
)

# ---------------------------------------------------------------------------
# Carga única de DataFrames (load_data está cacheada con lru_cache)
# ---------------------------------------------------------------------------
df_metrics, df_orders, df_metrics_long, df_orders_long = load_data()

# Semanas presentes en los datos (en orden cronológico: L8W_ROLL → L0W_ROLL)
_METRIC_WEEKS = [c for c in METRICS_WEEK_COLS if c in df_metrics.columns]
_ORDER_WEEKS  = [c for c in ORDERS_WEEK_COLS  if c in df_orders.columns]


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _filter_metrics(metric: str, country: str | None = None) -> pd.DataFrame:
    """Filtra df_metrics por métrica y opcionalmente por país."""
    mask = df_metrics["METRIC"] == metric
    if country:
        mask &= df_metrics["COUNTRY"] == country
    return df_metrics[mask].copy()


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Aplica un dict de filtros {columna: valor} a cualquier DataFrame."""
    if not filters:
        return df
    for col, val in filters.items():
        if col in df.columns:
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]
    return df


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def get_top_zones(
    metric: str,
    n: int = 5,
    country: str | None = None,
    week: str = "L0W_ROLL",
) -> pd.DataFrame:
    """
    Retorna las top/bottom N zonas para una métrica en una semana dada.

    Parameters
    ----------
    metric  : nombre exacto de la métrica (e.g. 'Lead Penetration')
    n       : cantidad de zonas. Positivo = top, negativo = bottom N.
    country : filtro opcional por COUNTRY (e.g. 'CO')
    week    : columna de semana a usar como criterio de ordenamiento

    Returns
    -------
    DataFrame con columnas: COUNTRY, CITY, ZONE, ZONE_TYPE, METRIC, {week}
    """
    if week not in _METRIC_WEEKS:
        raise ValueError(f"Semana '{week}' no disponible. Opciones: {_METRIC_WEEKS}")

    df = _filter_metrics(metric, country)
    df = df.dropna(subset=[week])

    ascending = n < 0
    result = (
        df[["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION", "METRIC", week]]
        .sort_values(week, ascending=ascending)
        .head(abs(n))
        .reset_index(drop=True)
    )
    result["rank"] = range(1, len(result) + 1)
    return result


def compare_groups(
    metric: str,
    group_col: str,
    filters: dict | None = None,
) -> pd.DataFrame:
    """
    Compara el valor promedio (y distribución) de una métrica entre grupos.

    Parameters
    ----------
    metric    : nombre de la métrica a comparar
    group_col : columna de agrupación (e.g. 'ZONE_TYPE', 'COUNTRY', 'ZONE_PRIORITIZATION')
    filters   : dict opcional para pre-filtrar (e.g. {'COUNTRY': 'MX'})

    Returns
    -------
    DataFrame con mean, median, std, min, max, count por grupo (semana L0W_ROLL)
    """
    df = _filter_metrics(metric)
    df = _apply_filters(df, filters)
    df = df.dropna(subset=["L0W_ROLL"])

    if group_col not in df.columns:
        raise ValueError(f"Columna '{group_col}' no existe. Opciones: {list(df.columns)}")

    result = (
        df.groupby(group_col)["L0W_ROLL"]
        .agg(mean="mean", median="median", std="std", min="min", max="max", count="count")
        .round(4)
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    result["metric"] = metric
    return result


def get_trend(
    zone: str,
    metric: str,
    n_weeks: int = 8,
) -> pd.DataFrame:
    """
    Retorna la serie temporal de una zona+métrica para las últimas N semanas.

    Parameters
    ----------
    zone    : nombre exacto de la zona (e.g. 'Chapinero')
    metric  : nombre de la métrica
    n_weeks : cuántas semanas hacia atrás incluir (máx 8)

    Returns
    -------
    DataFrame con columnas: ZONE, METRIC, SEMANA, VALOR, ordenado cronológicamente
    """
    n_weeks = min(n_weeks, len(_METRIC_WEEKS))

    mask = (df_metrics_long["ZONE"] == zone) & (df_metrics_long["METRIC"] == metric)
    df = df_metrics_long[mask].dropna(subset=["VALOR"])

    # Tomar las últimas n_weeks semanas disponibles
    semanas_disponibles = sorted(df["SEMANA"].unique())
    semanas_target = semanas_disponibles[-n_weeks:]
    df = df[df["SEMANA"].isin(semanas_target)]

    result = (
        df[["COUNTRY", "CITY", "ZONE", "METRIC", "SEMANA", "VALOR"]]
        .sort_values("SEMANA")
        .reset_index(drop=True)
    )
    # Agregar columna de cambio semana a semana
    result["delta"] = result["VALOR"].diff().round(4)
    result["delta_pct"] = (result["delta"] / result["VALOR"].shift(1) * 100).round(2)
    return result


def aggregate_metric(
    metric: str,
    group_by: str | list[str],
    agg_func: str = "mean",
) -> pd.DataFrame:
    """
    Agrega una métrica por una o más dimensiones geográficas/tipológicas.

    Parameters
    ----------
    metric   : nombre de la métrica
    group_by : columna(s) de agrupación (e.g. 'COUNTRY', ['COUNTRY', 'ZONE_TYPE'])
    agg_func : función de agregación: 'mean', 'median', 'sum', 'min', 'max'

    Returns
    -------
    DataFrame con el valor agregado por grupo, semana L0W_ROLL
    """
    valid_funcs = {"mean", "median", "sum", "min", "max"}
    if agg_func not in valid_funcs:
        raise ValueError(f"agg_func debe ser uno de {valid_funcs}")

    df = _filter_metrics(metric).dropna(subset=["L0W_ROLL"])

    if isinstance(group_by, str):
        group_by = [group_by]

    agg_map = {"mean": "mean", "median": "median", "sum": "sum", "min": "min", "max": "max"}
    result = (
        df.groupby(group_by)["L0W_ROLL"]
        .agg(**{agg_func: agg_map[agg_func], "count": "count"})
        .round(4)
        .reset_index()
        .sort_values(agg_func, ascending=False)
    )
    result["metric"] = metric
    return result


def multivariable_scan(
    metric_high: str,
    metric_low: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Detecta zonas que tienen desempeño alto en metric_high y bajo en metric_low.

    Usa cuantiles calculados por país para comparaciones justas.
    threshold=0.5 → zonas sobre la mediana en metric_high Y bajo la mediana en metric_low.
    threshold=0.7 → criterio más estricto (top 30% vs bottom 30%).

    Parameters
    ----------
    metric_high : métrica donde la zona debe tener valor ALTO
    metric_low  : métrica donde la zona debe tener valor BAJO
    threshold   : cuantil de corte (0–1). Alto = > threshold, Bajo = < (1 - threshold)

    Returns
    -------
    DataFrame con zonas que cumplen ambas condiciones + ambos valores
    """
    id_cols = ["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION"]

    high_df = (
        _filter_metrics(metric_high)
        .dropna(subset=["L0W_ROLL"])[id_cols + ["L0W_ROLL"]]
        .rename(columns={"L0W_ROLL": metric_high})
    )
    low_df = (
        _filter_metrics(metric_low)
        .dropna(subset=["L0W_ROLL"])[id_cols + ["L0W_ROLL"]]
        .rename(columns={"L0W_ROLL": metric_low})
    )

    merged = high_df.merge(low_df, on=id_cols, how="inner")

    # Cuantiles por país para comparación relativa
    merged["q_high"] = merged.groupby("COUNTRY")[metric_high].transform(
        lambda x: x.rank(pct=True)
    )
    merged["q_low"] = merged.groupby("COUNTRY")[metric_low].transform(
        lambda x: x.rank(pct=True)
    )

    result = merged[
        (merged["q_high"] > threshold) & (merged["q_low"] < (1 - threshold))
    ].copy()

    result = result.drop(columns=["q_high", "q_low"]).sort_values(
        metric_high, ascending=False
    ).reset_index(drop=True)

    return result


def growth_analysis(n_weeks: int = 5) -> pd.DataFrame:
    """
    Calcula el crecimiento en volumen de órdenes entre hace N semanas y la semana actual.

    Parameters
    ----------
    n_weeks : número de semanas hacia atrás para comparar con L0W

    Returns
    -------
    DataFrame con zonas, órdenes en L0W, órdenes hace N semanas, crecimiento % y absoluto,
    ordenado de mayor a menor crecimiento porcentual
    """
    n_weeks = min(n_weeks, len(_ORDER_WEEKS) - 1)
    start_col = f"L{n_weeks}W"
    end_col = "L0W"

    if start_col not in _ORDER_WEEKS:
        raise ValueError(f"No hay datos para {n_weeks} semanas atrás.")

    df = df_orders[["COUNTRY", "CITY", "ZONE", start_col, end_col]].dropna().copy()

    df["orders_start"] = df[start_col]
    df["orders_end"]   = df[end_col]
    df["growth_abs"]   = (df["orders_end"] - df["orders_start"]).round(0)
    df["growth_pct"]   = ((df["growth_abs"] / df["orders_start"]) * 100).round(2)

    result = (
        df[["COUNTRY", "CITY", "ZONE", "orders_start", "orders_end", "growth_abs", "growth_pct"]]
        .rename(columns={"orders_start": f"orders_L{n_weeks}W", "orders_end": "orders_L0W"})
        .sort_values("growth_pct", ascending=False)
        .reset_index(drop=True)
    )
    result["rank"] = range(1, len(result) + 1)
    return result


def benchmark_analysis(
    country: str,
    zone_type: str | None = None,
) -> pd.DataFrame:
    """
    Identifica zonas con desempeño divergente respecto a sus pares del mismo país y tipo.

    Una zona es "divergente" si su z-score en alguna métrica supera ±1.5 desvíos estándar
    del grupo de referencia (mismo COUNTRY + ZONE_TYPE si se especifica).

    Parameters
    ----------
    country   : código de país (e.g. 'CO', 'MX')
    zone_type : 'Wealthy' o 'Non Wealthy'. None = todos los tipos

    Returns
    -------
    DataFrame con zona, métrica, valor, media del grupo, z-score, ordenado por |z_score|
    """
    df = df_metrics[df_metrics["COUNTRY"] == country].dropna(subset=["L0W_ROLL"]).copy()

    if zone_type:
        df = df[df["ZONE_TYPE"] == zone_type]

    if df.empty:
        return pd.DataFrame()

    # Z-score por métrica dentro del grupo
    df["group_mean"] = df.groupby("METRIC")["L0W_ROLL"].transform("mean")
    df["group_std"]  = df.groupby("METRIC")["L0W_ROLL"].transform("std")
    df["z_score"]    = ((df["L0W_ROLL"] - df["group_mean"]) / df["group_std"]).round(3)

    result = df[df["z_score"].abs() > 1.5][
        ["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "METRIC", "L0W_ROLL", "group_mean", "group_std", "z_score"]
    ].copy()

    result["group_mean"] = result["group_mean"].round(4)
    result["group_std"]  = result["group_std"].round(4)
    result["L0W_ROLL"]   = result["L0W_ROLL"].round(4)

    result = result.sort_values("z_score", key=abs, ascending=False).reset_index(drop=True)
    return result


def detect_anomalies(threshold: float = 0.10) -> pd.DataFrame:
    """
    Detecta cambios bruscos entre L1W_ROLL y L0W_ROLL mayores al umbral dado.

    Parameters
    ----------
    threshold : cambio porcentual mínimo para considerar anomalía (default 0.10 = 10%)

    Returns
    -------
    DataFrame con zona, métrica, valor anterior, valor actual, cambio % — ordenado por |cambio|
    """
    if "L1W_ROLL" not in _METRIC_WEEKS:
        return pd.DataFrame(columns=["ZONE", "METRIC", "L1W_ROLL", "L0W_ROLL", "change_pct"])

    df = df_metrics[["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "METRIC", "L1W_ROLL", "L0W_ROLL"]].dropna().copy()

    df["change_abs"] = df["L0W_ROLL"] - df["L1W_ROLL"]
    df["change_pct"] = (df["change_abs"] / df["L1W_ROLL"].abs()).round(4)

    # Evitar divisiones por cero — zonas donde L1W_ROLL era exactamente 0
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["change_pct"])

    result = df[df["change_pct"].abs() > threshold].copy()
    result["direction"] = result["change_pct"].apply(lambda x: "mejora" if x > 0 else "caida")
    result = result.sort_values("change_pct", key=abs, ascending=False).reset_index(drop=True)
    return result


def detect_declining_trends(min_weeks: int = 3) -> pd.DataFrame:
    """
    Detecta combinaciones zona+métrica en deterioro durante al menos N semanas consecutivas.

    Una tendencia es "declinante" si los valores caen monotónicamente durante las
    últimas `min_weeks` semanas (sin importar el valor absoluto).

    Parameters
    ----------
    min_weeks : mínimo de semanas consecutivas de caída para reportar (default 3)

    Returns
    -------
    DataFrame con zona, métrica, valor inicial y final del tramo, semanas de caída,
    cambio total porcentual
    """
    # Necesitamos al menos min_weeks + 1 puntos para min_weeks caídas
    required = min_weeks + 1
    semanas = _METRIC_WEEKS[-required:]  # últimas semanas del rango

    if len(semanas) < required:
        return pd.DataFrame()

    df = (
        df_metrics_long[df_metrics_long["SEMANA"].isin(semanas)]
        .dropna(subset=["VALOR"])
        .sort_values(["COUNTRY", "CITY", "ZONE", "METRIC", "SEMANA"])
    )

    records = []

    for (country, city, zone, metric), group in df.groupby(
        ["COUNTRY", "CITY", "ZONE", "METRIC"], observed=True
    ):
        values = group.sort_values("SEMANA")["VALOR"].tolist()
        if len(values) < required:
            continue

        # Verificar que todos los diffs sean negativos (monotónicamente decreciente)
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        if all(d < 0 for d in diffs):
            val_start = values[0]
            val_end   = values[-1]
            change_pct = round((val_end - val_start) / abs(val_start) * 100, 2) if val_start != 0 else None
            records.append({
                "COUNTRY":        country,
                "CITY":           city,
                "ZONE":           zone,
                "METRIC":         metric,
                "weeks_declining": len(diffs),
                "val_start":      round(val_start, 4),
                "val_end":        round(val_end, 4),
                "change_pct":     change_pct,
                "semana_inicio":  semanas[0],
                "semana_fin":     semanas[-1],
            })

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values("change_pct")  # peores caídas primero
        .reset_index(drop=True)
    )


def detect_correlations(min_corr: float = 0.6) -> pd.DataFrame:
    """
    Encuentra pares de métricas con correlación fuerte en el snapshot de L0W_ROLL.

    Usa una matriz de correlación de Pearson sobre todas las zonas (una fila por zona,
    columnas = métricas). Pares con |r| ≥ min_corr son reportados.

    Parameters
    ----------
    min_corr : correlación mínima en valor absoluto (default 0.6)

    Returns
    -------
    DataFrame con columnas: metric_a, metric_b, correlation, direction
    ordenado de mayor a menor correlación absoluta
    """
    # Pivot: una fila por zona, columnas = métricas, valores = L0W_ROLL
    pivot = df_metrics.pivot_table(
        index=["COUNTRY", "CITY", "ZONE"],
        columns="METRIC",
        values="L0W_ROLL",
        aggfunc="first",
    )

    # Sólo métricas con datos suficientes (al menos 50% de zonas con valor)
    pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.5))

    corr_matrix = pivot.corr(method="pearson")

    records = []
    metrics = corr_matrix.columns.tolist()

    for i in range(len(metrics)):
        for j in range(i + 1, len(metrics)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= min_corr and not np.isnan(r):
                records.append({
                    "metric_a":    metrics[i],
                    "metric_b":    metrics[j],
                    "correlation": round(r, 4),
                    "direction":   "positiva" if r > 0 else "negativa",
                })

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values("correlation", key=abs, ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Smoke test cuando se ejecuta directamente
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint

    print("--- get_top_zones (Lead Penetration, CO) ---")
    print(get_top_zones("Lead Penetration", n=5, country="CO"))

    print("\n--- compare_groups (Perfect Orders, ZONE_TYPE, MX) ---")
    print(compare_groups("Perfect Orders", "ZONE_TYPE", filters={"COUNTRY": "MX"}))

    print("\n--- get_trend (Chapinero, Lead Penetration) ---")
    result = get_trend("Chapinero", "Lead Penetration", n_weeks=5)
    print(result if not result.empty else "Zona no encontrada")

    print("\n--- aggregate_metric (Gross Profit UE, COUNTRY) ---")
    print(aggregate_metric("Gross Profit UE", "COUNTRY"))

    print("\n--- multivariable_scan ---")
    print(multivariable_scan("Lead Penetration", "Perfect Orders", threshold=0.5).head())

    print("\n--- growth_analysis (5 semanas) ---")
    print(growth_analysis(n_weeks=5).head())

    print("\n--- benchmark_analysis (CO, Wealthy) ---")
    print(benchmark_analysis("CO", zone_type="Wealthy").head())

    print("\n--- detect_anomalies (>10%) ---")
    anom = detect_anomalies(0.10)
    print(f"{len(anom)} anomalías detectadas")
    print(anom.head())

    print("\n--- detect_declining_trends (3 semanas) ---")
    decl = detect_declining_trends(3)
    print(f"{len(decl)} tendencias declinantes")
    print(decl.head())

    print("\n--- detect_correlations (r >= 0.6) ---")
    print(detect_correlations(0.6))
