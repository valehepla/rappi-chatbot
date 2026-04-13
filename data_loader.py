import pandas as pd
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "rappi_data.xlsx"

METRICS_WEEK_COLS = [f"L{i}W_ROLL" for i in range(8, -1, -1)]  # L8W_ROLL ... L0W_ROLL
ORDERS_WEEK_COLS = [f"L{i}W" for i in range(8, -1, -1)]         # L8W ... L0W

METRICS_ID_COLS = ["COUNTRY", "CITY", "ZONE", "ZONE_TYPE", "ZONE_PRIORITIZATION", "METRIC"]
ORDERS_ID_COLS  = ["COUNTRY", "CITY", "ZONE"]


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both sheets from the Excel file."""
    xls = pd.ExcelFile(DATA_PATH)
    df_m = pd.read_excel(xls, sheet_name="RAW_INPUT_METRICS")
    df_o = pd.read_excel(xls, sheet_name="RAW_ORDERS")
    return df_m, df_o


def _clean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning for RAW_INPUT_METRICS.
    - Strip whitespace from string columns.
    - Keep nulls in week columns as-is (they represent zones that didn't exist yet).
    - Only week columns present in the sheet are kept (order normalised).
    """
    for col in METRICS_ID_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Keep only the week columns that actually exist in the sheet
    present_week_cols = [c for c in METRICS_WEEK_COLS if c in df.columns]
    return df[METRICS_ID_COLS + present_week_cols].copy()


def _clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning for RAW_ORDERS.
    - Strip whitespace from string columns.
    - Drop the METRIC column (always "Orders") — redundant.
    - Keep nulls (~20%) as-is; callers filter with dropna() when needed.
    """
    for col in ["COUNTRY", "CITY", "ZONE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    present_week_cols = [c for c in ORDERS_WEEK_COLS if c in df.columns]
    return df[ORDERS_ID_COLS + present_week_cols].copy()


def _melt_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long format for df_metrics.
    Columns: COUNTRY, CITY, ZONE, ZONE_TYPE, ZONE_PRIORITIZATION, METRIC, SEMANA, VALOR
    SEMANA values: 'L8W_ROLL' … 'L0W_ROLL'
    Rows with null VALOR are dropped (represent non-existent zone/week combos).
    """
    present_week_cols = [c for c in METRICS_WEEK_COLS if c in df.columns]
    long = df.melt(
        id_vars=METRICS_ID_COLS,
        value_vars=present_week_cols,
        var_name="SEMANA",
        value_name="VALOR",
    )
    long = long.dropna(subset=["VALOR"]).reset_index(drop=True)
    # Ordered categorical so sorting by SEMANA works chronologically
    long["SEMANA"] = pd.Categorical(long["SEMANA"], categories=present_week_cols, ordered=True)
    return long


def _melt_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long format for df_orders.
    Columns: COUNTRY, CITY, ZONE, SEMANA, ORDERS
    SEMANA values: 'L8W' … 'L0W'
    Rows with null ORDERS are dropped.
    """
    present_week_cols = [c for c in ORDERS_WEEK_COLS if c in df.columns]
    long = df.melt(
        id_vars=ORDERS_ID_COLS,
        value_vars=present_week_cols,
        var_name="SEMANA",
        value_name="ORDERS",
    )
    long = long.dropna(subset=["ORDERS"]).reset_index(drop=True)
    long["SEMANA"] = pd.Categorical(long["SEMANA"], categories=present_week_cols, ordered=True)
    return long


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare all DataFrames. Cached so the Excel is read only once per process.

    Returns
    -------
    df_metrics      : wide format — one row per (zone, metric)
    df_orders       : wide format — one row per zone
    df_metrics_long : long format — one row per (zone, metric, week)
    df_orders_long  : long format — one row per (zone, week)
    """
    raw_m, raw_o = _load_raw()
    df_metrics = _clean_metrics(raw_m)
    df_orders  = _clean_orders(raw_o)
    df_metrics_long = _melt_metrics(df_metrics)
    df_orders_long  = _melt_orders(df_orders)
    return df_metrics, df_orders, df_metrics_long, df_orders_long


def get_data_summary() -> dict:
    """
    Return basic statistics about the dataset — useful for the LLM system prompt
    and for displaying an info panel in the UI.
    """
    df_metrics, df_orders, df_metrics_long, df_orders_long = load_data()

    present_metric_weeks = [c for c in METRICS_WEEK_COLS if c in df_metrics.columns]
    present_order_weeks  = [c for c in ORDERS_WEEK_COLS  if c in df_orders.columns]

    # Null counts in wide format
    metrics_null_pct = (
        df_metrics[present_metric_weeks].isnull().sum().sum()
        / (len(df_metrics) * len(present_metric_weeks))
        * 100
    )
    orders_null_pct = (
        df_orders[present_order_weeks].isnull().sum().sum()
        / (len(df_orders) * len(present_order_weeks))
        * 100
    )

    return {
        # --- RAW_INPUT_METRICS ---
        "metrics_rows": len(df_metrics),
        "metrics_unique_zones": df_metrics["ZONE"].nunique(),
        "metrics_unique_countries": df_metrics["COUNTRY"].nunique(),
        "metrics_unique_metrics": df_metrics["METRIC"].nunique(),
        "metrics_list": sorted(df_metrics["METRIC"].unique().tolist()),
        "metrics_countries": sorted(df_metrics["COUNTRY"].unique().tolist()),
        "metrics_zone_types": sorted(df_metrics["ZONE_TYPE"].unique().tolist()),
        "metrics_week_cols": present_metric_weeks,
        "metrics_null_pct": round(metrics_null_pct, 2),
        # --- RAW_ORDERS ---
        "orders_rows": len(df_orders),
        "orders_unique_zones": df_orders["ZONE"].nunique(),
        "orders_week_cols": present_order_weeks,
        "orders_null_pct": round(orders_null_pct, 2),
        # --- LONG formats ---
        "metrics_long_rows": len(df_metrics_long),
        "orders_long_rows": len(df_orders_long),
    }


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_m, df_o, df_ml, df_ol = load_data()
    print("=== df_metrics ===")
    print(df_m.shape, "\n", df_m.head(2))
    print("\n=== df_orders ===")
    print(df_o.shape, "\n", df_o.head(2))
    print("\n=== df_metrics_long ===")
    print(df_ml.shape, "\n", df_ml.head(2))
    print("\n=== df_orders_long ===")
    print(df_ol.shape, "\n", df_ol.head(2))
    print("\n=== Summary ===")
    from pprint import pprint
    pprint(get_data_summary())
