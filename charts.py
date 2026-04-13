"""
charts.py — Generación de gráficos Plotly para el Rappi Intelligent Analysis System.

Regla: solo Plotly, nunca matplotlib. Cada función retorna un go.Figure
listo para st.plotly_chart(). Retorna None si no hay datos.
"""

import plotly.express as px
import plotly.graph_objects as go

import analytics as an

# Paleta corporativa coherente entre gráficos
_PALETTE = px.colors.qualitative.Set2


def _label_semana(s: str) -> str:
    """'L3W_ROLL' → 'L3W'  |  'L3W' → 'L3W'"""
    return s.replace("_ROLL", "")


# ---------------------------------------------------------------------------
# chart_trend — Line chart (tendencia temporal de una zona+métrica)
# ---------------------------------------------------------------------------

def chart_trend(zone: str, metric: str, n_weeks: int = 8) -> go.Figure | None:
    """
    Línea temporal de VALOR semana a semana para una zona+métrica dada.
    Incluye marcadores y anotación del cambio total.
    """
    df = an.get_trend(zone, metric, n_weeks)
    if df.empty:
        return None

    df = df.copy()
    df["semana_label"] = df["SEMANA"].astype(str).apply(_label_semana)

    fig = px.line(
        df,
        x="semana_label",
        y="VALOR",
        markers=True,
        title=f"<b>{metric}</b> — {zone}",
        labels={"semana_label": "Semana", "VALOR": "Valor"},
        color_discrete_sequence=[_PALETTE[0]],
    )

    # Sombreado de tendencia
    first, last = df["VALOR"].iloc[0], df["VALOR"].iloc[-1]
    delta_pct = ((last - first) / abs(first) * 100) if first != 0 else 0
    arrow = "▲" if delta_pct >= 0 else "▼"
    color = "green" if delta_pct >= 0 else "red"

    fig.add_annotation(
        x=df["semana_label"].iloc[-1],
        y=last,
        text=f"  {arrow} {abs(delta_pct):.1f}%",
        showarrow=False,
        font=dict(color=color, size=13),
        xanchor="left",
    )

    fig.update_traces(line=dict(width=2.5))
    fig.update_layout(
        xaxis_title="Semana (L8W = hace 8 sem · L0W = actual)",
        yaxis_title=metric,
        plot_bgcolor="white",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# chart_compare — Bar chart (comparación entre grupos)
# ---------------------------------------------------------------------------

def chart_compare(
    metric: str,
    group_col: str = "ZONE_TYPE",
    filters: dict | None = None,
) -> go.Figure | None:
    """
    Barras verticales con media ± desviación estándar por grupo.
    Ideal para comparar Wealthy vs Non Wealthy, o por país, etc.
    """
    df = an.compare_groups(metric, group_col, filters)
    if df.empty:
        return None

    title = f"<b>{metric}</b> por {group_col}"
    if filters:
        title += "  (" + ", ".join(f"{k}={v}" for k, v in filters.items()) + ")"

    fig = px.bar(
        df,
        x=group_col,
        y="mean",
        error_y="std",
        color=group_col,
        title=title,
        labels={"mean": "Promedio (L0W)", group_col: group_col},
        text_auto=".3f",
        color_discrete_sequence=_PALETTE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        yaxis_title="Valor promedio",
    )
    return fig


# ---------------------------------------------------------------------------
# chart_top_k — Horizontal bar chart (ranking de zonas)
# ---------------------------------------------------------------------------

def chart_top_k(
    metric: str,
    n: int = 5,
    country: str | None = None,
    week: str = "L0W_ROLL",
) -> go.Figure | None:
    """
    Ranking horizontal de las top/bottom N zonas.
    Color = país, para distinguir rápidamente si no hay filtro de país.
    """
    df = an.get_top_zones(metric, n, country, week)
    if df.empty:
        return None

    direction = "Top" if n > 0 else "Bottom"
    title = f"<b>{direction} {abs(n)} zonas — {metric}</b>"
    if country:
        title += f"  ({country})"

    # Ordenar de menor a mayor para que el #1 quede arriba en horizontal bar
    df_sorted = df.sort_values(week, ascending=True)

    fig = px.bar(
        df_sorted,
        x=week,
        y="ZONE",
        color="COUNTRY",
        orientation="h",
        title=title,
        labels={week: "Valor (L0W)", "ZONE": "Zona"},
        text_auto=".3f",
        color_discrete_sequence=_PALETTE,
        hover_data=["CITY", "ZONE_TYPE"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis_title="",
        plot_bgcolor="white",
        legend_title="País",
        height=max(300, abs(n) * 45),
    )
    return fig


# ---------------------------------------------------------------------------
# chart_multivariable — Scatter plot (zonas con contraste de métricas)
# ---------------------------------------------------------------------------

def chart_multivariable(
    metric_high: str,
    metric_low: str,
    threshold: float = 0.5,
) -> go.Figure | None:
    """
    Scatter plot: eje X = metric_high, eje Y = metric_low.
    Cuadrantes resaltan las zonas con el contraste buscado.
    """
    df = an.multivariable_scan(metric_high, metric_low, threshold)
    if df.empty:
        return None

    fig = px.scatter(
        df,
        x=metric_high,
        y=metric_low,
        color="COUNTRY",
        hover_name="ZONE",
        hover_data=["CITY", "ZONE_TYPE"],
        title=f"<b>Alto {metric_high} × Bajo {metric_low}</b>",
        color_discrete_sequence=_PALETTE,
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=9))
    fig.update_layout(plot_bgcolor="white")
    return fig


# ---------------------------------------------------------------------------
# chart_growth — Bar chart horizontal (crecimiento de órdenes)
# ---------------------------------------------------------------------------

def chart_growth(n_weeks: int = 5, top_n: int = 15) -> go.Figure | None:
    """Top N zonas con mayor crecimiento porcentual de órdenes."""
    df = an.growth_analysis(n_weeks).head(top_n)
    if df.empty:
        return None

    fig = px.bar(
        df.sort_values("growth_pct"),
        x="growth_pct",
        y="ZONE",
        color="COUNTRY",
        orientation="h",
        title=f"<b>Top {top_n} zonas por crecimiento de órdenes</b> (L{n_weeks}W → L0W)",
        labels={"growth_pct": "Crecimiento %", "ZONE": "Zona"},
        text_auto=".1f",
        color_discrete_sequence=_PALETTE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="", plot_bgcolor="white", height=max(350, top_n * 40))
    return fig


# ---------------------------------------------------------------------------
# Dispatcher principal — llamado desde app.py
# ---------------------------------------------------------------------------

def make_chart(intent: str, context: dict) -> go.Figure | None:
    """
    Genera el gráfico correspondiente al intent y los parámetros del last_context.
    Retorna None si no aplica o si hay error.
    """
    try:
        if intent == "trend":
            return chart_trend(
                zone=context["zone"],
                metric=context["metric"],
                n_weeks=context.get("n_weeks", 8),
            )
        if intent == "compare":
            return chart_compare(
                metric=context["metric"],
                group_col=context.get("group_col", "ZONE_TYPE"),
                filters=context.get("filters"),
            )
        if intent == "top_k":
            return chart_top_k(
                metric=context["metric"],
                n=context.get("n", 5),
                country=context.get("country"),
                week=context.get("week", "L0W_ROLL"),
            )
        if intent == "multivariable_scan":
            return chart_multivariable(
                metric_high=context["metric_high"],
                metric_low=context["metric_low"],
                threshold=context.get("threshold", 0.5),
            )
        if intent == "growth_inference":
            return chart_growth(n_weeks=context.get("n_weeks", 5))
    except Exception:
        return None
    return None
