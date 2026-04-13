"""
insights.py — Motor de insights automáticos del Rappi Intelligent Analysis System.

generate_report() escanea los datos con analytics.py y retorna un reporte
estructurado en Markdown listo para st.markdown().
"""

from datetime import datetime

import analytics as an
from data_loader import get_data_summary


def generate_report() -> str:
    """
    Ejecuta el escaneo completo y retorna un reporte Markdown con:
    1. Resumen ejecutivo
    2. Anomalías semana a semana
    3. Tendencias preocupantes
    4. Correlaciones entre métricas
    5. Oportunidades (zonas con mayor crecimiento)
    """
    summary = get_data_summary()

    # ── Cálculos ────────────────────────────────────────────────────────────
    anomalies  = an.detect_anomalies(threshold=0.10)
    declining  = an.detect_declining_trends(min_weeks=3)
    corr       = an.detect_correlations(min_corr=0.6)
    growth     = an.growth_analysis(n_weeks=5)

    # Métricas con más caídas (para titular el resumen ejecutivo)
    top_anomaly_metric = (
        anomalies[anomalies["direction"] == "caida"]["METRIC"].value_counts().index[0]
        if not anomalies.empty and "direction" in anomalies.columns
        else "N/A"
    )
    top_growth_zone = growth.iloc[0]["ZONE"] if not growth.empty else "N/A"

    ts = datetime.now().strftime("%d/%m/%Y %H:%M")

    lines: list[str] = []

    # ── Encabezado ───────────────────────────────────────────────────────────
    lines += [
        f"# Reporte de Insights Automáticos",
        f"*Generado: {ts}*",
        "",
        "---",
        "",
    ]

    # ── 1. Resumen ejecutivo ─────────────────────────────────────────────────
    n_caidas  = len(anomalies[anomalies["direction"] == "caida"]) if not anomalies.empty else 0
    n_mejoras = len(anomalies[anomalies["direction"] == "mejora"]) if not anomalies.empty else 0

    lines += [
        "## Resumen Ejecutivo",
        "",
        f"| Indicador | Valor |",
        f"|---|---|",
        f"| Zonas con caída >10% esta semana | **{n_caidas}** |",
        f"| Zonas con mejora >10% esta semana | **{n_mejoras}** |",
        f"| Combinaciones zona+métrica en deterioro 3+ semanas | **{len(declining)}** |",
        f"| Pares de métricas con correlación fuerte (r≥0.6) | **{len(corr)}** |",
        f"| Métrica con más caídas esta semana | **{top_anomaly_metric}** |",
        f"| Zona con mayor crecimiento de órdenes (5 sem) | **{top_growth_zone}** |",
        "",
    ]

    # ── 2. Anomalías ─────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Anomalías — Cambios >10% semana a semana (L1W → L0W)",
        "",
    ]

    if anomalies.empty:
        lines.append("*No se detectaron anomalías.*\n")
    else:
        # Separar caídas y mejoras
        caidas  = anomalies[anomalies["direction"] == "caida"].head(15)
        mejoras = anomalies[anomalies["direction"] == "mejora"].head(10)

        if not caidas.empty:
            lines += [
                f"### Caídas más fuertes ({len(anomalies[anomalies['direction']=='caida'])} en total)",
                "",
                caidas[["COUNTRY", "CITY", "ZONE", "METRIC", "L1W_ROLL", "L0W_ROLL", "change_pct"]]
                .rename(columns={"change_pct": "cambio_%", "L1W_ROLL": "semana_ant", "L0W_ROLL": "semana_act"})
                .to_markdown(index=False),
                "",
            ]
        if not mejoras.empty:
            lines += [
                f"### Mejoras más fuertes ({len(anomalies[anomalies['direction']=='mejora'])} en total)",
                "",
                mejoras[["COUNTRY", "CITY", "ZONE", "METRIC", "L1W_ROLL", "L0W_ROLL", "change_pct"]]
                .rename(columns={"change_pct": "cambio_%", "L1W_ROLL": "semana_ant", "L0W_ROLL": "semana_act"})
                .to_markdown(index=False),
                "",
            ]

    # ── 3. Tendencias preocupantes ───────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Tendencias Preocupantes — Deterioro 3+ semanas consecutivas",
        "",
    ]

    if declining.empty:
        lines.append("*No se detectaron tendencias declinantes.*\n")
    else:
        top_decl = declining.head(20)
        lines += [
            f"*{len(declining)} combinaciones zona+métrica en deterioro continuo. "
            f"Mostrando las 20 con mayor caída acumulada.*",
            "",
            top_decl[["COUNTRY", "CITY", "ZONE", "METRIC", "val_start", "val_end", "change_pct", "weeks_declining"]]
            .rename(columns={
                "val_start": "valor_inicio",
                "val_end": "valor_fin",
                "change_pct": "caida_%",
                "weeks_declining": "semanas_caida",
            })
            .to_markdown(index=False),
            "",
        ]

    # ── 4. Correlaciones ─────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Correlaciones entre Métricas (r ≥ 0.6)",
        "",
    ]

    if corr.empty:
        lines.append("*No se encontraron correlaciones fuertes.*\n")
    else:
        lines += [
            "*Pares de métricas que se mueven juntas — útil para priorizar intervenciones.*",
            "",
            corr.to_markdown(index=False),
            "",
            "**Interpretación:** una correlación positiva fuerte sugiere que mejorar una métrica "
            "puede impactar positivamente en la otra.",
            "",
        ]

    # ── 5. Oportunidades ─────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Oportunidades — Zonas con Mayor Crecimiento de Órdenes (5 semanas)",
        "",
    ]

    if growth.empty:
        lines.append("*No hay datos de órdenes suficientes.*\n")
    else:
        top_growth = growth.head(15)
        col_start = [c for c in top_growth.columns if c.startswith("orders_L") and c != "orders_L0W"][0]
        lines += [
            "*Zonas con tendencia positiva de demanda — candidatas a mayor inversión operacional.*",
            "",
            top_growth[["COUNTRY", "CITY", "ZONE", col_start, "orders_L0W", "growth_abs", "growth_pct"]]
            .rename(columns={
                col_start: "ordenes_inicio",
                "orders_L0W": "ordenes_actuales",
                "growth_abs": "crecimiento_abs",
                "growth_pct": "crecimiento_%",
            })
            .to_markdown(index=False),
            "",
        ]

    # ── Recomendaciones ──────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Recomendaciones",
        "",
        f"1. **Atención inmediata:** revisar las {min(n_caidas, 10)} zonas con mayor caída "
        f"en **{top_anomaly_metric}** — cambios >10% en una semana suelen indicar problemas operacionales.",
        "",
        f"2. **Tendencias:** las {min(len(declining), 5)} zonas con deterioro más prolongado "
        "merecen análisis de causa raíz (logística, competencia, estacionalidad).",
        "",
        "3. **Correlaciones:** priorizar mejoras en las métricas con alta correlación — "
        "el impacto se multiplica en varias dimensiones.",
        "",
        f"4. **Inversión:** la zona **{top_growth_zone}** y sus pares con crecimiento acelerado "
        "de órdenes son candidatas a mayor disponibilidad de couriers y catálogo.",
        "",
        "---",
        "*Reporte generado automáticamente por el sistema de análisis de Rappi.*",
    ]

    return "\n".join(lines)
