"""
app.py — Streamlit UI para el Rappi Intelligent Analysis System.

Estructura:
  · Sidebar  : estadísticas del dataset + limpiar conversación + preguntas de ejemplo
  · Tab Chat : historial conversacional + gráficos automáticos por intent
  · Tab Insights: reporte ejecutivo automático generado con insights.py
"""

import streamlit as st

# ── Page config (debe ser la primera llamada Streamlit) ──────────────────────
st.set_page_config(
    page_title="Rappi Analytics Bot",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports de la app ────────────────────────────────────────────────────────
from bot import chat
from charts import make_chart
from data_loader import get_data_summary
import insights as ins

# ── Session state — inicializar solo una vez ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

if "last_context" not in st.session_state:
    st.session_state.last_context: dict = {}

if "pending_message" not in st.session_state:
    st.session_state.pending_message: str | None = None

if "insights_report" not in st.session_state:
    st.session_state.insights_report: str | None = None

# ── Datos del dataset (cacheado) ─────────────────────────────────────────────
@st.cache_data
def _get_summary() -> dict:
    return get_data_summary()

summary = _get_summary()

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛵 Rappi Analytics")
    st.caption("Sistema de análisis de métricas operacionales")

    st.divider()

    # ── Estadísticas del dataset ─────────────────────────────────────────────
    st.subheader("Dataset")
    col1, col2 = st.columns(2)
    col1.metric("Zonas", f"{summary['metrics_unique_zones']:,}")
    col2.metric("Países", summary["metrics_unique_countries"])
    col1.metric("Métricas", summary["metrics_unique_metrics"])
    col2.metric("Semanas", len(summary["metrics_week_cols"]))

    with st.expander("Ver métricas disponibles"):
        for m in summary["metrics_list"]:
            st.caption(f"• {m}")

    with st.expander("Ver países"):
        st.caption("  ·  ".join(summary["metrics_countries"]))

    st.divider()

    # ── Limpiar conversación ─────────────────────────────────────────────────
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_context = {}
        st.session_state.pending_message = None
        st.session_state.insights_report = None
        st.rerun()

    st.divider()

    # ── Preguntas de ejemplo ─────────────────────────────────────────────────
    st.subheader("Preguntas de ejemplo")
    st.caption("Haz clic para enviar al chat automáticamente")

    EJEMPLOS = [
        "¿Cuáles son las 5 zonas con mayor Lead Penetration en Colombia?",
        "Muestra la evolución de Perfect Orders en Chapinero las últimas 8 semanas",
        "Compara Gross Profit UE entre Wealthy y Non Wealthy en México",
        "¿Cuáles son las zonas problemáticas en Argentina?",
        "¿Qué zonas tienen alto Lead Penetration pero bajo Perfect Orders?",
    ]

    for pregunta in EJEMPLOS:
        # Truncar etiqueta del botón para que quepa en el sidebar
        label = pregunta if len(pregunta) <= 55 else pregunta[:52] + "..."
        if st.button(label, use_container_width=True, key=f"btn_{pregunta[:20]}"):
            st.session_state.pending_message = pregunta
            # El click ya desencadena un rerun; el pending_message se procesa abajo

# ────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ────────────────────────────────────────────────────────────────────────────
tab_chat, tab_insights = st.tabs(["💬 Chat", "📊 Insights automáticos"])

# ════════════════════════════════════════════════════════════════════════════
# TAB CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # ── Resolver mensaje pendiente (botón de ejemplo) ────────────────────────
    # st.chat_input siempre retorna None en el mismo rerun que seteamos pending_message,
    # así que lo leemos aquí, antes del widget.
    prompt: str | None = None

    if st.session_state.pending_message:
        prompt = st.session_state.pending_message
        st.session_state.pending_message = None

    # ── Mostrar historial ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-mostrar gráfico guardado (solo el último turno del asistente)
            if msg["role"] == "assistant" and msg.get("chart_context"):
                ctx = msg["chart_context"]
                fig = make_chart(ctx.get("intent", ""), ctx)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"hist_{id(msg)}")

    # ── Input del usuario (fijo en la parte inferior) ────────────────────────
    user_input = st.chat_input("Pregunta sobre métricas de Rappi…")
    if user_input:
        prompt = user_input  # prioridad al input manual sobre pending

    # ── Procesar mensaje ─────────────────────────────────────────────────────
    if prompt:
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)

        # Guardar el historial ANTES del turno actual para pasarlo al bot
        history_before = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Llamar al bot y mostrar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Analizando datos…"):
                try:
                    respuesta, updated_ctx = chat(
                        prompt,
                        history_before,
                        st.session_state.last_context,
                    )
                    st.session_state.last_context = updated_ctx
                    error_ocurrido = False
                except Exception as exc:
                    st.error(f"Error al procesar la consulta: {exc}")
                    respuesta = (
                        "Lo siento, ocurrió un error al procesar tu consulta. "
                        "Por favor, intenta reformular tu pregunta."
                    )
                    updated_ctx = st.session_state.last_context
                    error_ocurrido = True

            # Mostrar respuesta en texto
            st.markdown(respuesta)

            # Gráfico automático según intent
            intent = updated_ctx.get("intent", "")
            chart_rendered = False
            if not error_ocurrido and intent in ("trend", "compare", "top_k", "multivariable_scan", "growth_inference"):
                try:
                    fig = make_chart(intent, updated_ctx)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        chart_rendered = True
                except Exception:
                    pass  # gráfico opcional — no crashear si falla

        # Persistir en historial (con contexto del gráfico para re-renderizar al recargar)
        assistant_entry: dict = {"role": "assistant", "content": respuesta}
        if chart_rendered:
            assistant_entry["chart_context"] = dict(updated_ctx)
        st.session_state.messages.append(assistant_entry)

# ════════════════════════════════════════════════════════════════════════════
# TAB INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.subheader("Reporte ejecutivo automático")
    st.caption(
        "Escaneo completo del dataset: anomalías, tendencias preocupantes, "
        "correlaciones y oportunidades de crecimiento."
    )

    col_btn, col_info = st.columns([2, 5])

    with col_btn:
        generar = st.button(
            "Generar reporte",
            type="primary",
            use_container_width=True,
        )

    with col_info:
        if st.session_state.insights_report:
            st.caption("Reporte listo. Haz clic en 'Generar reporte' para actualizar.")
        else:
            st.caption("El análisis puede tardar unos segundos.")

    if generar:
        with st.spinner("Analizando anomalías, tendencias, correlaciones y oportunidades…"):
            try:
                report = ins.generate_report()
                st.session_state.insights_report = report
            except Exception as exc:
                st.error(f"Error al generar el reporte: {exc}")
                st.session_state.insights_report = None

    if st.session_state.insights_report:
        st.divider()
        st.markdown(st.session_state.insights_report)
