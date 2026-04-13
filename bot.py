"""
bot.py — Chatbot conversacional sobre métricas operacionales de Rappi.

Flujo por turno:
    1. LLM detecta intent + extrae params → JSON
    2. analytics.py ejecuta el cálculo (pandas, nunca el LLM)
    3. LLM redacta respuesta en español con el resultado como contexto
"""

import json
import os
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

import analytics as an
from data_loader import get_data_summary

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_ROWS_TO_LLM = 20       # Límite de filas al pasar DataFrames al LLM
INTENT_HISTORY_WINDOW = 6  # Mensajes recientes que ve el detector de intents

# ---------------------------------------------------------------------------
# Singletons inicializados una sola vez al importar el módulo
# ---------------------------------------------------------------------------
_client: Groq | None = None
_summary: dict = get_data_summary()


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY no encontrada. Agrega la clave en el archivo .env"
            )
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Construcción de system prompts (se calculan una sola vez al importar)
# ---------------------------------------------------------------------------

def _build_intent_system_prompt(summary: dict) -> str:
    metrics_list = "\n".join(f"  - {m}" for m in summary["metrics_list"])
    countries = ", ".join(summary["metrics_countries"])

    return f"""Eres un parser de intents para un chatbot analítico de Rappi.
Tu ÚNICA tarea es analizar el mensaje del usuario y retornar un JSON con el intent y los parámetros.
No respondas la pregunta. No agregues texto fuera del JSON.

## MÉTRICAS DISPONIBLES (copia el nombre EXACTAMENTE como aparece):
{metrics_list}

## PAÍSES: {countries}
Mapa de nombres en español → código:
  Argentina→AR, Brasil→BR, Chile→CL, Colombia→CO,
  Costa Rica→CR, Ecuador→EC, México→MX, Perú→PE, Uruguay→UY

## OTROS VALORES EXACTOS:
- ZONE_TYPE: "Wealthy" o "Non Wealthy"
- ZONE_PRIORITIZATION: "High Priority", "Prioritized", "Not Prioritized"
- Semanas (métricas): L8W_ROLL (más antigua) ... L0W_ROLL (más reciente, semana actual)

## INTENTS Y SUS SCHEMAS:

### top_k — Top o bottom N zonas por métrica en una semana
{{"intent": "top_k", "params": {{"metric": "<nombre exacto>", "n": 5, "country": null, "week": "L0W_ROLL"}}}}
· n positivo = mejores zonas. n negativo = peores zonas.
· Omitir country (null) si el usuario no especifica país.

### compare — Comparar distribución de una métrica entre grupos
{{"intent": "compare", "params": {{"metric": "<nombre>", "group_col": "ZONE_TYPE", "filters": null}}}}
· group_col opciones: ZONE_TYPE, COUNTRY, ZONE_PRIORITIZATION, CITY
· filters: objeto {{"COUNTRY": "MX"}} o null si no hay filtro.

### trend — Evolución temporal de una zona+métrica
{{"intent": "trend", "params": {{"zone": "<nombre exacto de zona>", "metric": "<nombre>", "n_weeks": 8}}}}
· n_weeks: cuántas semanas hacia atrás (1-8).

### aggregate — Agrupación / estadística por dimensión
{{"intent": "aggregate", "params": {{"metric": "<nombre>", "group_by": "COUNTRY", "agg_func": "mean"}}}}
· group_by: COUNTRY, CITY, ZONE_TYPE, ZONE_PRIORITIZATION
· agg_func: "mean", "median", "sum", "min", "max"

### multivariable_scan — Zonas con combinación contrastante de métricas
{{"intent": "multivariable_scan", "params": {{"metric_high": "<nombre>", "metric_low": "<nombre>", "threshold": 0.5}}}}
· threshold 0.5 = usar la mediana como corte. 0.7 = criterio más estricto.

### growth_inference — Crecimiento de órdenes entre semanas
{{"intent": "growth_inference", "params": {{"n_weeks": 5}}}}
· n_weeks: ventana de comparación (compara L0W vs L{{n_weeks}}W).

### benchmark — Zonas con desempeño divergente respecto a sus pares del mismo país
{{"intent": "benchmark", "params": {{"country": "<código>", "zone_type": null}}}}
· zone_type: "Wealthy", "Non Wealthy" o null para todos.

### problematic_zones — Zonas problemáticas / deterioradas
{{"intent": "problematic_zones", "params": {{"country": null, "threshold": 0.10, "min_weeks": 3}}}}
· Usar cuando el usuario dice "zonas problemáticas", "mal estado", "deterioradas", "con caídas", etc.

### correlations — Correlaciones entre métricas
{{"intent": "correlations", "params": {{"min_corr": 0.6}}}}

### follow_up — Pregunta de seguimiento que ajusta solo algún parámetro de la consulta anterior
{{"intent": "follow_up", "params": {{"<solo los parámetros que CAMBIAN>"}}}}
· Usar cuando la pregunta modifica implícitamente algo de la consulta anterior:
  "¿y en Colombia?", "¿y las últimas 3 semanas?", "¿y para Wealthy?", "dame 10 en vez de 5"
· En params incluye ÚNICAMENTE lo que cambia. El resto se hereda del contexto anterior.

### conversational — Pregunta sobre el historial de la conversación, sin necesidad de datos
{{"intent": "conversational", "params": {{}}}}
· Usar cuando el usuario pregunta sobre algo dicho antes o pide un resumen de la sesión:
  "¿cuál fue el país de la primera pregunta?", "¿qué métrica analizamos antes?",
  "¿me puedes resumir lo que vimos?", "¿de qué hablamos?"
· NO usar si la pregunta puede responderse con datos nuevos.

### unknown — No se puede determinar el intent con certeza
{{"intent": "unknown", "params": {{}}}}

## REGLAS IMPORTANTES:
1. Retorna ÚNICAMENTE JSON válido, sin texto adicional.
2. Nombres de métricas deben coincidir EXACTAMENTE con la lista (case-sensitive).
3. Si el usuario escribe una métrica aproximada, mapéala al nombre más cercano de la lista.
4. Para follow_up incluye SOLO los parámetros que cambian, no todo el contexto anterior.
5. Si el usuario pide "peores" zonas o "bottom", usa n negativo en top_k.
6. Si no puedes determinar el intent → unknown.
"""


def _build_response_system_prompt(summary: dict) -> str:
    metrics = ", ".join(summary["metrics_list"])
    countries = ", ".join(summary["metrics_countries"])

    return f"""Eres un analista de datos senior de Rappi especializado en métricas operacionales.

CONTEXTO DE LOS DATOS:
- {summary['metrics_rows']:,} registros de KPIs en {summary['metrics_unique_zones']} zonas operacionales
- {summary['metrics_unique_countries']} países: {countries}
- Métricas: {metrics}
- Datos de 9 semanas (L8W_ROLL = hace 8 semanas, L0W_ROLL = semana actual)

INSTRUCCIONES DE RESPUESTA:
- Responde SIEMPRE en español, de forma clara y concisa.
- Interpreta los datos y narra los hallazgos; no copies la tabla raw.
- Destaca los valores más extremos, patrones llamativos u outliers.
- Si hay oportunidades de mejora o alertas, menciónalas brevemente.
- Mantén coherencia con el historial de la conversación para dar continuidad.
- Si los datos están vacíos o hay error, explícalo amablemente y sugiere cómo reformular.
- Limita tu respuesta a lo esencial: máximo 3-4 párrafos salvo que se pida detalle explícito.
"""


# System prompts — se construyen una sola vez al importar el módulo
_INTENT_SYS = _build_intent_system_prompt(_summary)
_RESPONSE_SYS = _build_response_system_prompt(_summary)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _clean_messages(messages: list[dict]) -> list[dict]:
    """
    Filtra los mensajes para que solo contengan las claves 'role' y 'content'.
    Groq rechaza con 400 cualquier clave extra (e.g. 'chart_context' que guarda app.py).
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _result_to_str(result: Any) -> str:
    """Serializa el resultado de analytics a texto legible para el LLM."""
    if result is None:
        return "No se obtuvieron resultados."

    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "No se encontraron datos para los filtros aplicados."
        df = result.head(MAX_ROWS_TO_LLM)
        try:
            text = df.to_markdown(index=False)
        except Exception:
            text = df.to_string(index=False)
        if len(result) > MAX_ROWS_TO_LLM:
            text += f"\n\n_(Mostrando {MAX_ROWS_TO_LLM} de {len(result)} filas totales)_"
        return text

    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    return str(result)


def _detect_intent(
    user_message: str,
    messages: list[dict],
    last_context: dict,
) -> dict:
    """
    Primera llamada LLM: detecta el intent y extrae los parámetros.
    Incluye el historial reciente y el contexto previo para resolver follow-ups.
    Retorna dict con 'intent' y 'params', o {"intent": "unknown", "params": {}} ante errores.
    """
    context_hint = ""
    if last_context:
        context_hint = (
            f"\nContexto de la consulta anterior: {json.dumps(last_context, ensure_ascii=False)}"
            f"\nSi la pregunta actual es un seguimiento (ej: '¿y en México?', '¿y las últimas 3 semanas?'),"
            f" retorna intent='follow_up' y en params solo lo que cambia."
        )

    system = _INTENT_SYS + context_hint

    # Solo el mensaje actual — el context_hint ya provee lo necesario para follow_ups.
    # Incluir el historial completo infla el conteo de tokens y provoca errores 429
    # en el free tier de Groq (6000 TPM).
    detection_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    last_exc: Exception | None = None
    for attempt in range(2):  # un reintento ante errores transitorios (rate limit, etc.)
        try:
            resp = _get_client().chat.completions.create(
                model=MODEL,
                messages=detection_messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            print(f"[DEBUG] raw: {raw}")
            parsed = json.loads(raw)
            print(f"[DEBUG] parsed: {parsed}")
            if "intent" not in parsed:
                return {"intent": "unknown", "params": {}}
            parsed.setdefault("params", {})
            return parsed
        except json.JSONDecodeError as exc:
            print(f"[DEBUG] JSONDecodeError (intento {attempt + 1}): {exc} | raw={raw!r}")
            last_exc = exc
            break  # JSON roto no se arregla reintentando
        except Exception as exc:
            last_exc = exc
            print(f"[DEBUG] API error (intento {attempt + 1}): {type(exc).__name__}: {exc}")
            if attempt == 0:
                time.sleep(2)  # pausa breve antes del reintento (rate limit, timeout, etc.)

    print(f"[DEBUG] _detect_intent falló definitivamente: {last_exc}")
    return {"intent": "unknown", "params": {}}


def _execute_analytics(intent: str, params: dict) -> Any:
    """
    Mapea el intent a la función de analytics y la ejecuta con los params extraídos.
    Lanza excepciones — el caller las captura y genera un mensaje de error amigable.
    """
    p = params

    if intent == "top_k":
        return an.get_top_zones(
            metric=p["metric"],
            n=int(p.get("n", 5)),
            country=p.get("country") or None,
            week=p.get("week", "L0W_ROLL"),
        )

    if intent == "compare":
        return an.compare_groups(
            metric=p["metric"],
            group_col=p.get("group_col", "ZONE_TYPE"),
            filters=p.get("filters") or None,
        )

    if intent == "trend":
        return an.get_trend(
            zone=p["zone"],
            metric=p["metric"],
            n_weeks=int(p.get("n_weeks", 8)),
        )

    if intent == "aggregate":
        return an.aggregate_metric(
            metric=p["metric"],
            group_by=p.get("group_by", "COUNTRY"),
            agg_func=p.get("agg_func", "mean"),
        )

    if intent == "multivariable_scan":
        return an.multivariable_scan(
            metric_high=p["metric_high"],
            metric_low=p["metric_low"],
            threshold=float(p.get("threshold", 0.5)),
        )

    if intent == "growth_inference":
        return an.growth_analysis(n_weeks=int(p.get("n_weeks", 5)))

    if intent == "benchmark":
        return an.benchmark_analysis(
            country=p["country"],
            zone_type=p.get("zone_type") or None,
        )

    if intent == "problematic_zones":
        anomalies = an.detect_anomalies(threshold=float(p.get("threshold", 0.10)))
        declining = an.detect_declining_trends(min_weeks=int(p.get("min_weeks", 3)))

        country = p.get("country") or None
        if country:
            if not anomalies.empty:
                anomalies = anomalies[anomalies["COUNTRY"] == country]
            if not declining.empty:
                declining = declining[declining["COUNTRY"] == country]

        return {
            "total_anomalias": len(anomalies),
            "total_declinantes": len(declining),
            "anomalias_semana": (
                anomalies.head(MAX_ROWS_TO_LLM).to_dict(orient="records")
                if not anomalies.empty else []
            ),
            "tendencias_declinantes": (
                declining.head(MAX_ROWS_TO_LLM).to_dict(orient="records")
                if not declining.empty else []
            ),
        }

    if intent == "correlations":
        return an.detect_correlations(min_corr=float(p.get("min_corr", 0.6)))

    return None


def _generate_response(
    user_message: str,
    analytics_result: Any,
    messages: list[dict],
    client: Groq,
) -> str:
    """
    Segunda llamada LLM: genera la respuesta en español basada en el resultado analítico.
    Recibe el historial completo para dar continuidad a la conversación.
    """
    result_str = _result_to_str(analytics_result)

    user_content = (
        f"Pregunta del usuario: {user_message}\n\n"
        f"Resultado del análisis:\n{result_str}\n\n"
        "Interpreta estos datos y responde a la pregunta de forma clara y útil en español."
    )

    response_messages = [
        {"role": "system", "content": _RESPONSE_SYS},
        *_clean_messages(messages),
        {"role": "user", "content": user_content},
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=response_messages,
        temperature=0.4,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


def _generate_error_response(user_message: str, error: str, client: Groq) -> str:
    """Genera una respuesta amigable cuando analytics lanza una excepción."""
    msgs = [
        {"role": "system", "content": "Eres un asistente de datos de Rappi. Responde siempre en español."},
        {
            "role": "user",
            "content": (
                f"El usuario preguntó: '{user_message}'\n"
                f"Ocurrió un error al procesar la consulta: {error}\n"
                "Explica amablemente que no pudiste obtener los datos y sugiere "
                "cómo podría reformular la pregunta para obtener resultados."
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=msgs, temperature=0.3, max_tokens=256
    )
    return resp.choices[0].message.content.strip()


def _generate_conversational_response(
    user_message: str,
    messages: list[dict],
) -> str:
    """
    Responde preguntas sobre el historial de la conversación sin consultar analytics.
    El LLM recibe el historial completo y la pregunta — responde solo desde memoria.
    """
    response_messages = [
        {"role": "system", "content": _RESPONSE_SYS},
        *_clean_messages(messages),
        {"role": "user", "content": user_message},
    ]
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=response_messages,
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


_UNKNOWN_RESPONSE = (
    "Lo siento, no pude interpretar tu pregunta con claridad. "
    "Puedes preguntarme cosas como:\n\n"
    "- *¿Cuáles son las 5 zonas con mayor Lead Penetration en Colombia?*\n"
    "- *Muestra la evolución de Perfect Orders en Chapinero las últimas 8 semanas*\n"
    "- *¿Cuál es el promedio de Gross Profit UE por país?*\n"
    "- *¿Qué zonas tienen problemas en México?*\n"
    "- *¿Qué zonas tienen alto Lead Penetration pero bajo Perfect Orders?*\n\n"
    "¿Podrías reformular tu pregunta?"
)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def chat(
    user_message: str,
    messages: list[dict],
    last_context: dict,
) -> tuple[str, dict]:
    """
    Función principal del chatbot. Orquesta la detección de intent, ejecución
    analítica y generación de respuesta.

    Parameters
    ----------
    user_message : Mensaje en español del usuario (turno actual).
    messages     : Historial de la conversación [{role, content}, ...].
                   No debe incluir el turno actual — se agrega en app.py.
    last_context : Dict con el intent y params de la consulta anterior.
                   Permite resolver follow-ups sin repetir toda la pregunta.
                   Estructura esperada: {"intent": "...", "metric": "...", ...}

    Returns
    -------
    (respuesta_str, last_context_actualizado)
    """
    client = _get_client()

    # ── Paso 1: detectar intent ──────────────────────────────────────────────
    intent_result = _detect_intent(user_message, messages, last_context)
    intent = intent_result.get("intent", "unknown")
    params = intent_result.get("params", {})

    # ── Paso 2: resolver follow_up ───────────────────────────────────────────
    if intent == "follow_up":
        if not last_context or "intent" not in last_context:
            return _UNKNOWN_RESPONSE, last_context

        # Merge completo: heredar todo el contexto anterior y pisar solo lo nuevo
        prev_intent = last_context["intent"]
        merged_params = {k: v for k, v in last_context.items() if k != "intent"}
        merged_params.update(params)   # nuevos params sobrescriben los anteriores
        intent = prev_intent
        params = merged_params

    # ── Paso 3: manejar preguntas de memoria (sin analytics) ─────────────────
    if intent == "conversational":
        # La respuesta viene únicamente del historial de mensajes — no se toca analytics
        respuesta = _generate_conversational_response(user_message, messages)
        return respuesta, last_context  # last_context no cambia

    # ── Paso 4: manejar intent desconocido ───────────────────────────────────
    if intent == "unknown":
        return _UNKNOWN_RESPONSE, last_context

    # ── Paso 5: ejecutar analytics ───────────────────────────────────────────
    try:
        analytics_result = _execute_analytics(intent, params)
    except Exception as exc:
        respuesta = _generate_error_response(user_message, str(exc), client)
        return respuesta, last_context

    # ── Paso 6: generar respuesta en español ─────────────────────────────────
    respuesta = _generate_response(user_message, analytics_result, messages, client)

    # ── Paso 7: actualizar last_context ──────────────────────────────────────
    updated_context = {"intent": intent, **params}

    return respuesta, updated_context


# ---------------------------------------------------------------------------
# Smoke test interactivo al ejecutar directamente
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Bot test interactivo (escribe 'salir' para terminar) ===\n")
    history: list[dict] = []
    ctx: dict = {}

    while True:
        try:
            msg = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if msg.lower() in {"salir", "exit", "quit"}:
            break
        if not msg:
            continue

        respuesta, ctx = chat(msg, history, ctx)
        print(f"\nBot: {respuesta}\n")
        print(f"[ctx: intent={ctx.get('intent')}, params={dict(list(ctx.items())[:4])}]\n")

        # Actualizar historial
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": respuesta})
