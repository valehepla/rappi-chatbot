# Rappi Intelligent Analysis System

Bot conversacional con motor de insights automáticos sobre métricas operacionales de Rappi. Permite hacer preguntas en español sobre KPIs de 964 zonas en 9 países, obtener respuestas narrativas generadas por LLM y visualizar los datos con gráficos interactivos.

**Stack:** Python · Streamlit · Groq API (Llama 3.3 70B) · Pandas · Plotly

---

## Arquitectura

La regla central del sistema es que **el LLM nunca calcula — Pandas siempre calcula**.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Usuario (español)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ pregunta
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  bot.py — Llamada 1 al LLM (Groq / Llama 3.3 70B)             │
│                                                                 │
│  Tarea: detectar intent + extraer parámetros                    │
│  Salida: JSON  {"intent": "top_k",                              │
│                 "params": {"metric": "Lead Penetration",        │
│                            "n": 5, "country": "CO"}}           │
│                                                                 │
│  · temperature=0.0   · response_format: json_object            │
│  · Recibe las últimas 6 mensajes del historial                  │
│  · Si el intent es follow_up → merge con last_context           │
└───────────────────────────────┬─────────────────────────────────┘
                                │ intent + params
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  analytics.py — Cálculo puro con Pandas                        │
│                                                                 │
│  get_top_zones()          get_trend()         compare_groups()  │
│  aggregate_metric()       multivariable_scan() growth_analysis()│
│  benchmark_analysis()     detect_anomalies()                    │
│  detect_declining_trends() detect_correlations()               │
│                                                                 │
│  Fuente: data_loader.py → rappi_data.xlsx                       │
│  · df_metrics (wide)     · df_metrics_long (long)              │
│  · df_orders  (wide)     · df_orders_long  (long)              │
└───────────────────────────────┬─────────────────────────────────┘
                                │ DataFrame / dict con resultados
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  bot.py — Llamada 2 al LLM (Groq / Llama 3.3 70B)             │
│                                                                 │
│  Tarea: redactar respuesta en español                           │
│  Entrada: resultado de Pandas + pregunta original + historial   │
│  Salida: texto narrativo con interpretación y recomendaciones   │
│                                                                 │
│  · temperature=0.4   · max_tokens=1024                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ respuesta + last_context
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  app.py — Streamlit UI                                          │
│                                                                 │
│  · Muestra respuesta en burbuja del asistente                   │
│  · charts.py genera gráfico Plotly automático según intent      │
│  · Tab "Insights" → insights.py escanea anomalías y tendencias  │
└─────────────────────────────────────────────────────────────────┘
```

### Intents soportados

| Intent | Ejemplo de pregunta | Función analytics |
|---|---|---|
| `top_k` | ¿Cuáles son las 5 zonas con mayor Lead Penetration en Colombia? | `get_top_zones()` |
| `compare` | Compara Perfect Orders entre Wealthy y Non Wealthy en México | `compare_groups()` |
| `trend` | Evolución de Gross Profit UE en Chapinero las últimas 8 semanas | `get_trend()` |
| `aggregate` | ¿Cuál es el promedio de Lead Penetration por país? | `aggregate_metric()` |
| `multivariable_scan` | ¿Qué zonas tienen alto Lead Penetration pero bajo Perfect Orders? | `multivariable_scan()` |
| `growth_inference` | ¿Cuáles son las zonas que más crecen en órdenes? | `growth_analysis()` |
| `benchmark` | Compara zonas similares de Colombia con performance divergente | `benchmark_analysis()` |
| `problematic_zones` | ¿Cuáles son las zonas problemáticas en Argentina? | `detect_anomalies()` + `detect_declining_trends()` |
| `correlations` | ¿Qué métricas están correlacionadas entre sí? | `detect_correlations()` |
| `follow_up` | ¿Y en México? / ¿Y las últimas 3 semanas? | *(reutiliza intent anterior)* |

---

## Estructura del proyecto

```
rappi-bot/
├── app.py              # Streamlit UI — chat, gráficos, reporte de insights
├── bot.py              # Orquestador: intent detection → analytics → respuesta LLM
├── analytics.py        # Motor analítico puro (solo Pandas, sin LLM)
├── charts.py           # Gráficos Plotly por tipo de consulta
├── insights.py         # Reporte ejecutivo automático
├── data_loader.py      # Carga y limpieza del Excel, expone 4 DataFrames
├── data/
│   └── rappi_data.xlsx # Fuente de datos (no incluida en el repo)
├── .env                # GROQ_API_KEY — nunca commitear
├── requirements.txt
└── README.md
```

---

## Cómo obtener la API key de Groq

Groq ofrece un **tier gratuito** con acceso a Llama 3.3 70B sin costo.

1. Ir a [console.groq.com](https://console.groq.com)
2. Crear una cuenta (o iniciar sesión con Google/GitHub)
3. En el menú lateral, hacer clic en **API Keys**
4. Hacer clic en **Create API Key**, darle un nombre y copiar la clave generada
5. La clave tiene el formato `gsk_...`

> La clave es de un solo uso al mostrarla — guárdala inmediatamente.

---

## Instalación paso a paso

### Prerequisitos

- Python 3.11 o superior
- El archivo `data/rappi_data.xlsx` (provisto por separado)

### 1. Clonar o descargar el proyecto

```bash
git clone <url-del-repo>
cd rappi-bot
```

### 2. Crear y activar un entorno virtual (recomendado)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar la API key

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```
GROQ_API_KEY=gsk_tu_clave_aqui
```

> Nunca subas este archivo a un repositorio público. Ya está incluido en `.gitignore`.

### 5. Colocar el archivo de datos

Copiar `rappi_data.xlsx` dentro de la carpeta `data/`:

```
rappi-bot/
└── data/
    └── rappi_data.xlsx   ← aquí
```

---

## Cómo correr la app

```bash
streamlit run app.py
```

La app abre automáticamente en el navegador en `http://localhost:8501`.

Para especificar un puerto distinto:

```bash
streamlit run app.py --server.port 8080
```

---

## Ejemplos de preguntas

### Rankings y top zonas

```
¿Cuáles son las 5 zonas con mayor Lead Penetration en Colombia?
Dame las 10 zonas con peor Perfect Orders en México
Top 3 zonas por Gross Profit UE en Brasil
```

### Tendencias temporales

```
Muestra la evolución de Perfect Orders en Chapinero las últimas 8 semanas
¿Cómo fue el comportamiento de Turbo Adoption en Polanco las últimas 5 semanas?
Evolución de Lead Penetration en Miraflores desde hace 6 semanas
```

### Comparaciones entre grupos

```
Compara Gross Profit UE entre zonas Wealthy y Non Wealthy en Argentina
¿Cuál es la diferencia de Perfect Orders entre países?
Compara Pro Adoption entre zonas de alta y baja prioridad en Colombia
```

### Promedios y agregaciones

```
¿Cuál es el promedio de Lead Penetration por país?
Promedio de Gross Profit UE por tipo de zona en Chile
¿Qué país tiene el mayor Turbo Adoption en promedio?
```

### Análisis multivariable

```
¿Qué zonas tienen alto Lead Penetration pero bajo Perfect Orders?
Zonas con mucha adopción Pro pero poca penetración de lead
¿Dónde hay alto Gross Profit pero baja adopción de Turbo?
```

### Crecimiento de órdenes

```
¿Cuáles son las zonas que más crecieron en órdenes en las últimas 5 semanas?
¿Qué zonas perdieron más órdenes en las últimas 8 semanas?
Crecimiento de pedidos en Colombia en las últimas 3 semanas
```

### Zonas problemáticas y benchmarking

```
¿Cuáles son las zonas problemáticas en Argentina?
Muestra zonas con caídas fuertes en México esta semana
Compara zonas similares de Colombia con performance muy diferente
¿Qué zonas Wealthy de Perú tienen métricas divergentes?
```

### Correlaciones

```
¿Qué métricas están correlacionadas entre sí?
¿Existe relación entre Lead Penetration y Perfect Orders?
```

### Preguntas de seguimiento (follow-up)

El bot recuerda el contexto de la conversación, por lo que puedes hacer preguntas encadenadas:

```
Usuario: ¿Cuáles son las 5 zonas con mayor Lead Penetration en Colombia?
Bot:     [respuesta con top 5 de Colombia]

Usuario: ¿Y en México?
Bot:     [top 5 de México, sin necesidad de repetir la métrica]

Usuario: Dame 10 en vez de 5
Bot:     [top 10 de México con la misma métrica]
```

---

## Dataset

| Hoja | Filas | Descripción |
|---|---|---|
| RAW_INPUT_METRICS | 12,573 | KPIs operacionales por zona y semana |
| RAW_ORDERS | 1,242 | Volumen de órdenes por zona y semana |

- **9 países:** AR, BR, CL, CO, CR, EC, MX, PE, UY
- **964 zonas** operacionales
- **13 métricas:** Lead Penetration, Perfect Orders, Gross Profit UE, Pro Adoption, Turbo Adoption, MLTV Top Verticals Adoption, Non-Pro PTC > OP, % PRO Users Who Breakeven, % Restaurants Sessions With Optimal Assortment, Restaurants Markdowns / GMV, Restaurants SS > ATC CVR, Restaurants SST > SS CVR, Retail SST > SS CVR
- **9 semanas** de histórico (L8W = hace 8 semanas · L0W = semana actual)
