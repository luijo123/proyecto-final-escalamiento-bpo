"""Dashboard interactivo de predicción de escalamiento en servicio al cliente."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, Input, Output, dash_table, dcc, html

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


ROOT = Path(__file__).resolve().parent

DATA = pd.read_excel(
    ROOT / "data" / "dataset meta.xlsx",
    sheet_name="Base de Datos"
)

PRED = pd.read_csv(
    ROOT / "outputs" / "predicciones_prueba.csv"
)

MODELS = pd.read_csv(
    ROOT / "outputs" / "comparacion_modelos.csv"
)

IMPORTANCE = pd.read_csv(
    ROOT / "outputs" / "importancia_variables.csv"
).head(12)


binder_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")

if binder_prefix:
    requests_prefix = f"{binder_prefix}proxy/8050/"
else:
    requests_prefix = "/"

app = Dash(
    __name__,
    title="Predicción de escalamiento BPO",
    suppress_callback_exceptions=True,
    requests_pathname_prefix=requests_prefix
)

server = app.server


def card(title, value_id, note):
    return html.Div(
        [
            html.P(
                title,
                className="kpi-title"
            ),
            html.H2(
                id=value_id
            ),
            html.P(
                note,
                className="kpi-note"
            )
        ],
        className="kpi-card"
    )


app.layout = html.Div(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.P(
                            "CASO DE ESTUDIO BPO",
                            className="eyebrow"
                        ),
                        html.H1(
                            "Predicción de escalamiento en interacciones de servicio"
                        ),
                        html.P(
                            "Análisis del comportamiento de las interacciones, desempeño de los modelos y factores asociados al escalamiento.",
                            className="subtitle"
                        )
                    ]
                ),
                html.Div(
                    "DASH + PLOTLY",
                    className="badge"
                )
            ],
            className="hero"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "Filtros interactivos"
                        ),

                        html.Label(
                            "Canal"
                        ),

                        dcc.Dropdown(
                            sorted(
                                DATA["channel"]
                                .dropna()
                                .unique()
                            ),
                            multi=True,
                            id="channel",
                            placeholder="Todos los canales"
                        ),

                        html.Label(
                            "Tipo de cliente"
                        ),

                        dcc.Dropdown(
                            sorted(
                                DATA["client_type"]
                                .dropna()
                                .unique()
                            ),
                            multi=True,
                            id="client-type",
                            placeholder="Todos los tipos"
                        ),

                        html.Label(
                            "Tipo de caso"
                        ),

                        dcc.Dropdown(
                            sorted(
                                DATA["issue_type"]
                                .dropna()
                                .unique()
                            ),
                            multi=True,
                            id="issue-type",
                            placeholder="Todos los casos"
                        ),

                        html.Label(
                            "Sentimiento"
                        ),

                        dcc.Dropdown(
                            sorted(
                                DATA["sentiment"]
                                .dropna()
                                .unique()
                            ),
                            multi=True,
                            id="sentiment",
                            placeholder="Todos los sentimientos"
                        ),

                        html.Hr(),

                        html.Label(
                            "Umbral de clasificación"
                        ),

                        dcc.Slider(
                            0.10,
                            0.50,
                            0.01,
                            value=0.20,
                            marks={
                                0.10: "0.10",
                                0.20: "0.20",
                                0.30: "0.30",
                                0.40: "0.40",
                                0.50: "0.50"
                            },
                            id="umbral"
                        ),

                        html.P(
                            "El umbral modifica la clasificación de las probabilidades, pero no reentrena el modelo.",
                            className="help"
                        ),

                        html.Button(
                            "Restablecer filtros",
                            id="reset",
                            n_clicks=0
                        )
                    ],
                    className="sidebar"
                ),

                html.Main(
                    [
                        html.Div(
                            [
                                card(
                                    "Interacciones",
                                    "kpi-n",
                                    "población filtrada"
                                ),
                                card(
                                    "Escalamiento observado",
                                    "kpi-rate",
                                    "proporción de casos"
                                ),
                                card(
                                    "CSAT promedio",
                                    "kpi-csat",
                                    "puntuación media"
                                ),
                                card(
                                    "Casos sobre umbral",
                                    "kpi-high",
                                    "muestra de prueba"
                                )
                            ],
                            className="kpi-grid"
                        ),

                        html.Div(
                            [
                                dcc.Graph(
                                    id="grafico-canal"
                                ),
                                dcc.Graph(
                                    id="grafico-probabilidades"
                                )
                            ],
                            className="grid-2"
                        ),

                        html.Div(
                            [
                                dcc.Graph(
                                    id="grafico-confusion"
                                ),
                                dcc.Graph(
                                    id="grafico-modelos"
                                )
                            ],
                            className="grid-2"
                        ),

                        html.Div(
                            [
                                dcc.Graph(
                                    id="grafico-importancia"
                                )
                            ]
                        ),

                        html.Div(
                            [
                                html.H3(
                                    "Interpretación del modelo"
                                ),
                                html.Div(
                                    id="narrativa"
                                )
                            ],
                            className="narrative"
                        ),

                        html.Div(
                            [
                                html.H3(
                                    "Interacciones con mayor probabilidad de escalamiento"
                                ),

                                dash_table.DataTable(
                                    id="tabla-riesgo",
                                    columns=[
                                        {
                                            "name": "ID",
                                            "id": "interaction_id"
                                        },
                                        {
                                            "name": "Canal",
                                            "id": "channel"
                                        },
                                        {
                                            "name": "Tipo de cliente",
                                            "id": "client_type"
                                        },
                                        {
                                            "name": "Tipo de caso",
                                            "id": "issue_type"
                                        },
                                        {
                                            "name": "Sentimiento",
                                            "id": "sentiment"
                                        },
                                        {
                                            "name": "CSAT",
                                            "id": "csat_score"
                                        },
                                        {
                                            "name": "AHT",
                                            "id": "aht_seconds"
                                        },
                                        {
                                            "name": "Probabilidad",
                                            "id": "probabilidad_escalamiento"
                                        }
                                    ],
                                    page_size=10,
                                    sort_action="native",
                                    filter_action="native",
                                    style_table={
                                        "overflowX": "auto"
                                    },
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "8px",
                                        "fontFamily": "Arial"
                                    },
                                    style_header={
                                        "fontWeight": "bold"
                                    }
                                )
                            ],
                            className="narrative"
                        ),

                        html.Footer(
                            "Fuente: dataset de interacciones de servicio al cliente. Evaluación con conjunto de prueba estratificado."
                        )
                    ],
                    className="content"
                )
            ],
            className="shell"
        )
    ],
    className="app"
)


@app.callback(
    Output("channel", "value"),
    Output("client-type", "value"),
    Output("issue-type", "value"),
    Output("sentiment", "value"),
    Output("umbral", "value"),
    Input("reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(_):
    return [], [], [], [], 0.20


@app.callback(
    Output("kpi-n", "children"),
    Output("kpi-rate", "children"),
    Output("kpi-csat", "children"),
    Output("kpi-high", "children"),

    Output("grafico-canal", "figure"),
    Output("grafico-probabilidades", "figure"),
    Output("grafico-confusion", "figure"),
    Output("grafico-modelos", "figure"),
    Output("grafico-importancia", "figure"),

    Output("narrativa", "children"),
    Output("tabla-riesgo", "data"),

    Input("channel", "value"),
    Input("client-type", "value"),
    Input("issue-type", "value"),
    Input("sentiment", "value"),
    Input("umbral", "value")
)
def update_dashboard(
    channels,
    client_types,
    issue_types,
    sentiments,
    umbral
):

    d = DATA.copy()
    p = PRED.copy()

    if channels:
        d = d[
            d["channel"].isin(
                channels
            )
        ]

        p = p[
            p["channel"].isin(
                channels
            )
        ]

    if client_types:
        d = d[
            d["client_type"].isin(
                client_types
            )
        ]

        p = p[
            p["client_type"].isin(
                client_types
            )
        ]

    if issue_types:
        d = d[
            d["issue_type"].isin(
                issue_types
            )
        ]

        p = p[
            p["issue_type"].isin(
                issue_types
            )
        ]

    if sentiments:
        d = d[
            d["sentiment"].isin(
                sentiments
            )
        ]

        p = p[
            p["sentiment"].isin(
                sentiments
            )
        ]

    total = len(d)

    tasa_escalamiento = (
        d["escalated"].mean()
        if len(d)
        else np.nan
    )

    csat_promedio = (
        d["csat_score"].mean()
        if len(d)
        else np.nan
    )

    casos_umbral = (
        int(
            (
                p["probabilidad_escalamiento"]
                >= umbral
            ).sum()
        )
        if len(p)
        else 0
    )

    # ----------------------------
    # GRÁFICO 1: ESCALAMIENTO POR CANAL
    # ----------------------------

    if len(d):

        canal_df = (
            d.groupby(
                "channel",
                as_index=False
            )["escalated"]
            .mean()
        )

        canal_df[
            "tasa_escalamiento"
        ] = (
            canal_df["escalated"]
            * 100
        )

        fig_canal = px.bar(
            canal_df,
            x="channel",
            y="tasa_escalamiento",
            title="Tasa de escalamiento por canal",
            labels={
                "channel": "Canal",
                "tasa_escalamiento": "Escalamiento (%)"
            },
            text_auto=".1f"
        )

    else:
        fig_canal = go.Figure()

    # ----------------------------
    # GRÁFICO 2: DISTRIBUCIÓN DE PROBABILIDADES
    # ----------------------------

    if len(p):

        fig_prob = px.histogram(
            p,
            x="probabilidad_escalamiento",
            color="escalated",
            nbins=30,
            barmode="overlay",
            opacity=0.60,
            title="Distribución de probabilidades estimadas",
            labels={
                "probabilidad_escalamiento":
                    "Probabilidad de escalamiento",
                "escalated":
                    "Escalamiento real"
            }
        )

        fig_prob.add_vline(
            x=umbral,
            line_dash="dash",
            annotation_text=f"Umbral = {umbral:.2f}"
        )

    else:
        fig_prob = go.Figure()

    # ----------------------------
    # GRÁFICO 3: MATRIZ DE CONFUSIÓN
    # ----------------------------

    if (
        len(p)
        and p["escalated"].nunique() > 1
    ):

        pred_umbral = (
            p[
                "probabilidad_escalamiento"
            ]
            >= umbral
        ).astype(int)

        cm = confusion_matrix(
            p["escalated"],
            pred_umbral
        )

        fig_conf = px.imshow(
            cm,
            text_auto=True,
            x=[
                "Pred. No",
                "Pred. Sí"
            ],
            y=[
                "Real No",
                "Real Sí"
            ],
            title="Matriz de confusión"
        )

    else:
        fig_conf = go.Figure()

    # ----------------------------
    # GRÁFICO 4: COMPARACIÓN DE MODELOS
    # ----------------------------

    modelos_plot = MODELS[
        [
            "version",
            "roc_auc_prueba",
            "f1"
        ]
    ].melt(
        id_vars="version",
        value_vars=[
            "roc_auc_prueba",
            "f1"
        ],
        var_name="metrica",
        value_name="valor"
    )

    fig_modelos = px.bar(
        modelos_plot,
        x="version",
        y="valor",
        color="metrica",
        barmode="group",
        title="Comparación de modelos",
        labels={
            "version": "Modelo",
            "valor": "Resultado",
            "metrica": "Métrica"
        }
    )

    # ----------------------------
    # GRÁFICO 5: IMPORTANCIA DE VARIABLES
    # ----------------------------

    importancia_plot = (
        IMPORTANCE
        .sort_values(
            "importancia",
            ascending=True
        )
    )

    fig_importancia = px.bar(
        importancia_plot,
        x="importancia",
        y="variable",
        orientation="h",
        title="Importancia de variables",
        labels={
            "importancia": "Importancia",
            "variable": "Variable"
        }
    )

    # ----------------------------
    # MÉTRICAS Y NARRATIVA
    # ----------------------------

    if (
        len(p)
        and p["escalated"].nunique() > 1
    ):

        pred_umbral = (
            p[
                "probabilidad_escalamiento"
            ]
            >= umbral
        ).astype(int)

        accuracy = accuracy_score(
            p["escalated"],
            pred_umbral
        )

        precision = precision_score(
            p["escalated"],
            pred_umbral,
            zero_division=0
        )

        recall = recall_score(
            p["escalated"],
            pred_umbral,
            zero_division=0
        )

        f1 = f1_score(
            p["escalated"],
            pred_umbral,
            zero_division=0
        )

        auc = roc_auc_score(
            p["escalated"],
            p[
                "probabilidad_escalamiento"
            ]
        )

        narrativa = [
            html.P(
                f"Con un umbral de {umbral:.2f}, el modelo obtiene una exactitud de {accuracy:.3f}, precisión de {precision:.3f}, recall de {recall:.3f} y F1 de {f1:.3f}."
            ),
            html.P(
                f"El ROC-AUC en el subconjunto filtrado es {auc:.3f}."
            ),
            html.P(
                "Los resultados deben interpretarse con cautela, ya que la capacidad discriminante del modelo es limitada y las probabilidades de escalamiento presentan una separación reducida entre clases."
            )
        ]

    else:

        narrativa = html.P(
            "No hay suficientes clases diferentes en el subconjunto filtrado para calcular todas las métricas."
        )

    # ----------------------------
    # TABLA DE MAYOR PROBABILIDAD
    # ----------------------------

    if len(p):

        tabla_riesgo = (
            p[
                [
                    "interaction_id",
                    "channel",
                    "client_type",
                    "issue_type",
                    "sentiment",
                    "csat_score",
                    "aht_seconds",
                    "probabilidad_escalamiento"
                ]
            ]
            .sort_values(
                "probabilidad_escalamiento",
                ascending=False
            )
            .head(25)
            .copy()
        )

        tabla_riesgo[
            "probabilidad_escalamiento"
        ] = (
            tabla_riesgo[
                "probabilidad_escalamiento"
            ]
            .round(4)
        )

        tabla_data = (
            tabla_riesgo
            .to_dict("records")
        )

    else:

        tabla_data = []

    return (
        f"{total:,}",

        (
            f"{tasa_escalamiento:.1%}"
            if not np.isnan(
                tasa_escalamiento
            )
            else "--"
        ),

        (
            f"{csat_promedio:.2f}"
            if not np.isnan(
                csat_promedio
            )
            else "--"
        ),

        f"{casos_umbral:,}",

        fig_canal,
        fig_prob,
        fig_conf,
        fig_modelos,
        fig_importancia,
        narrativa,
        tabla_data
    )


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=8050
    )
