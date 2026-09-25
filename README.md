# Proyecto final - Predicción de escalamiento en interacciones de servicio al cliente

## Descripción

Este proyecto corresponde al desarrollo de un pipeline de ciencia de datos orientado a analizar y predecir el escalamiento de interacciones de servicio al cliente.

El proyecto incluye:

- preparación y transformación de datos;
- entrenamiento de modelos de clasificación;
- comparación de métricas;
- análisis de importancia de variables;
- generación de archivos de resultados;
- dashboard interactivo desarrollado con Dash y Plotly;
- documentación del caso de estudio.

La variable objetivo es `escalated`, donde:

- `0` representa una interacción no escalada;
- `1` representa una interacción escalada.

---

## Objetivo

Evaluar si las variables operativas disponibles permiten estimar la probabilidad de escalamiento de una interacción de servicio al cliente.

También se busca comparar diferentes estrategias de modelado y presentar los resultados mediante un dashboard interactivo.

---

## Dataset

El conjunto de datos utilizado contiene 10.000 interacciones de servicio al cliente.

Archivo principal:

`data/dataset meta.xlsx`

Hoja utilizada:

`Base de Datos`

Entre las variables disponibles se encuentran:

- `interaction_id`
- `date`
- `time`
- `agent_id`
- `channel`
- `client_type`
- `issue_type`
- `fcr`
- `csat_score`
- `aht_seconds`
- `escalated`
- `ad_spend_recovered`
- `sentiment`

---

## Modelos evaluados

El proyecto compara tres modelos principales:

### 1. Línea base

Regresión logística utilizando `aht_seconds` como predictor principal.

### 2. Regresión logística regularizada

Incluye variables numéricas y categóricas, escalamiento, codificación One-Hot, transformación logarítmica y ajuste de hiperparámetros mediante validación cruzada.

### 3. Bosque aleatorio

Modelo Random Forest con ajuste de hiperparámetros mediante GridSearchCV.

El bosque aleatorio obtuvo el mayor ROC-AUC en el conjunto de prueba, aunque el desempeño general sigue siendo limitado.

---

## Métricas utilizadas

Se evalúan las siguientes métricas:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Debido al desbalance de clases, la accuracy no se utiliza como único criterio de evaluación.

---

## Estructura del proyecto

```text
Proyecto_Dashboard_Desercion_Academica/
│
├── app.py
├── entrenar_modelo.py
├── CASO_ESTUDIO.md
├── README.md
├── requirements.txt
├── environment.yml
├── ENLACES.txt
│
├── data/
│   ├── dataset meta.xlsx
│   ├── datos_estudiantes_ficticios.csv
│   └── diccionario_datos.csv
│
├── outputs/
│   ├── comparacion_modelos.csv
│   ├── predicciones_prueba.csv
│   ├── importancia_variables.csv
│   └── resumen_modelo.json
│
├── models/
│   └── modelo_escalamiento.joblib
│
├── assets/
│   └── style.css
│
├── tests/
│   └── test_pipeline.py
│
├── binder/
│   └── start
│
├── Procfile
├── ejecutar_dashboard.bat
└── ejecutar_dashboard.sh
