# Caso de estudio: Predicción de escalamiento en interacciones de servicio al cliente

## 1. Problemática

En operaciones de servicio al cliente, algunas interacciones requieren ser escaladas a niveles superiores de soporte. Identificar anticipadamente cuáles casos presentan mayor probabilidad de escalamiento puede apoyar el análisis operativo y permitir una mejor comprensión de los factores asociados a este comportamiento.

El proyecto busca evaluar si la información disponible en cada interacción contiene suficiente señal para diferenciar entre casos escalados y no escalados.

## 2. Unidad de análisis

Cada fila del conjunto de datos representa una interacción individual de servicio al cliente.

El conjunto contiene 10.000 registros y variables relacionadas con el canal de atención, tipo de cliente, tipo de solicitud, resolución en el primer contacto, satisfacción, tiempo de atención, sentimiento y otros indicadores operativos.

## 3. Variable objetivo

La variable objetivo es:

`escalated`

Su codificación es:

- `0`: la interacción no fue escalada.
- `1`: la interacción fue escalada.

Se trata, por lo tanto, de un problema de clasificación binaria.

## 4. Objetivo general

Desarrollar y evaluar modelos de clasificación que permitan estimar la probabilidad de que una interacción de servicio al cliente sea escalada, comparando distintas estrategias de modelado y presentando los resultados mediante un dashboard interactivo.

## 5. Objetivos específicos

- Analizar la distribución y características principales de las interacciones.
- Establecer un modelo de línea base para servir como referencia.
- Incorporar variables numéricas y categóricas al proceso de modelado.
- Aplicar escalamiento, codificación, balanceo de clases y transformación logarítmica.
- Evaluar diferentes configuraciones de regresión logística.
- Comparar el desempeño de un modelo lineal con un modelo no lineal de bosque aleatorio.
- Analizar accuracy, precision, recall, F1 y ROC-AUC.
- Identificar las variables con mayor importancia dentro del modelo seleccionado.
- Construir un dashboard interactivo que permita explorar los resultados mediante filtros y diferentes umbrales de clasificación.

## 6. Variables predictoras

Entre las principales variables utilizadas se encuentran:

- `channel`
- `client_type`
- `issue_type`
- `fcr`
- `csat_score`
- `aht_seconds`
- `ad_spend_recovered`
- `sentiment`

También se construyó:

`ad_spend_log = log(1 + ad_spend_recovered)`

como parte de la experimentación de transformación de variables.

## 7. Modelos evaluados

Se compararon tres enfoques principales:

1. Regresión logística de línea base utilizando AHT como predictor.
2. Regresión logística regularizada con múltiples variables, transformación logarítmica, balanceo de clases y optimización mediante validación cruzada.
3. Random Forest con ajuste de hiperparámetros mediante validación cruzada.

## 8. Métricas de evaluación

Se utilizaron:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Debido al desbalance de clases, la accuracy no se interpreta de manera aislada.

## 9. Error de mayor interés

Dentro del contexto del proyecto, un falso negativo representa una interacción que realmente fue escalada pero que el modelo clasificó como no escalada.

Este tipo de error es especialmente relevante porque reduce la capacidad de detectar anticipadamente situaciones que pueden requerir atención adicional.

Sin embargo, también debe considerarse el costo de los falsos positivos, ya que un modelo excesivamente sensible podría generar demasiadas alertas.

## 10. Resultado general

El modelo de bosque aleatorio obtuvo el mayor ROC-AUC entre las alternativas comparadas.

Sin embargo, el valor obtenido se encuentra cercano a 0.50 y las distribuciones de probabilidades para casos escalados y no escalados presentan un solapamiento considerable.

Por esta razón, el modelo se interpreta como una herramienta exploratoria y no como un sistema confiable para automatizar decisiones operativas.

## 11. Uso del dashboard

El dashboard permite:

- filtrar interacciones por canal;
- filtrar por tipo de cliente;
- filtrar por tipo de caso;
- filtrar por sentimiento;
- modificar el umbral de clasificación;
- observar indicadores generales;
- comparar modelos;
- consultar la matriz de confusión;
- analizar la distribución de probabilidades;
- revisar la importancia de variables;
- consultar las interacciones con mayor probabilidad estimada de escalamiento.

Modificar el umbral no reentrena el modelo. Únicamente cambia la forma en que las probabilidades estimadas se convierten en clases.

## 12. Limitaciones

Los modelos evaluados muestran una capacidad discriminante limitada.

Esto indica que las variables disponibles no contienen suficiente información para separar de forma clara los casos escalados de los no escalados.

La incorporación de nuevas variables operativas, históricas o relacionadas con el comportamiento del agente o del cliente podría mejorar futuros modelos.

Los resultados del proyecto no deben utilizarse para tomar decisiones automáticas sobre agentes o clientes.
