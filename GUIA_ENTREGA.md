# Guía de entrega del proyecto final

## Proyecto

Predicción de escalamiento en interacciones de servicio al cliente.

## Autor

Luis Rivas

## Archivos principales

### 1. Dashboard

Archivo principal:

`app.py`

Permite:

- filtrar por canal;
- filtrar por tipo de cliente;
- filtrar por tipo de caso;
- filtrar por sentimiento;
- modificar el umbral de clasificación;
- visualizar KPI;
- analizar la tasa de escalamiento;
- observar la distribución de probabilidades;
- consultar la matriz de confusión;
- comparar modelos;
- analizar importancia de variables;
- consultar las interacciones con mayor probabilidad estimada.

### 2. Entrenamiento de modelos

Archivo:

`entrenar_modelo.py`

Genera los modelos y produce los archivos de salida utilizados por el dashboard.

### 3. Datos

Archivo principal:

`data/dataset meta.xlsx`

Hoja:

`Base de Datos`

### 4. Resultados del modelado

Carpeta:

`outputs/`

Contiene:

- `comparacion_modelos.csv`
- `predicciones_prueba.csv`
- `importancia_variables.csv`
- `resumen_modelo.json`

### 5. Modelo serializado

Archivo:

`models/modelo_escalamiento.joblib`

### 6. Caso de estudio

Archivo:

`CASO_ESTUDIO.md`

Describe:

- problemática;
- objetivo;
- unidad de análisis;
- variable objetivo;
- modelos evaluados;
- métricas;
- limitaciones;
- uso esperado del dashboard.

### 7. Documentación

Archivo:

`README.md`

Incluye:

- descripción del proyecto;
- estructura;
- instalación;
- ejecución;
- modelos;
- dashboard;
- limitaciones.

### 8. Publicación

Archivo:

`ENLACES.txt`

Debe contener:

- enlace al repositorio de GitHub;
- enlace al dashboard en Binder.

## Orden sugerido de ejecución

1. Instalar dependencias.
2. Ejecutar `entrenar_modelo.py`.
3. Verificar los archivos de `outputs/`.
4. Ejecutar `app.py`.
5. Validar filtros, gráficos, KPI y tabla.
6. Publicar el proyecto.
7. Verificar los enlaces.
8. Completar `ENLACES.txt`.

## Resultado general

El modelo de bosque aleatorio obtuvo el mayor ROC-AUC entre los modelos comparados, con un desempeño cercano a 0.51.

El resultado indica una capacidad discriminante limitada.

Por esta razón, el dashboard se utiliza como una herramienta exploratoria para analizar los resultados del modelado y no como un sistema para automatizar decisiones operativas.

## Lista de comprobación

- [x] Dataset incluido
- [x] Variable objetivo definida
- [x] Modelo de línea base
- [x] Regresión logística regularizada
- [x] Random Forest
- [x] Validación cruzada
- [x] Comparación de métricas
- [x] Importancia de variables
- [x] Archivos de salida generados
- [x] Dashboard interactivo
- [x] Filtros funcionales
- [x] Umbral interactivo
- [x] Matriz de confusión
- [x] Comparación de modelos
- [x] Tabla de interacciones
- [x] Documentación del caso
- [x] README
- [x] Informe final
- [ ] Repositorio GitHub
- [ ] Enlace Binder
- [ ] ENLACES.txt final
- [ ] ZIP de entrega
