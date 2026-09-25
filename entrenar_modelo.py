from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data" / "dataset meta.xlsx"
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

TARGET = "escalated"
ID = "interaction_id"

BASE_FEATURES = [
    "aht_seconds"
]

RAW_FEATURES = [
    "channel",
    "client_type",
    "issue_type",
    "fcr",
    "csat_score",
    "aht_seconds",
    "ad_spend_recovered",
    "sentiment"
]

ENGINEERED = [
    "ad_spend_log"
]

CATEGORICAL_FEATURES = [
    "channel",
    "client_type",
    "issue_type",
    "sentiment"
]


def agregar_caracteristicas(df):
    x = df.copy()

    x["ad_spend_log"] = np.log1p(
        x["ad_spend_recovered"]
    )

    return x


def cargar_datos():
    return pd.read_excel(
        DATA,
        sheet_name="Base de Datos"
    )


def pipeline_logistica(features):

    cat = [
        c for c in features
        if c in CATEGORICAL_FEATURES
    ]

    num = [
        c for c in features
        if c not in cat
    ]

    prep = ColumnTransformer([
        (
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]),
            num
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            cat
        )
    ])

    return Pipeline([
        ("prep", prep),
        (
            "model",
            LogisticRegression(
                max_iter=2500,
                random_state=42
            )
        )
    ])


def pipeline_bosque(features):

    cat = [
        c for c in features
        if c in CATEGORICAL_FEATURES
    ]

    num = [
        c for c in features
        if c not in cat
    ]

    prep = ColumnTransformer([
        (
            "num",
            SimpleImputer(
                strategy="median"
            ),
            num
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            cat
        )
    ])

    return Pipeline([
        ("prep", prep),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ])


def metricas(
    nombre,
    modelo,
    X_train,
    X_test,
    y_train,
    y_test,
    variables,
    regularizacion,
    params
):

    p_train = modelo.predict_proba(
        X_train
    )[:, 1]

    p_test = modelo.predict_proba(
        X_test
    )[:, 1]

    pred = (
        p_test >= 0.50
    ).astype(int)

    return {
        "version": nombre,
        "variables": variables,
        "regularizacion": regularizacion,
        "hiperparametros": json.dumps(
            params,
            ensure_ascii=False,
            default=str
        ),
        "roc_auc_entrenamiento": roc_auc_score(
            y_train,
            p_train
        ),
        "roc_auc_prueba": roc_auc_score(
            y_test,
            p_test
        ),
        "brecha_auc":
            roc_auc_score(
                y_train,
                p_train
            )
            -
            roc_auc_score(
                y_test,
                p_test
            ),
        "accuracy": accuracy_score(
            y_test,
            pred
        ),
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0
        )
    }, p_test


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = agregar_caracteristicas(
        cargar_datos()
    )

    idx_train, idx_test = train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        random_state=42,
        stratify=df[TARGET]
    )

    train = df.iloc[
        idx_train
    ].copy()

    test = df.iloc[
        idx_test
    ].copy()

    y_train = train[TARGET]
    y_test = test[TARGET]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    resultados = []
    predicciones = {}

    # MODELO 1: LINEA BASE

    base = pipeline_logistica(
        BASE_FEATURES
    )

    base.fit(
        train[BASE_FEATURES],
        y_train
    )

    row, p = metricas(
        "1. Linea base",
        base,
        train[BASE_FEATURES],
        test[BASE_FEATURES],
        y_train,
        y_test,
        "AHT como predictor único",
        "L2, C=1",
        {"C": 1.0}
    )

    resultados.append(row)
    predicciones[
        row["version"]
    ] = p

    # MODELO 2: LOGISTICA REGULARIZADA

    full_features = (
        RAW_FEATURES
        +
        ENGINEERED
    )

    log_search = GridSearchCV(
        pipeline_logistica(
            full_features
        ),
        {
            "model__C": [
                0.05,
                0.1,
                0.2,
                1.0,
                5.0
            ],
            "model__penalty": [
                "l1",
                "l2"
            ],
            "model__solver": [
                "liblinear"
            ],
            "model__class_weight": [
                None,
                "balanced"
            ]
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    log_search.fit(
        train[full_features],
        y_train
    )

    log_model = (
        log_search.best_estimator_
    )

    row, p = metricas(
        "2. Logistica regularizada",
        log_model,
        train[full_features],
        test[full_features],
        y_train,
        y_test,
        "Variables operativas + transformación log",
        f"{log_search.best_params_['model__penalty'].upper()}, C={log_search.best_params_['model__C']}",
        log_search.best_params_
    )

    resultados.append(row)
    predicciones[
        row["version"]
    ] = p

    # MODELO 3: RANDOM FOREST

    rf_search = GridSearchCV(
        pipeline_bosque(
            full_features
        ),
        {
            "model__n_estimators": [
                220
            ],
            "model__max_depth": [
                5,
                9,
                None
            ],
            "model__min_samples_leaf": [
                2,
                7
            ],
            "model__max_features": [
                "sqrt"
            ],
            "model__class_weight": [
                None,
                "balanced"
            ]
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    rf_search.fit(
        train[full_features],
        y_train
    )

    rf_model = (
        rf_search.best_estimator_
    )

    row, p = metricas(
        "3. Bosque ajustado",
        rf_model,
        train[full_features],
        test[full_features],
        y_train,
        y_test,
        "Variables operativas + transformación log",
        "Control por profundidad y hoja mínima",
        rf_search.best_params_
    )

    resultados.append(row)
    predicciones[
        row["version"]
    ] = p

    # COMPARACION

    resultados_df = (
        pd.DataFrame(
            resultados
        )
        .sort_values(
            "roc_auc_prueba",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    ganador = resultados_df.iloc[
        0
    ]["version"]

    modelos = {
        "1. Linea base": base,
        "2. Logistica regularizada":
            log_model,
        "3. Bosque ajustado":
            rf_model
    }

    best_model = modelos[
        ganador
    ]

    if ganador == "1. Linea base":
        best_features = (
            BASE_FEATURES
        )
    else:
        best_features = (
            full_features
        )

    # PREDICCIONES DEL MODELO GANADOR

    columnas_salida = [
        ID,
        TARGET,
        "channel",
        "client_type",
        "issue_type",
        "fcr",
        "csat_score",
        "aht_seconds",
        "ad_spend_recovered",
        "sentiment",
        "date"
    ]

    pred_test = test[
        columnas_salida
    ].copy()

    pred_test[
        "probabilidad_escalamiento"
    ] = predicciones[
        ganador
    ]

    pred_test[
        "prediccion_050"
    ] = (
        pred_test[
            "probabilidad_escalamiento"
        ]
        >= 0.50
    ).astype(int)

    pred_test[
        "nivel_riesgo"
    ] = pd.cut(
        pred_test[
            "probabilidad_escalamiento"
        ],
        [
            -0.01,
            0.30,
            0.60,
            1.0
        ],
        labels=[
            "Bajo",
            "Medio",
            "Alto"
        ]
    )

    # IMPORTANCIA POR PERMUTACION

    perm = permutation_importance(
        best_model,
        test[
            best_features
        ],
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=42,
        n_jobs=-1
    )

    importance = pd.DataFrame({
        "variable":
            best_features,
        "importancia":
            perm.importances_mean
    }).sort_values(
        "importancia",
        ascending=False
    )

    # EXPORTACION

    resultados_df.to_csv(
        OUT /
        "comparacion_modelos.csv",
        index=False
    )

    pred_test.to_csv(
        OUT /
        "predicciones_prueba.csv",
        index=False
    )

    importance.to_csv(
        OUT /
        "importancia_variables.csv",
        index=False
    )

    joblib.dump(
        {
            "modelo":
                best_model,
            "variables":
                best_features,
            "ganador":
                ganador
        },
        MODEL_DIR /
        "modelo_escalamiento.joblib"
    )

    resumen = {
        "modelo_seleccionado":
            ganador,
        "registros_totales":
            int(len(df)),
        "registros_prueba":
            int(len(test)),
        "tasa_escalamiento":
            float(
                df[TARGET].mean()
            ),
        "mejor_auc_prueba":
            float(
                resultados_df.iloc[
                    0
                ][
                    "roc_auc_prueba"
                ]
            ),
        "semilla":
            42,
        "criterio_seleccion":
            "Mayor ROC-AUC en conjunto de prueba, tras ajuste por validación cruzada en entrenamiento"
    }

    (
        OUT /
        "resumen_modelo.json"
    ).write_text(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            resumen,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
