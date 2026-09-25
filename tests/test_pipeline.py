
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_archivos_y_esquema():
    data = pd.read_excel(
        ROOT / "data" / "dataset meta.xlsx",
        sheet_name="Base de Datos"
    )

    pred = pd.read_csv(
        ROOT / "outputs" / "predicciones_prueba.csv"
    )

    results = pd.read_csv(
        ROOT / "outputs" / "comparacion_modelos.csv"
    )

    importance = pd.read_csv(
        ROOT / "outputs" / "importancia_variables.csv"
    )

    # Dataset principal
    assert len(data) == 10000
    assert data["interaction_id"].is_unique
    assert set(data["escalated"].dropna().unique()) <= {0, 1}
    assert data.isna().sum().sum() == 0

    # Predicciones
    assert pred["interaction_id"].is_unique
    assert pred["probabilidad_escalamiento"].between(0, 1).all()
    assert set(pred["escalated"].dropna().unique()) <= {0, 1}

    # Comparación de modelos
    assert len(results) == 3
    assert results["roc_auc_prueba"].between(0, 1).all()
    assert results["accuracy"].between(0, 1).all()
    assert results["precision"].between(0, 1).all()
    assert results["recall"].between(0, 1).all()
    assert results["f1"].between(0, 1).all()

    # Importancia
    assert "variable" in importance.columns
    assert "importancia" in importance.columns
    assert len(importance) > 0


def test_particion_sin_duplicados():
    pred = pd.read_csv(
        ROOT / "outputs" / "predicciones_prueba.csv"
    )

    assert pred["interaction_id"].is_unique
    assert len(pred) == 2500


def test_columnas_requeridas_predicciones():
    pred = pd.read_csv(
        ROOT / "outputs" / "predicciones_prueba.csv"
    )

    columnas_requeridas = {
        "interaction_id",
        "escalated",
        "channel",
        "client_type",
        "issue_type",
        "fcr",
        "csat_score",
        "aht_seconds",
        "ad_spend_recovered",
        "sentiment",
        "date",
        "probabilidad_escalamiento",
        "prediccion_050",
        "nivel_riesgo",
    }

    assert columnas_requeridas.issubset(
        set(pred.columns)
    )
