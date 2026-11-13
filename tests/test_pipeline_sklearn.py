"""Pruebas de ajuste temporal, grupos y exportacion."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline_sklearn import (
    aprender_reglas_de_grupos, armar_pipeline, escribir_linaje,
    guardar_dataset_final, separar_xy,
)
from variables_planta import NOMBRES_NUEVAS


def test_mediana_de_entrenamiento_y_parametros_estables():
    X = pd.DataFrame({"medicion": [10, np.nan, 30, 50]})
    pre = armar_pipeline("A", ["medicion"]).named_steps["preprocessor"].fit(X)
    imputador = pre.named_transformers_["numericas"].named_steps["imputar"]
    futuro = pd.DataFrame({"medicion": [np.nan, 5000]})
    pre.transform(futuro)
    assert_allclose(imputador.statistics_, [30])
    assert_allclose(imputador.transform(futuro), [[30], [5000]])


def test_indicador_de_faltantes():
    X = pd.DataFrame({"medicion": [10, np.nan, 30, 40], "otra": [1, 2, 3, 4]})
    pre = armar_pipeline("C", list(X)).named_steps["preprocessor"].fit(X)
    imputador = pre.named_transformers_["numericas"].named_steps["imputar"]
    assert_allclose(imputador.transform(X), [[10, 1, 0], [30, 2, 1], [30, 3, 0], [40, 4, 0]])


def test_grupos_serializables_y_escalas_binaria_y_ciclica():
    razon = NOMBRES_NUEVAS[0]
    X = pd.DataFrame({
        "binaria": [0, 1, 0, 1], "dia_anio_seno": [-1, 0, 1, np.nan],
        razon: [2, 3, np.nan, 8], "medicion": [10, 20, 30, 40],
    })
    grupos = aprender_reglas_de_grupos(X, list(X))
    assert json.loads(json.dumps(grupos)) == {
        "binarias": ["binaria"], "ciclicas": ["dia_anio_seno"],
        "razones": [razon], "numericas": ["medicion"],
    }
    pre = armar_pipeline("E", list(X), grupos).named_steps["preprocessor"]
    assert_allclose(pre.fit_transform(X)[:, :2], [[0, -1], [1, 0], [0, 1], [1, 0]])


def test_separar_xy_conserva_indices_de_filas_etiquetadas():
    tabla = pd.DataFrame({"x": [9, 8, 7], "objetivo": [2, np.nan, 4]}, index=[3, 7, 9])
    X, y = separar_xy(tabla, ["x"])
    assert X["x"].to_dict() == {3: 9, 9: 7}
    assert y.to_dict() == {3: 2, 9: 4}


def test_guardado_conserva_escala_y_registra_huella_y_linaje(tmp_path):
    tabla = pd.DataFrame({
        "DIA": pd.date_range("2026-01-01", periods=5),
        "x": [10, 20, 30, 40, 50], "objetivo": [1, np.nan, 3, 4, 5],
    })
    salida = tmp_path / "processed"
    ruta = guardar_dataset_final(tabla, ["x"], salida)
    guardada = pd.read_csv(ruta)
    assert list(guardada.columns) == ["DIA", "x", "objetivo", "tramo"]
    assert guardada["x"].tolist() == [10, 30, 40, 50]
    assert guardada["tramo"].tolist() == ["entrenamiento", "entrenamiento", "validacion", "prueba"]
    assert "processed/dataset_final.csv" in json.loads((tmp_path / "checksums.json").read_text())
    reglas = {"grupos": {"numericas": ["x"]}}
    shapes = {"X_train": [2, 1], "X_val": [1, 1], "X_test": [1, 1]}
    linaje = json.loads(escribir_linaje(salida, reglas, ["x"], shapes).read_text(encoding="utf-8"))
    assert linaje["reglas"] == reglas and linaje["shapes"] == shapes
