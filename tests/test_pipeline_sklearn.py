"""Pruebas de ajuste temporal, grupos y exportacion."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline_sklearn import (
    aprender_reglas_de_grupos, armar_pipeline, comparar_variantes, elegir_variante,
    escribir_linaje, evaluar_en_validacion, guardar_dataset_final, resumir_comparacion,
    separar_xy,
)
from variables_planta import NOMBRES_NUEVAS


def _tabla_diaria(dias, desde_cambio=None):
    """Tabla diaria chica: `x` es la mitad del consumo de hoy y `objetivo` el consumo de mañana.

    Si se da `desde_cambio`, el consumo de esos días en adelante se multiplica por 10 (el "futuro" cambia).
    """
    consumo = 1000 + 10.0 * np.arange(dias) + 30 * np.sin(np.arange(dias))
    if desde_cambio is not None:
        consumo[desde_cambio:] *= 10
    tabla = pd.DataFrame({"DIA": pd.date_range("2026-01-01", periods=dias),
                          "consumo_dia": consumo, "x": consumo / 2})
    tabla["objetivo"] = tabla["consumo_dia"].shift(-1)         # el consumo del día siguiente
    return tabla


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


def test_comparar_variantes_devuelve_una_fila_por_variante_alpha_y_pliegue():
    detalle, referencias = comparar_variantes(_tabla_diaria(60), ["x"], variantes=("A", "B"),
                                              alphas=(1, 10), pliegues=3)
    assert len(detalle) == 2 * 2 * 3                           # variantes x alphas x pliegues
    assert sorted(detalle["pliegue"].unique()) == [1, 2, 3]
    assert referencias["filas_ajuste"].is_monotonic_increasing  # cada pliegue ve más historia
    assert (referencias["inicio"].iloc[1:].values > referencias["fin"].iloc[:-1].values).all()  # sin solaparse


def test_el_futuro_no_cambia_el_error_de_los_pliegues_anteriores():
    base, _ = comparar_variantes(_tabla_diaria(60), ["x"], variantes=("A",), alphas=(1,), pliegues=3)
    cambiado, _ = comparar_variantes(_tabla_diaria(60, desde_cambio=50), ["x"], variantes=("A",),
                                     alphas=(1,), pliegues=3)
    assert_allclose(base["mae"].iloc[:2], cambiado["mae"].iloc[:2])      # pliegues 1 y 2 no se enteran
    assert base["mae"].iloc[2] != cambiado["mae"].iloc[2]                # el último sí ve el cambio


def test_resumir_y_elegir_variante_con_la_regla_de_un_desvio():
    detalle = pd.DataFrame({
        "variante": ["A"] * 4 + ["B"] * 4, "alpha": [1, 1, 10, 10] * 2, "pliegue": [1, 2] * 4,
        "mae": [110, 110, 120, 120, 90, 110, 100, 100],
    })
    resumen = resumir_comparacion(detalle)
    assert resumen["alpha"].to_dict() == {"A": 1, "B": 10}      # el alpha de menor MAE promedio por variante
    assert resumen["mae_promedio"].to_dict() == {"A": 110, "B": 100}
    variante, limite = elegir_variante(resumen)                 # B tiene desvío 0: el límite es 100 y A (110) queda afuera
    assert (variante, limite) == ("B", 100)
    resumen.loc["B", "desvio"] = 20                             # con desvío 20 el límite sube a 120 y gana la más simple
    assert elegir_variante(resumen) == ("A", 120)


def test_evaluar_en_validacion_usa_las_filas_con_baselines_y_calcula_la_mejora():
    tabla = _tabla_diaria(120)
    entrenamiento, validacion = tabla.iloc[:90], tabla.iloc[90:]
    resultado = evaluar_en_validacion(entrenamiento, validacion, ["x"], "A", 1)
    assert resultado["filas"] == 29                             # se descarta la última fila: su objetivo es del tramo siguiente
    mejor = min(resultado["persistencia"], resultado["media_movil"])
    assert_allclose(resultado["mejora_relativa"], 100 * (mejor - resultado["mae"]) / mejor)


def test_un_producto_con_el_coseno_no_es_ciclico():
    X = pd.DataFrame({"dia_anio_coseno": [-1, 0, 1, 0.5],
                      "dia_anio_coseno x Agua": [-500, 0, 800, 300], "medicion": [10, 20, 30, 40]})
    grupos = aprender_reglas_de_grupos(X, list(X))
    assert grupos["ciclicas"] == ["dia_anio_coseno"]
    assert grupos["numericas"] == ["dia_anio_coseno x Agua", "medicion"]
