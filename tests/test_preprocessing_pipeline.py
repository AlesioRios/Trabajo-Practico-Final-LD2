"""Pruebas de fitear_reglas_de_variables y transform_variables con una tabla sintética."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing_pipeline import fitear_reglas_de_variables, transform_variables
from variables_interacciones import nombres_de_interacciones
from variables_planta import HL_ENVASADOS, NOMBRES_NUEVAS as NOMBRES_PLANTA, PLANTA_KW
from variables_temporales import NOMBRES_NUEVAS as NOMBRES_TEMPORALES

AIRE = "Consolidado Aire | Aire Producido (M3)"
GAS = "Consolidado GasVapor | Gas Planta (Mj)"
FILAS_ENTRENAMIENTO = 100


def tabla_sintetica(filas=150):
    """150 días seguidos desde el 01/01/2022, con las columnas que leen las variables y 8 más al azar."""
    azar = np.random.default_rng(42)
    frio = 20000 + np.cumsum(azar.normal(scale=300, size=filas))
    tabla = pd.DataFrame({
        "DIA": pd.date_range("2022-01-01", periods=filas),
        "Frio (Kw)": frio,
        PLANTA_KW: frio * 3 + azar.normal(scale=500, size=filas),
        "Servicios (Kw)": frio * 0.5 + azar.normal(scale=200, size=filas),
        HL_ENVASADOS: azar.uniform(5000, 15000, size=filas),
        AIRE: azar.uniform(1e5, 2e5, size=filas),
        GAS: azar.uniform(1e4, 3e4, size=filas)})
    for k in range(8):
        tabla[f"otra_{k}"] = frio * azar.uniform(0.1, 1) + azar.normal(scale=2000, size=filas)
    tabla["objetivo"] = tabla["Frio (Kw)"].shift(-1)
    return tabla


def test_las_reglas_tienen_planta_e_interacciones_y_se_pueden_guardar_en_json():
    reglas = fitear_reglas_de_variables(tabla_sintetica().iloc[:FILAS_ENTRENAMIENTO])
    assert set(reglas) == {"planta", "interacciones"}
    json.dumps(reglas)                                          # para el data_lineage.json


def test_transform_agrega_las_columnas_en_orden_sin_tocar_filas_ni_originales():
    tabla = tabla_sintetica()
    reglas = fitear_reglas_de_variables(tabla.iloc[:FILAS_ENTRENAMIENTO])
    resultado = transform_variables(tabla, reglas)
    base = reglas["interacciones"]["variables_base"]
    esperadas = list(tabla.columns) + NOMBRES_TEMPORALES + NOMBRES_PLANTA + nombres_de_interacciones(base)
    assert list(resultado.columns) == esperadas
    assert resultado.index.equals(tabla.index)
    assert resultado[tabla.columns].equals(tabla)


def test_transform_no_modifica_la_tabla_recibida():
    tabla = tabla_sintetica()
    copia = tabla.copy()
    transform_variables(tabla, fitear_reglas_de_variables(tabla.iloc[:FILAS_ENTRENAMIENTO]))
    assert tabla.equals(copia)


def test_fitear_es_determinista():
    entrenamiento = tabla_sintetica().iloc[:FILAS_ENTRENAMIENTO]
    assert fitear_reglas_de_variables(entrenamiento) == fitear_reglas_de_variables(entrenamiento)


def test_entrenamiento_da_lo_mismo_calculado_solo_o_dentro_de_la_tabla_completa():
    """Las variables miran solo hacia atrás: las filas de entrenamiento valen igual con o sin el futuro."""
    tabla = tabla_sintetica()
    reglas = fitear_reglas_de_variables(tabla.iloc[:FILAS_ENTRENAMIENTO])
    sola = transform_variables(tabla.iloc[:FILAS_ENTRENAMIENTO], reglas)
    dentro = transform_variables(tabla, reglas).iloc[:FILAS_ENTRENAMIENTO]
    pd.testing.assert_frame_equal(sola, dentro)


def test_cambiar_validacion_y_prueba_no_cambia_las_reglas_ni_el_entrenamiento():
    tabla = tabla_sintetica()
    reglas = fitear_reglas_de_variables(tabla.iloc[:FILAS_ENTRENAMIENTO])
    alterada = tabla.copy()
    alterada.loc[FILAS_ENTRENAMIENTO:, ["Frio (Kw)", PLANTA_KW, HL_ENVASADOS]] *= 50     # futuro muy distinto
    assert fitear_reglas_de_variables(alterada.iloc[:FILAS_ENTRENAMIENTO]) == reglas
    antes = transform_variables(tabla, reglas).iloc[:FILAS_ENTRENAMIENTO - 1]
    despues = transform_variables(alterada, reglas).iloc[:FILAS_ENTRENAMIENTO - 1]
    pd.testing.assert_frame_equal(antes, despues)
