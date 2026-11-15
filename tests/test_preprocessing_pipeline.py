"""Pruebas de fitear_reglas_de_variables y transform_variables con una tabla sintética."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pipeline_sklearn
import seleccion_de_variables
from preprocessing_pipeline import (CARPETA_DATOS_POR_DEFECTO, completar_fase2, dividir_por_fecha, ejecutar_preprocesamiento,
                                    fitear_reglas_de_variables, transform_variables)
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


@pytest.mark.skipif(len(list(CARPETA_DATOS_POR_DEFECTO.glob("Totalizadores Planta *.xlsx"))) != 3,
                    reason="Faltan los Excel en data/ (no se versionan; ver README, sección Datos)")
def test_el_punto_de_entrada_da_la_tabla_con_todas_las_variables():
    """Prueba de punta a punta con los datos reales (tarda ~30 s con la copia en data/interim/; ~10 min la primera vez)."""
    resultado = ejecutar_preprocesamiento()
    assert resultado["tabla_con_variables"].shape == (1212, 337)
    assert [len(resultado[k]) for k in ("entrenamiento", "validacion", "prueba")] == [848, 182, 182]
    assert set(resultado["reglas"]) == {"limpieza", "variables"}


# ----------------------------------------------------------------------------------------------------
# completar_fase2: selección, Pipeline y guardado, con una tabla chica y la selección reemplazada por una lista fija.

def resultado_de_cierre(dias=100):
    """Tabla de 100 días (70 de entrenamiento, 15 de validación, 15 de prueba) con dos variables y su consumo."""
    consumo = 1000 + 10.0 * np.arange(dias)
    tabla = pd.DataFrame({"DIA": pd.date_range("2026-01-01", periods=dias), "consumo_dia": consumo,
                          "x": consumo / 2, "y": np.arange(dias) % 7 + 0.5})
    tabla.loc[[3, 80], "x"] = np.nan                          # un vacío en entrenamiento y otro en prueba
    tabla["objetivo"] = tabla["consumo_dia"].shift(-1)
    return {"tabla_con_variables": tabla, "reglas": {"limpieza": {"a": 1}, "variables": {"b": [2]}}}


@pytest.fixture
def seleccion_de_dos_variables(monkeypatch):
    """Reemplaza la lista oficial de 14 variables por ["x", "y"] para poder usar una tabla chica."""
    monkeypatch.setattr(seleccion_de_variables, "seleccionar_variables", lambda X, y: ["x", "y"])


def test_el_dataset_final_no_trae_las_filas_de_frontera_y_queda_sin_escalar(tmp_path, seleccion_de_dos_variables):
    resultado = resultado_de_cierre()
    cierre = completar_fase2(resultado, tmp_path / "processed")
    guardado = pd.read_csv(cierre["rutas"]["dataset"], parse_dates=["DIA"])
    entrenamiento, validacion, prueba = dividir_por_fecha(resultado["tabla_con_variables"])
    assert list(guardado.columns) == ["DIA", "x", "y", "consumo_dia", "objetivo", "tramo"]
    assert entrenamiento["DIA"].iloc[-1] not in set(guardado["DIA"])      # su etiqueta es del primer día de validación
    assert validacion["DIA"].iloc[-1] not in set(guardado["DIA"])         # y la de esta, del primer día de prueba
    assert guardado["tramo"].value_counts().to_dict() == {"entrenamiento": 69, "validacion": 14, "prueba": 14}
    original = resultado["tabla_con_variables"].set_index("DIA").loc[guardado["DIA"]]
    np.testing.assert_allclose(guardado["consumo_dia"], original["consumo_dia"])   # mismas unidades que la tabla
    np.testing.assert_allclose(guardado["x"], original["x"])                       # los vacíos también se conservan


def test_las_shapes_y_el_linaje_coinciden_con_lo_guardado(tmp_path, seleccion_de_dos_variables):
    cierre = completar_fase2(resultado_de_cierre(), tmp_path / "processed")
    assert cierre["shapes"] == {"X_train": [69, 2], "X_val": [14, 2], "X_test": [14, 2]}
    assert cierre["forma_tras_el_pipeline"] == cierre["shapes"]            # la variante A no agrega columnas
    linaje = json.loads(cierre["rutas"]["linaje"].read_text(encoding="utf-8"))
    assert linaje["shapes"] == cierre["shapes"] and linaje["columnas"] == ["x", "y"]
    assert linaje["columnas_auxiliares"] == ["consumo_dia"]
    assert linaje["reglas"]["limpieza"] == {"a": 1} and "grupos" in linaje["reglas"]
    assert "processed/dataset_final.csv" in json.loads((tmp_path / "checksums.json").read_text(encoding="utf-8"))


def test_el_pipeline_se_ajusta_solo_con_entrenamiento(tmp_path, seleccion_de_dos_variables, monkeypatch):
    armados = []
    original = pipeline_sklearn.armar_pipeline

    def espia(*argumentos, **opciones):
        armados.append(original(*argumentos, **opciones))
        return armados[-1]

    monkeypatch.setattr(pipeline_sklearn, "armar_pipeline", espia)
    resultado = resultado_de_cierre()
    resultado["tabla_con_variables"].loc[70:, "x"] = 1e9                    # validación y prueba con valores enormes
    completar_fase2(resultado, tmp_path / "processed")
    imputador = armados[0].named_steps["preprocessor"].named_transformers_["numericas"].named_steps["imputar"]
    entrenamiento, _, _ = dividir_por_fecha(resultado["tabla_con_variables"])
    esperado = entrenamiento["x"].iloc[:-1].median()                        # sin la fila de frontera y sin vacíos
    assert imputador.statistics_[0] == pytest.approx(esperado)


def test_sin_guardar_no_escribe_ningun_archivo(tmp_path, seleccion_de_dos_variables):
    cierre = completar_fase2(resultado_de_cierre(), tmp_path / "processed", guardar=False)
    assert cierre["rutas"] == {} and not (tmp_path / "processed").exists()
    assert cierre["shapes"]["X_train"] == [69, 2]


def test_falla_con_un_mensaje_claro_si_falta_una_columna(tmp_path, seleccion_de_dos_variables):
    resultado = resultado_de_cierre()
    resultado["tabla_con_variables"] = resultado["tabla_con_variables"].drop(columns=["consumo_dia"])
    with pytest.raises(ValueError, match="consumo_dia"):
        completar_fase2(resultado, tmp_path / "processed")


def test_con_la_seleccion_oficial_avisa_que_variables_faltan(tmp_path):
    with pytest.raises(ValueError, match="Faltan variables definitivas"):
        completar_fase2(resultado_de_cierre(), tmp_path / "processed")
