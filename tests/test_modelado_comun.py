"""Pruebas de las piezas comunes de la Fase 3: carga, filas comunes, baselines y registro de experimentos."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modelado_comun import (_estado_de_git, COLUMNAS_DEL_LOG, RUTA_DATASET_FINAL, RUTA_LINAJE, cargar_dataset_final,
                            calcular_metricas, evaluar_baselines, filas_comunes, registrar_baselines,
                            registrar_experimento, validacion_cruzada_temporal)


@pytest.fixture
def dataset_chico(tmp_path):
    """Dataset de 8 filas (4 de entrenamiento, 2 de validación y 2 de prueba) y su linaje, en una carpeta temporal."""
    tabla = pd.DataFrame({
        "DIA": pd.date_range("2026-01-01", periods=8), "a": np.arange(8.0), "b": np.arange(8.0) * 2,
        "consumo_dia": [10.0, 20, 30, np.nan, 50, 60, 70, 80],
        "baseline_media_movil": [np.nan, 15.0, 20, 25, 30, 40, np.nan, 60],
        "objetivo": [20.0, 30, 40, 50, 60, 70, 80, 90],
        "tramo": ["entrenamiento"] * 4 + ["validacion"] * 2 + ["prueba"] * 2})
    ruta = tmp_path / "dataset_final.csv"
    tabla.to_csv(ruta, index=False)
    linaje = {"columnas": ["a", "b"], "baselines": {"persistencia": "consumo_dia", "media_movil": "baseline_media_movil"}}
    (tmp_path / "data_lineage.json").write_text(json.dumps(linaje), encoding="utf-8")
    return ruta, tmp_path / "data_lineage.json"


def test_cargar_separa_los_tramos_y_no_incluye_la_prueba_por_defecto(dataset_chico):
    ruta, linaje = dataset_chico
    datos = cargar_dataset_final(ruta, linaje)
    assert list(datos) == ["entrenamiento", "validacion"]                    # la prueba no se carga si no se pide
    assert list(datos["entrenamiento"]["X"].columns) == ["a", "b"]           # los auxiliares no son entradas
    assert datos["validacion"]["y"].tolist() == [60, 70]
    assert list(datos["validacion"]["X"].index) == [4, 5]                    # índice = fila del CSV, único
    assert list(cargar_dataset_final(ruta, linaje, incluir_prueba=True)) == ["entrenamiento", "validacion", "prueba"]


def test_cargar_falla_con_un_mensaje_claro_si_falta_una_columna(dataset_chico):
    ruta, linaje = dataset_chico
    pd.read_csv(ruta).drop(columns=["baseline_media_movil"]).to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="baseline_media_movil"):
        cargar_dataset_final(ruta, linaje)


def test_filas_comunes_y_mae_de_los_baselines_a_mano(dataset_chico):
    ruta, linaje = dataset_chico
    entrenamiento = cargar_dataset_final(ruta, linaje)["entrenamiento"]
    # Fila 0 sin media móvil y fila 3 sin persistencia: quedan las filas 1 y 2
    assert filas_comunes(entrenamiento).tolist() == [False, True, True, False]
    resultado = evaluar_baselines(entrenamiento)
    assert resultado["filas"] == 2
    assert_allclose(resultado["persistencia"], np.mean([abs(30 - 20), abs(40 - 30)]))      # objetivo - consumo_dia
    assert_allclose(resultado["media_movil"], np.mean([abs(30 - 15), abs(40 - 20)]))


def test_las_metricas_se_calculan_a_mano():
    metricas = calcular_metricas([1, 2, 3], [2, 2, 5])
    assert_allclose(metricas["mae"], 1.0)                                    # (1 + 0 + 2) / 3
    assert_allclose(metricas["rmse"], np.sqrt(5 / 3))                        # (1 + 0 + 4) / 3
    assert_allclose(metricas["r2"], 1 - 5 / 2)                               # suma de residuos 5, suma total 2


@pytest.mark.parametrize("real, predicho", [([1, 2], [1]), ([1, np.nan], [1, 2]), ([], [])])
def test_las_metricas_rechazan_longitudes_distintas_o_vacios(real, predicho):
    with pytest.raises(ValueError):
        calcular_metricas(real, predicho)


def test_el_registro_crea_el_archivo_y_agrega_sin_pisar(tmp_path):
    log = tmp_path / "results" / "experiment_logs.csv"
    registrar_experimento("ridge", {"alpha": 10}, [1, 2, 3], [2, 2, 5], "validacion", ruta=log)
    registrar_experimento("xgboost", {"max_depth": 3, "n_estimators": 50}, [1, 2, 3], [1, 2, 3], "entrenamiento",
                          notas="prueba", ruta=log)
    registros = pd.read_csv(log)
    assert list(registros.columns) == COLUMNAS_DEL_LOG
    assert registros["modelo"].tolist() == ["ridge", "xgboost"]              # la segunda fila no pisó la primera
    assert registros["mae"].tolist() == [pytest.approx(1.0), pytest.approx(0.0)]
    assert json.loads(registros["hiperparametros"][1]) == {"max_depth": 3, "n_estimators": 50}
    assert (registros["semilla"] == 42).all() and registros["filas"].tolist() == [3, 3]


def test_el_registro_de_la_prueba_exige_confirmacion_explicita(tmp_path):
    log = tmp_path / "experiment_logs.csv"
    with pytest.raises(ValueError, match="prueba"):
        registrar_experimento("ridge", {}, [1, 2], [1, 2], "prueba", ruta=log)
    assert not log.exists()                                                  # no escribió nada
    registrar_experimento("ridge", {}, [1, 2], [1, 2], "prueba", ruta=log, evaluacion_final=True)
    assert len(pd.read_csv(log)) == 1


def test_el_registro_rechaza_un_archivo_con_otras_columnas(tmp_path):
    log = tmp_path / "experiment_logs.csv"
    log.write_text("modelo,mae\nridge,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="otras columnas"):
        registrar_experimento("ridge", {}, [1, 2], [1, 2], "validacion", ruta=log)


def test_registrar_baselines_deja_dos_filas_por_tramo_sobre_las_filas_comunes(dataset_chico, tmp_path):
    ruta, linaje = dataset_chico
    log = tmp_path / "experiment_logs.csv"
    registrar_baselines(cargar_dataset_final(ruta, linaje), ruta=log)
    registros = pd.read_csv(log)
    assert registros[["modelo", "tramo"]].values.tolist() == [
        ["persistencia", "entrenamiento"], ["media_movil", "entrenamiento"],
        ["persistencia", "validacion"], ["media_movil", "validacion"]]
    assert registros["filas"].tolist() == [2, 2, 2, 2]                       # solo las filas comunes de cada tramo
    assert_allclose(registros["mae"][:2], [10, 17.5])                        # entrenamiento: filas 1 y 2 (ver arriba)
    assert_allclose(registros["mae"][2:], [10, 30])                          # validación: |60-50|, |70-60| y |60-30|, |70-40|


def test_registrar_baselines_no_repite_filas_al_ejecutarse_otra_vez(dataset_chico, tmp_path):
    ruta, linaje = dataset_chico
    log, datos = tmp_path / "experiment_logs.csv", cargar_dataset_final(ruta, linaje)
    assert len(registrar_baselines(datos, ruta=log)) == 4
    assert registrar_baselines(datos, ruta=log) == []                       # la segunda vez no agrega nada
    assert len(pd.read_csv(log)) == 4


def test_los_pliegues_son_cronologicos_y_dejan_un_dia_de_separacion():
    X = pd.DataFrame({"a": np.arange(60.0)})
    pliegues = list(validacion_cruzada_temporal(pliegues=3).split(X))
    assert len(pliegues) == 3
    for ajuste, evaluacion in pliegues:
        assert ajuste.max() + 1 < evaluacion.min()                          # el gap deja libre al menos una fila
        assert (np.diff(evaluacion) == 1).all()                             # la evaluación es un bloque seguido
    assert pliegues[1][0].max() >= pliegues[0][1].max() - 1                 # cada pliegue ve más historia que el anterior


def test_el_estado_de_git_ignora_el_registro_pero_no_otros_cambios(tmp_path):
    def git(*argumentos):
        subprocess.run(["git", "-c", "user.name=prueba", "-c", "user.email=prueba@example.com", *argumentos],
                       cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "results").mkdir()
    (tmp_path / "codigo.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "results" / "experiment_logs.csv").write_text("a\n", encoding="utf-8")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "inicial")
    commit, sucio = _estado_de_git(tmp_path)
    assert len(commit) == 40 and sucio is False                              # todo commiteado
    (tmp_path / "results" / "experiment_logs.csv").write_text("a\nb\n", encoding="utf-8")
    assert _estado_de_git(tmp_path)[1] is False                              # agregar una fila al registro no cuenta
    (tmp_path / "codigo.py").write_text("x = 2\n", encoding="utf-8")
    assert _estado_de_git(tmp_path)[1] is True                               # editar el código sí


def test_el_dataset_final_versionado_cumple_el_contrato():
    """Comprueba el CSV real del repositorio (se versiona): tramos, columnas y baselines."""
    datos = cargar_dataset_final(incluir_prueba=True)
    linaje = json.loads(RUTA_LINAJE.read_text(encoding="utf-8"))
    assert {t: len(d["X"]) for t, d in datos.items()} == {"entrenamiento": 830, "validacion": 176, "prueba": 178}
    assert list(datos["entrenamiento"]["X"].columns) == linaje["columnas"] and len(linaje["columnas"]) == 14
    assert_allclose(datos["validacion"]["persistencia"].dropna(), datos["validacion"]["X"]["Frio (Kw)"].dropna())
    assert pd.read_csv(RUTA_DATASET_FINAL, parse_dates=["DIA"])["DIA"].is_monotonic_increasing
    # Con los baselines del dataset, las filas comunes de validación son las 171 que medimos en la Fase 2
    assert evaluar_baselines(datos["validacion"])["filas"] == 171
