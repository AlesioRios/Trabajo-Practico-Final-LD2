"""Pruebas aritméticas y de evidencia archivada; no ejecutan modelos."""

import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import diagnostico_xgboost as diagnostico


def tabla_chica():
    """Tres predicciones con un acierto y errores de signo opuesto."""
    return pd.DataFrame({"indice_csv": [3, 5, 7], "DIA": ["2026-01-01", "2026-01-01", "2026-01-03"],
                         "real": [10., 20., 30.], "predicho": [8., 20., 34.]})


def test_metricas_y_signo_del_residuo_calculables_a_mano():
    tabla = diagnostico.preparar_residuos(tabla_chica())
    assert tabla.residuo.tolist() == [2, 0, -4]
    assert tabla.error_absoluto.tolist() == [2, 0, 4]
    resumen = diagnostico.resumir_residuos(tabla)
    assert resumen["mae"] == pytest.approx(2)
    assert resumen["rmse"] == pytest.approx(np.sqrt(20 / 3))
    assert resumen["r2"] == pytest.approx(.9)
    assert resumen["sesgo_real_menos_predicho"] == pytest.approx(-2 / 3)
    assert resumen["subestimaciones"] == resumen["sobreestimaciones"] == 1


@pytest.mark.parametrize("indices", [[3, 3, 7], [7, 5, 3], [3, 5, 7.5]])
def test_rechaza_identidad_ambigua_o_desordenada(indices):
    tabla = tabla_chica()
    tabla["indice_csv"] = indices
    with pytest.raises(ValueError, match="indice_csv"):
        diagnostico.preparar_residuos(tabla)


@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_rechaza_predicciones_no_finitas(valor):
    tabla = tabla_chica()
    tabla.loc[0, "predicho"] = valor
    with pytest.raises(ValueError, match="finitas"):
        diagnostico.preparar_residuos(tabla)


def test_auditoria_detecta_log_alterado_sin_escribirlo(monkeypatch):
    lector = pd.read_csv

    def alterar_solo_en_memoria(ruta, **opciones):
        tabla = lector(ruta, **opciones)
        fila = tabla.index[tabla.modelo.eq("xgboost")][0]
        tabla.loc[fila, "hiperparametros"] = "{}"
        return tabla

    monkeypatch.setattr(diagnostico.pd, "read_csv", alterar_solo_en_memoria)
    with pytest.raises(ValueError, match="Hiperparámetros"):
        diagnostico.auditar_tracking(diagnostico.RAIZ)


def test_tracking_historico_completo_y_fuentes_intactas():
    filas, resumen = diagnostico.auditar_tracking(diagnostico.RAIZ)
    assert len(filas) == resumen["registros_xgboost"] == 139
    assert resumen["cv_completos"] == 137 and resumen["validaciones"] == 2
    assert resumen["cumplimiento_literal_por_ajuste"] == "Parcial"
    carpeta = diagnostico.RAIZ / diagnostico.CARPETA_SALIDA
    manifiesto = diagnostico.leer_json(carpeta / "verificacion.json")
    assert all(manifiesto["controles"].values())
    for ruta, huella in manifiesto["fuentes_sha256"].items():
        if ruta == "results/experiment_logs.csv":
            # El log es append-only y auditar_tracking ya valida todos los registros
            # XGBoost contra sus parámetros, predicciones y métricas originales.
            continue
        contenido = (diagnostico.RAIZ / ruta).read_bytes()
        firmas = {hashlib.sha256(contenido).hexdigest()}
        if ruta.endswith((".csv", ".json")):
            firmas.add(hashlib.sha256(contenido.replace(b"\r\n", b"\n")).hexdigest())
        assert huella in firmas, ruta
    for nombre in manifiesto["figuras"]:
        assert (carpeta / nombre).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_diagnostico_y_celdas_nuevas_sin_rutas_de_entrenamiento():
    fuentes = [Path(diagnostico.__file__).read_text(encoding="utf-8")]
    notebook = diagnostico.leer_json(diagnostico.RAIZ / "notebooks/modelado_xgboost.ipynb")
    fuentes.extend("".join(c["source"]) for c in notebook["cells"]
                   if c["cell_type"] == "code" and c.get("metadata", {}).get("diagnostico_offline"))
    assert len(fuentes) > 1
    for fuente in fuentes:
        arbol = ast.parse(fuente)
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Call):
                nombre = getattr(nodo.func, "attr", getattr(nodo.func, "id", ""))
                assert nombre not in {"fit", "fit_transform", "optimize", "predict", "load_model", "load_study"}
            if isinstance(nodo, (ast.Import, ast.ImportFrom)):
                nombres = [a.name for a in nodo.names] if isinstance(nodo, ast.Import) else [nodo.module or ""]
                assert all(n.split(".")[0] not in {"xgboost", "lightgbm", "sklearn", "optuna", "joblib",
                                                        "modelado_comun", "modelado_xgboost"} for n in nombres)


def test_residuos_exportados_reproducen_las_evaluaciones():
    referencia = None
    for nombre, (_, carpeta) in diagnostico.EXPERIMENTOS.items():
        tabla = pd.read_csv(diagnostico.RAIZ / diagnostico.CARPETA_SALIDA / f"residuos_{nombre}.csv")
        calculadas = diagnostico.metricas_guardadas(tabla.real, tabla.predicho.to_numpy(dtype=np.float32).astype(float))
        registro = diagnostico.leer_json(diagnostico.RAIZ / "results" / carpeta / "evaluacion.json")["registro"]
        diagnostico.comprobar_metricas(calculadas, registro)
        np.testing.assert_allclose(tabla.residuo, tabla.real - tabla.predicho)
        claves = tabla[["indice_csv", "DIA", "real"]]
        if referencia is not None:
            pd.testing.assert_frame_equal(claves, referencia)
        referencia = claves
