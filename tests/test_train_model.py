"""Protecciones del cierre con datos sintéticos, sin consumir test del proyecto."""
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_model as cierre


def test_ajuste_no_convierte_observaciones_de_prueba(tmp_path):
    ruta = tmp_path / "datos.csv"
    ruta.write_text("DIA,a,objetivo,consumo_dia,baseline_media_movil,tramo\n"
                    "2026-01-01,1,2,1,1,entrenamiento\n"
                    "2026-01-02,3,4,3,2,validacion\n"
                    "no_fecha,secreto,secreto,secreto,secreto,prueba\n", encoding="utf-8")
    tabla = cierre.cargar_tramos(ruta, {"entrenamiento", "validacion"}, ["a"])
    assert tabla.index.tolist() == [0, 1]
    assert tabla.objetivo.tolist() == [2, 4]


def test_mediana_y_escalado_se_aprenden_solo_del_ajuste():
    modelo = cierre.fabricar_modelo(["a"])
    modelo.fit(pd.DataFrame({"a": [1., np.nan, 3., 5.]}), [2., 3., 4., 6.])
    pre = modelo.named_steps["preprocessor"].named_transformers_["numericas"]
    np.testing.assert_equal(pre.named_steps["imputar"].statistics_, [3.])
    modelo.predict(pd.DataFrame({"a": [100000., np.nan]}))
    np.testing.assert_equal(pre.named_steps["imputar"].statistics_, [3.])
    assert modelo.named_steps["regressor"].get_params()["alpha"] == 75


def test_versiones_historicas_no_se_pisan(tmp_path):
    (tmp_path / "modelo_v1.0.1.pkl").write_bytes(b"historia")
    assert cierre.siguiente_version({"1.0.0": {}}, tmp_path) == "1.0.2"


def preparar_evaluacion_sintetica(tmp_path):
    for carpeta in ["models", "data/processed", "results/lasso_final"]:
        (tmp_path / carpeta).mkdir(parents=True, exist_ok=True)
    datos = pd.DataFrame({"DIA": pd.date_range("2026-01-01", periods=5), "a": [1., 2., 3., 4., 5.],
        "objetivo": [2., 4., 6., 8., 10.], "consumo_dia": [1., 2., 3., np.nan, 5.],
        "baseline_media_movil": [1., 1.5, 2., 3., 4.], "tramo": ["entrenamiento", "validacion", "prueba", "prueba", "prueba"]})
    ruta = tmp_path / "data/processed/dataset_final.csv"
    datos.to_csv(ruta, index=False)
    modelo = cierre.fabricar_modelo(["a"]).fit(datos.loc[:1, ["a"]], datos.objetivo[:2])
    joblib.dump(modelo, tmp_path / "models/modelo_v1.0.0.pkl")
    meta = {"version": "1.0.0", "columnas": ["a"], "path": "models/modelo_v1.0.0.pkl",
        "modelo_sha256": cierre.huella(tmp_path / "models/modelo_v1.0.0.pkl"),
        "dataset_sha256": cierre.huella(ruta), "indices_ajuste": [0, 1], "ultima_fecha_ajuste": "2026-01-02"}
    cierre.guardar_json(tmp_path / "results/lasso_final/entrenamiento.json", meta)
    cierre.guardar_json(tmp_path / "models/model_registry.json", {"1.0.0": meta})
    (tmp_path / "results/experiment_logs.csv").write_text(
        ",".join(cierre.registrar_experimento.__globals__["COLUMNAS_DEL_LOG"]) + "\n", encoding="utf-8")
    return tmp_path / "results/lasso_final"


def test_test_unico_mismo_soporte_y_reconsulta_sin_predecir(tmp_path, monkeypatch):
    salida = preparar_evaluacion_sintetica(tmp_path)
    original = Pipeline.predict
    llamadas = []
    def contar(self, X, *args, **kwargs):
        llamadas.append(X.index.tolist())
        return original(self, X, *args, **kwargs)
    monkeypatch.setattr(Pipeline, "predict", contar)
    resultado = cierre.evaluar_test_una_vez(tmp_path)
    assert llamadas == [[2, 4]]
    assert resultado["filas_test_etiquetadas"] == 3
    assert resultado["filas_evaluadas"] == 2
    assert resultado["indices_excluidos"] == [3]
    assert cierre.evaluar_test_una_vez(tmp_path) == resultado
    assert llamadas == [[2, 4]]
    filas = list(csv.DictReader((tmp_path / "results/experiment_logs.csv").open()))
    assert len(filas) == 3 and all(r["filas"] == "2" and r["tramo"] == "prueba" for r in filas)
    assert cierre.leer_json(salida / "test_consumido.json")["estado"] == "completo_no_repetir"


def test_interrupcion_bloquea_repeticion_y_archivo_alterado_se_detecta(tmp_path):
    salida = preparar_evaluacion_sintetica(tmp_path)
    (salida / "test_consumido.json").write_text("{}")
    with pytest.raises(FileExistsError):
        cierre.evaluar_test_una_vez(tmp_path)


def test_contrato_del_ganador_conserva_nombres_y_orden():
    columnas = cierre.columnas_ganadoras()
    assert len(columnas) == 14
    assert columnas[0] == "Frio (Kw)"
    assert columnas[-1] == "Totalizadores Energia | KW Trafo 9"
    assert "objetivo" not in columnas and "es_fin_de_anio" not in columnas


def test_notebook_solo_lee_evidencia_sin_ejecutar_modelos():
    import ast
    notebook = json.loads((cierre.RAIZ / "notebooks/modelado.ipynb").read_text(encoding="utf-8"))
    for celda in notebook["cells"]:
        if celda["cell_type"] != "code":
            continue
        for nodo in ast.walk(ast.parse("".join(celda["source"]))):
            if isinstance(nodo, ast.Call):
                nombre = getattr(nodo.func, "attr", getattr(nodo.func, "id", ""))
                assert nombre not in {"fit", "fit_transform", "predict", "entrenar", "evaluar_test_una_vez", "cargar_tramos"}


def test_evidencia_final_si_existe_coincide_con_registry_y_log():
    salida = cierre.RAIZ / "results/lasso_final"
    if not (salida / "evaluacion_test.json").exists():
        pytest.skip("La evaluación única todavía no fue ejecutada.")
    resultado = cierre.validar_archivo_test(salida)
    meta = cierre.leer_json(salida / "entrenamiento.json")
    registry = cierre.leer_json(cierre.RAIZ / "models/model_registry.json")[meta["version"]]
    assert registry["metricas_test_local"] == resultado["metricas"]
    assert registry["metricas_validacion_original"] == cierre.VALIDACION
    assert cierre.huella(cierre.RAIZ / meta["path"]) == meta["modelo_sha256"]
    with (cierre.RAIZ / "results/experiment_logs.csv").open(encoding="utf-8") as archivo:
        filas = [r for r in csv.DictReader(archivo) if r["tramo"] == "prueba"]
    assert len(filas) == 3 and all(int(r["filas"]) == resultado["filas_evaluadas"] for r in filas)
    for r in filas:
        nombre = r["modelo"].split("_v")[0]
        for metrica in ["mae", "rmse", "r2"]:
            assert float(r[metrica]) == resultado["metricas"][nombre][metrica]
