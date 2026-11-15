"""Verificación offline de evidencia LightGBM; no ejecuta estimadores."""

import ast
import builtins
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import diagnostico_lightgbm as diagnostico


def test_predicciones_residuos_y_metricas_son_los_archivados():
    referencia = None
    resumen = pd.read_csv(diagnostico.RAIZ / diagnostico.SALIDA / "resumen_residuos.csv")
    for nombre, carpeta in diagnostico.EXPERIMENTOS.items():
        fuente = pd.read_csv(diagnostico.RAIZ / "results" / carpeta / "predicciones_validacion.csv",
                             float_precision="round_trip")
        derivada = pd.read_csv(diagnostico.RAIZ / diagnostico.SALIDA / f"residuos_{nombre}.csv",
                               float_precision="round_trip")
        pd.testing.assert_frame_equal(fuente, derivada[list(fuente.columns)])
        assert len(derivada) == 171
        np.testing.assert_allclose(derivada.residuo, fuente.real - fuente.predicho, rtol=0, atol=1e-10)
        np.testing.assert_allclose(derivada.error_absoluto, derivada.residuo.abs(), rtol=0, atol=1e-10)
        calculadas = diagnostico.metricas_guardadas(derivada.real, derivada.predicho)
        evaluacion = diagnostico.leer_json(diagnostico.RAIZ / "results" / carpeta / "validacion.json")
        diagnostico.comprobar_metricas(calculadas, evaluacion["metricas"])
        diagnostico.comprobar_metricas(calculadas, resumen.loc[resumen.experimento.eq(nombre)].iloc[0])
        claves = fuente[["indice_csv", "DIA", "real"]]
        if referencia is not None:
            pd.testing.assert_frame_equal(referencia, claves)
        referencia = claves


@pytest.mark.parametrize("campo,valor", [("hiperparametros", "{}"), ("mae", -1),
                                         ("fecha_hora", "fecha inventada"), ("commit", "invalido")])
def test_auditoria_rechaza_alteraciones_del_log_en_memoria(monkeypatch, campo, valor):
    lector = pd.read_csv

    def alterar(ruta, **opciones):
        tabla = lector(ruta, **opciones)
        fila = tabla.index[tabla.modelo.eq("lightgbm")][0]
        tabla.loc[fila, campo] = valor
        return tabla

    monkeypatch.setattr(diagnostico.pd, "read_csv", alterar)
    with pytest.raises(ValueError):
        diagnostico.auditar_tracking()


def test_tracking_y_fuentes_preservadas():
    tabla, tracking = diagnostico.auditar_tracking()
    assert len(tabla) == tracking["registros_lightgbm"] == 86
    assert tracking["cv_completos"] == 84 and tracking["validaciones"] == 2
    assert tabla.arbol_sin_commitear.all()
    assert tracking["cumplimiento_literal_por_ajuste"] == "Parcial"
    salida = diagnostico.RAIZ / diagnostico.SALIDA
    manifiesto = diagnostico.leer_json(salida / "verificacion.json")
    assert all(manifiesto["controles"].values())
    assert tracking["registros_propios_sha256"] == manifiesto["tracking"]["registros_propios_sha256"]
    for ruta, firma in manifiesto["fuentes_sha256"].items():
        if ruta != "results/experiment_logs.csv":
            assert diagnostico.huella(diagnostico.RAIZ / ruta) == firma
    for nombre in manifiesto["figuras"]:
        assert (salida / nombre).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_integrar_otra_familia_no_invalida_los_registros_propios(monkeypatch):
    _, original = diagnostico.auditar_tracking()
    lector = pd.read_csv

    def agregar_otra_familia(ruta, **opciones):
        tabla = lector(ruta, **opciones)
        otra = tabla.iloc[[0]].copy()
        otra["modelo"] = "otra_familia"
        return pd.concat([tabla, otra], ignore_index=True)

    monkeypatch.setattr(diagnostico.pd, "read_csv", agregar_otra_familia)
    _, nuevo = diagnostico.auditar_tracking()
    assert original["registros_propios_sha256"] == nuevo["registros_propios_sha256"]


def test_registros_podados_de_otra_familia_no_cambian_la_firma(tmp_path):
    """El CSV compartido admite métricas ausentes sin cambiar tipos de LightGBM."""
    ruta = tmp_path / "log.csv"
    ruta.write_text("modelo,mae,rmse,r2\nlightgbm,1.5,2.0,0.5\notra,,,\n", encoding="utf-8")
    tabla = pd.read_csv(ruta, keep_default_na=False, na_values={k: [""] for k in diagnostico.METRICAS},
                        dtype={k: float for k in diagnostico.METRICAS})
    assert tabla.loc[0, "mae"] == 1.5 and np.isnan(tabla.loc[1, "mae"])


def test_rutas_offline_no_leen_datasets_ni_modelos(monkeypatch):
    abrir = builtins.open
    abrir_path = Path.open

    def controlar(ruta):
        if isinstance(ruta, (str, Path)):
            texto = str(ruta).replace("\\", "/").lower()
            assert "/data/" not in texto and "/models/" not in texto
            assert not texto.endswith((".joblib", ".pkl", ".sqlite3"))

    def seguro(ruta, *args, **kwargs):
        controlar(ruta)
        return abrir(ruta, *args, **kwargs)

    def seguro_path(ruta, *args, **kwargs):
        controlar(ruta)
        return abrir_path(ruta, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", seguro)
    monkeypatch.setattr(Path, "open", seguro_path)
    diagnostico.auditar_tracking()


def test_codigo_y_notebook_sin_ejecucion_de_modelos():
    raiz = diagnostico.RAIZ
    fuentes = [(raiz / "src" / nombre).read_text(encoding="utf-8") for nombre in
               ("diagnostico_lightgbm.py", "actualizar_notebook_lightgbm.py", "diagnostico_xgboost.py")]
    notebook = diagnostico.leer_json(raiz / "notebooks/modelado_lightgbm.ipynb")
    codigo = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(codigo) == 10 and all(c["execution_count"] is not None for c in codigo)
    assert not any(o.get("output_type") == "error" for c in codigo for o in c["outputs"])
    fuentes.extend("".join(c["source"]) for c in codigo)
    for fuente in fuentes:
        for nodo in ast.walk(ast.parse(fuente)):
            if isinstance(nodo, ast.Call):
                assert getattr(nodo.func, "attr", "") not in {
                    "fit", "fit_transform", "optimize", "predict", "load_model", "load_study"}
            if isinstance(nodo, (ast.Import, ast.ImportFrom)):
                modulos = [a.name for a in nodo.names] if isinstance(nodo, ast.Import) else [nodo.module or ""]
                assert all(m.split(".")[0] not in {"lightgbm", "xgboost", "optuna", "joblib", "sklearn",
                                                           "modelado_lightgbm", "modelado_comun"} for m in modulos)
