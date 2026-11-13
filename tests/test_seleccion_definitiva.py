"""Regresiones del contrato oficial sin modelos, Validation ni Test."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import seleccion_definitiva as d
import seleccion_de_variables as s


def test_oficial_no_recalcula_ni_usa_etiquetas(monkeypatch):
    reglas = d.cargar_seleccion_definitiva()
    X = pd.DataFrame({c: [1., np.nan] for c in reglas["variables"]})
    def prohibido(*args, **kwargs):
        raise AssertionError("No debe recalcular selección")
    monkeypatch.setattr(s, "aprender_seleccion", prohibido)
    assert s.seleccionar_variables(X, object()) == reglas["variables"]
    assert d.preprocesar_conjunto_definitivo(X).shape == (2, 27)
    with pytest.raises(ValueError, match="Faltan"):
        s.seleccionar_variables(X.drop(columns=X.columns[0]), None)
    with pytest.raises(ValueError, match="únicas"):
        d.aplicar_conjunto_definitivo(pd.concat([X, X], axis=1))
    with pytest.raises(ValueError, match="ingeniería"):
        s.seleccionar_variables(X, None, reglas_variables={})


def test_huella_rechaza_lista_alterada(tmp_path):
    original = d.cargar_seleccion_definitiva()
    original["variables"].reverse()
    p = tmp_path / "seleccion_definitiva.json"
    p.write_text(json.dumps(original), encoding="utf-8")
    (tmp_path / "sha256.json").write_text(json.dumps({p.name: "incorrecta"}))
    with pytest.raises(ValueError, match="Huella"):
        d.cargar_seleccion_definitiva(p)


def test_causalidad_con_reglas_congeladas():
    reglas = d.cargar_seleccion_definitiva()
    cols = reglas["limpieza"]["columnas_conservadas"]
    tabla = pd.DataFrame({c: np.arange(1., 6.) for c in cols})
    tabla["DIA"] = pd.date_range("2022-01-01", periods=5)
    tabla["filas"] = 25
    tabla["valido"] = True
    tabla["consumo_dia"] = tabla["Frio (Kw)"]
    tabla["objetivo"] = 999999.
    a = d.reconstruir_conjunto_definitivo(tabla)
    otra = tabla.copy()
    otra.loc[3:, cols] = 1e12
    otra["objetivo"] = -999999.
    b = d.reconstruir_conjunto_definitivo(otra)
    pd.testing.assert_frame_equal(a.iloc[:3], b.iloc[:3])
    pd.testing.assert_frame_equal(a.iloc[:3], d.reconstruir_conjunto_definitivo(tabla.iloc[:3]))
    assert a["frio_hace_2_dias"].iloc[2] == 1.
    assert a["frio_media_7_dias"].iloc[2] == 2.
    assert a["Frio (Kw) x frio_hace_2_dias"].iloc[2] == 3.
    assert np.isfinite(d.preprocesar_conjunto_definitivo(a)).all()


def test_resultados_historicos_y_criterio():
    reglas = d.cargar_seleccion_definitiva()
    resultados = json.loads((d.ARTEFACTO.parent / "resultados.json").read_text())
    assert len(resultados["indices_validation"]) == 176
    assert len(set(resultados["indices_validation"])) == 176
    assert resultados["indices_validation"] == reglas["indices_validation"]
    tabla = {r["metodo"]: r for r in resultados["resultados"]}
    assert set(tabla) == {"spearman", "pearson", "mutua", "lasso", "forward"}
    for r in reglas["comparacion"]:
        assert r["mae"] == tabla[r["metodo"]]["mae"]
        assert r["filas"] == 176
    assert tabla["pearson"]["mae"] < tabla["spearman"]["mae"] <= 1.01 * tabla["pearson"]["mae"]
    assert tabla["spearman"]["preparacion"] == reglas["preprocesamiento"]


def test_contrato_337_columnas_con_transformacion_compartida_y_pipeline():
    from limpieza import COLUMNAS_DE_CONTROL
    from preprocessing_pipeline import transform_variables, dividir_por_fecha
    from pipeline_sklearn import separar_xy, armar_pipeline
    from variables_interacciones import construir_mapa_interacciones
    reglas = d.cargar_seleccion_definitiva()
    raiz = d.RAIZ / "results/seleccion/ejecucion-20261113T184058Z-969d4545/preparacion/final.json"
    originales = json.loads(raiz.read_text(encoding="utf-8"))
    columnas = originales["limpieza"]["columnas_conservadas"]
    tabla = pd.DataFrame({c: np.arange(1., 21.) for c in columnas})
    for c in COLUMNAS_DE_CONTROL:
        if c not in tabla: tabla[c] = 0.
    tabla["DIA"] = pd.date_range("2022-01-01", periods=20)
    tabla["objetivo"] = np.arange(20.)
    creada = transform_variables(tabla, reglas["ingenieria"])
    assert creada.shape == (20, 337)
    assert s.seleccionar_variables(creada, object()) == reglas["variables"]
    mapa = construir_mapa_interacciones(reglas["ingenieria"])
    for c in reglas["variables"]:
        assert set(mapa.get(c, [])).issubset(reglas["variables"])
    for tramo in dividir_por_fecha(creada):
        X, y = separar_xy(tramo, reglas["variables"])
        assert X.columns.tolist() == reglas["variables"]
        assert X.index.equals(y.index)
    pipeline = armar_pipeline(columnas=reglas["variables"])
    assert pipeline is not None  # solo construcción; no fit ni evaluación
