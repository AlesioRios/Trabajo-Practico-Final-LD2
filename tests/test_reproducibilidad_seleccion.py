"""Regresiones de compatibilidad preparadas sin ejecución durante la auditoría.

Los Excel de caché son archivos ficticios: se sustituye su constructor y nunca
se interpretan como libros. No se necesitan los originales de la planta.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocessing_pipeline as p
import seleccion_de_variables as s
from limpieza import aprender_reglas_de_limpieza, limpiar


@pytest.fixture
def entrada():
    tabla = pd.DataFrame({"DIA": pd.date_range("2022-01-01", periods=4),
                          "a": [1., 2., 3., 4.], "b": [2.]*4,
                          "a x b": [2., 4., 6., 8.], "valido": True,
                          "objetivo": [2., np.nan, 6., 8.]}, index=[10, 20, 30, 40])
    return tabla, {"interacciones": {"variables_base": ["a", "b"]}}


def test_preparacion_explicita_sin_attrs_y_sin_modificar_historia(entrada):
    tabla, reglas = entrada
    original = tabla.copy()
    X, y, mapa = s.preparar_datos_seleccion(tabla, reglas)
    assert X.columns.tolist() == ["a", "b", "a x b"]
    assert X.index.tolist() == y.index.tolist() == [10, 30, 40]
    assert mapa == {"a x b": ["a", "b"]}
    assert json.loads(json.dumps(mapa, allow_nan=False)) == mapa
    pd.testing.assert_frame_equal(tabla, original)
    with pytest.raises(ValueError, match="faltan"):
        s.preparar_datos_seleccion(tabla.drop(columns="b"), reglas)
    with pytest.raises(ValueError, match="cronológico"):
        s.preparar_datos_seleccion(tabla.iloc[::-1], reglas)
    with pytest.raises(ValueError, match="finitos"):
        s.preparar_datos_seleccion(tabla.assign(objetivo=np.inf), reglas)


def test_reglas_tienen_prioridad_sobre_attrs_y_rechazan_mapa_contradictorio(entrada, monkeypatch):
    tabla, reglas = entrada
    X, y, mapa = s.preparar_datos_seleccion(tabla, reglas)
    X.attrs = {"componentes_interacciones": {"errónea": ["falso", "padre"]}}
    recibidos = []
    def aprender(X, y, cfg, componentes, **contexto):
        recibidos.append(componentes)
        return {"seleccionadas": ["a"]}
    monkeypatch.setattr(s, "aprender_seleccion", aprender)
    cfg = {"metodo": "spearman", "k": 1}
    assert s.seleccionar_variables(X, y, cfg, reglas_variables=reglas) == ["a"]
    assert recibidos == [mapa]
    with pytest.raises(ValueError, match="no coincide"):
        s.seleccionar_variables(X, y, cfg, componentes={}, reglas_variables=reglas)


def test_advertencia_provisional_y_copia_de_configuracion(entrada, monkeypatch):
    tabla, reglas = entrada
    X, y, mapa = s.preparar_datos_seleccion(tabla, reglas)
    configuraciones = []
    def aprender(X, y, cfg, componentes, **contexto):
        configuraciones.append(cfg.copy())
        cfg["k"] = 999
        return {"seleccionadas": ["a"]}
    monkeypatch.setattr(s, "aprender_seleccion", aprender)
    with pytest.warns(UserWarning, match="k=1") as avisos:
        assert s.seleccionar_variables(X, y, configuracion=s.CONFIGURACION_PROVISIONAL, componentes=mapa) == ["a"]
    mensaje = str(avisos[0].message)
    assert "no es definitiva" in mensaje and "método ganador" in mensaje
    assert "comparación experimental" in mensaje and "sustituirse" in mensaje
    assert s.CONFIGURACION_PROVISIONAL["k"] == 1
    cfg = {"metodo": "pearson", "k": 2}
    assert s.seleccionar_variables(X, y, cfg, componentes=mapa) == ["a"]
    assert cfg["k"] == 2
    assert configuraciones[-1] == cfg


def test_reglas_limpieza_json_estandar_y_compatibilidad_series():
    tabla = pd.DataFrame({"DIA": pd.date_range("2022-01-01", periods=20),
                          "valido": True, "objetivo": np.arange(20.),
                          "Frio (Kw)": np.arange(1., 21.),
                          "Agua | Temp": [-2., -1.] + [0.]*18})
    reglas = aprender_reglas_de_limpieza(tabla)
    restauradas = json.loads(json.dumps(reglas, allow_nan=False))
    assert restauradas["limite_superior"]["Agua | Temp"] is None
    esperada = limpiar(tabla, reglas)
    pd.testing.assert_frame_equal(limpiar(tabla, restauradas), esperada)
    anteriores = dict(reglas)
    anteriores["limite_superior"] = pd.Series(reglas["limite_superior"], dtype=float)
    pd.testing.assert_frame_equal(limpiar(tabla, anteriores), esperada)
