"""Contratos y aislamiento temporal con ejemplos sintéticos."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit, KFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import seleccion_de_variables as s


def test_jerarquia_y_aplicacion():
    X = pd.DataFrame({"A": [1, 2, 3], "B": [2, 2, 2], "A x B": [2, 4, 6]}, index=[7, 8, 9])
    elegidas = s.cerrar_jerarquia(["A x B"], {"A x B": ["A", "B"]}, X.columns)
    assert elegidas == ["A x B", "A", "B"]
    salida = s.aplicar_seleccion(X, {"seleccionadas": elegidas})
    assert salida.index.tolist() == [7, 8, 9]
    assert salida["A x B"].tolist() == [2, 4, 6]
    assert X.columns.tolist() == ["A", "B", "A x B"]
    with pytest.raises(ValueError, match="Faltan columnas"):
        s.aplicar_seleccion(X.drop(columns="A"), {"seleccionadas": elegidas})


def test_spearman_json_y_configuracion():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 1, 3], "constante": [1]*4})
    y = pd.Series([10, 20, 30, 40])
    reglas = s.aprender_seleccion(X, y, {"metodo": "spearman", "k": 1})
    assert reglas["seleccionadas"] == ["a"]
    assert reglas["puntajes"]["a"] == pytest.approx(1)
    json.dumps(reglas, allow_nan=False)
    with pytest.raises(ValueError, match="aprobar"):
        s.seleccionar_variables(X, y)
    with pytest.raises(ValueError, match="mapa"):
        s.aprender_seleccion(X.rename(columns={"a": "a x b"}), y, {"metodo": "spearman", "k": 1})
    with pytest.raises(ValueError, match="finito"):
        s.aprender_seleccion(X, pd.Series([1, 2, 3, np.nan]), {"metodo": "spearman", "k": 1})
    with pytest.raises(ValueError, match="infinitos"):
        s.aprender_seleccion(X.assign(a=[1, 2, 3, np.inf]), y, {"metodo": "spearman", "k": 1})


@pytest.mark.parametrize("cfg", [
    {"metodo": "mutua", "k": 1, "vecinos": 2, "discretas": []},
    {"metodo": "lasso", "criterio": "no_cero", "parametros": {"alpha": 0.01, "max_iter": 10000}},
])
def test_metodos_reproducibles_sinteticos(cfg):
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8], "b": [0, 1, 0, 1, 0, 1, 0, 1]})
    y = pd.Series([2, 4, 6, 8, 10, 12, 14, 16])
    uno = s.aprender_seleccion(X, y, cfg)
    dos = s.aprender_seleccion(X, y, cfg)
    assert uno["seleccionadas"] == dos["seleccionadas"]
    assert uno["puntajes"] == dos["puntajes"]
    assert "a" in uno["seleccionadas"]


def test_pliegues_reaprenden_y_futuro_no_cambia_reglas(monkeypatch):
    llamadas = []
    def aprender(tabla):
        llamadas.append(tabla.index.tolist())
        return {}
    monkeypatch.setattr(s, "aprender_reglas_de_limpieza", aprender)
    monkeypatch.setattr(s, "limpiar", lambda tabla, reglas: tabla.copy())
    monkeypatch.setattr(s, "fitear_reglas_de_variables", lambda tabla: {"interacciones": {"variables_base": []}})
    monkeypatch.setattr(s, "transform_variables", lambda tabla, reglas: tabla.copy())
    tabla = pd.DataFrame({"DIA": pd.date_range("2020-01-01", periods=12), "a": range(12), "objetivo": range(1, 13)}, index=range(100, 112))
    cfg = {"metodo": "spearman", "k": 1}
    uno = s.evaluar_temporal(tabla, TimeSeriesSplit(n_splits=3, gap=0), DummyRegressor(), cfg)
    assert llamadas == [list(range(100, 103)), list(range(100, 106)), list(range(100, 109))]
    assert uno[0]["fin_entrenamiento"] == 2
    assert uno[0]["inicio_evaluacion"] == 3
    assert uno[0]["indices_evaluados"] == [103, 104, 105]
    tabla.loc[103:, ["a", "objetivo"]] = -999
    dos = s.evaluar_temporal(tabla, TimeSeriesSplit(n_splits=3, gap=0), DummyRegressor(), cfg)
    for campo in ["seleccionadas", "puntajes", "candidatas"]:
        assert uno[0]["reglas"][campo] == dos[0]["reglas"][campo]
    with pytest.raises(ValueError, match="TimeSeriesSplit"):
        s.evaluar_temporal(tabla, KFold(3), DummyRegressor())


def test_etiquetas_gap_cero_y_mascara_conservan_posiciones(monkeypatch):
    monkeypatch.setattr(s, "aprender_reglas_de_limpieza", lambda t: {})
    monkeypatch.setattr(s, "limpiar", lambda t, r: t.copy())
    monkeypatch.setattr(s, "fitear_reglas_de_variables", lambda t: {"interacciones": {"variables_base": []}})
    monkeypatch.setattr(s, "transform_variables", lambda t, r: t.copy())
    tabla = pd.DataFrame({"DIA": pd.date_range("2020-01-01", periods=12), "a": range(12), "objetivo": range(1, 13)})
    tabla.loc[4, "objetivo"] = np.nan
    resultado = s.evaluar_temporal(tabla, TimeSeriesSplit(n_splits=3), DummyRegressor())
    assert resultado[0]["indices_evaluados"] == [3, 5]
    assert resultado[0]["fin_entrenamiento"] == 2
    assert tabla.loc[2, "objetivo"] == tabla.loc[3, "a"]


def test_estabilidad_calculada_a_mano():
    frecuencia, pares = s.resumir_estabilidad([
        {"seleccionadas": ["a", "b"]}, {"seleccionadas": ["a", "c"]}])
    assert frecuencia.Frecuencia.tolist() == [1.0, 0.5, 0.5]
    assert pares.Jaccard.iloc[0] == pytest.approx(1/3)


def test_flujo_cientifico_real_sin_fuga(monkeypatch):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    t = np.arange(48, dtype=float)
    frio = 100 + t + 5*np.sin(t)
    tabla = pd.DataFrame({"DIA": pd.date_range("2020-01-01", periods=48),
        "Frio (Kw)": frio, "Planta (Kw)": 1000 + 3*t,
        "Servicios (Kw)": 400 + t,
        "Consolidado Produccion | Hl Cerveza Envasada": 100 + 2*t,
        "Consolidado Aire | Aire Producido (M3)": 200 + t*t,
        "Consolidado GasVapor | Gas Planta (Mj)": 500 + t,
        "valido": True, "objetivo": pd.Series(frio).shift(-1)})
    llamadas = []
    original = s.fitear_reglas_de_variables
    def registrar(tabla):
        llamadas.append(tabla.index.tolist())
        return original(tabla)
    monkeypatch.setattr(s, "fitear_reglas_de_variables", registrar)
    estimador = make_pipeline(SimpleImputer(strategy="median"), Ridge())
    cfg = {"metodo": "spearman", "k": 1}
    uno = s.evaluar_temporal(tabla, TimeSeriesSplit(n_splits=2), estimador, cfg)
    assert llamadas == [list(range(16)), list(range(32))]
    alterada = tabla.copy()
    alterada.loc[16:, "Frio (Kw)"] *= 100
    alterada.loc[16:, "objetivo"] *= 100
    dos = s.evaluar_temporal(alterada, TimeSeriesSplit(n_splits=2), estimador, cfg)
    assert uno[0]["reglas"]["puntajes"] == dos[0]["reglas"]["puntajes"]
    assert uno[0]["seleccionadas"] == dos[0]["seleccionadas"]
    assert uno[0]["indices_evaluados"] == list(range(16, 32))


@pytest.mark.parametrize("metodo", ["forward", "backward"])
def test_wrappers_temporales_sinteticos(monkeypatch, metodo):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    monkeypatch.setattr(s, "aprender_reglas_de_limpieza", lambda t: {})
    monkeypatch.setattr(s, "limpiar", lambda t, r: t.copy())
    monkeypatch.setattr(s, "fitear_reglas_de_variables", lambda t: {"interacciones": {"variables_base": []}})
    monkeypatch.setattr(s, "transform_variables", lambda t, r: t.copy())
    tabla = pd.DataFrame({"DIA": pd.date_range("2020-01-01", periods=18),
        "a": np.arange(18.), "b": [0,1,0]*6, "objetivo": 2*np.arange(18.)})
    cfg = {"metodo": metodo, "candidatas": ["a", "b"], "iniciales": [],
        "cantidad_objetivo": 1, "max_pasos": 2, "max_ajustes": 40, "mejora_minima": None}
    modelo = make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression())
    contexto = dict(tabla_bruta=tabla, estimador=modelo, particion_interna=TimeSeriesSplit(2))
    uno = s.aprender_seleccion(tabla[["a", "b"]], tabla.objetivo, cfg, **contexto)
    dos = s.aprender_seleccion(tabla[["a", "b"]], tabla.objetivo, cfg, **contexto)
    assert uno["seleccionadas"] == ["a"]
    assert uno["trayectoria"] == dos["trayectoria"]
    assert uno["ajustes"] <= 40
    assert uno["cantidad_directas"] == uno["cantidad_efectiva"] == 1


@pytest.mark.parametrize("metodo", ["forward", "backward"])
def test_wrapper_jerarquia_durante_busqueda_y_presupuesto(monkeypatch, metodo):
    evaluadas = []
    def evaluar(tabla, particion, estimador, configuracion=None, *, columnas_fijas=None):
        assert "a x b" not in columnas_fijas or {"a", "b"} <= set(columnas_fijas)
        evaluadas.append(columnas_fijas)
        return [{"mae": 0 if "a x b" in columnas_fijas else 10}]*2
    monkeypatch.setattr(s, "evaluar_temporal", evaluar)
    cfg = {"metodo": metodo, "candidatas": ["a x b", "ruido"], "iniciales": [],
        "cantidad_objetivo": 1, "max_pasos": 1, "max_ajustes": 6, "mejora_minima": None}
    reglas = s.aprender_seleccion(None, None, cfg, {"a x b": ["a", "b"]},
        tabla_bruta=pd.DataFrame(), estimador=DummyRegressor(), particion_interna=TimeSeriesSplit(2))
    assert reglas["directas"] == ["a x b"]
    assert reglas["seleccionadas"] == ["a x b", "a", "b"]
    assert reglas["cantidad_efectiva"] == 3
    assert reglas["ajustes"] == (4 if metodo == "forward" else 6)
    cfg["max_ajustes"] = 1
    cfg["metodo"] = "forward"
    with pytest.raises(ValueError, match="Presupuesto"):
        s.aprender_seleccion(None,None,cfg, {"a x b": ["a", "b"]},
            tabla_bruta=pd.DataFrame(), estimador=DummyRegressor(), particion_interna=TimeSeriesSplit(2))


def test_primera_prediccion_invariante_a_futuro_real(monkeypatch):
    monkeypatch.setattr(s, "aprender_reglas_de_limpieza", lambda t: {})
    monkeypatch.setattr(s, "limpiar", lambda t, r: t.copy())
    monkeypatch.setattr(s, "fitear_reglas_de_variables", lambda t: {"interacciones": {"variables_base": []}})
    monkeypatch.setattr(s, "transform_variables", lambda t, r: t.copy())
    from sklearn.linear_model import LinearRegression
    class Registrar(LinearRegression):
        predicciones = []
        def predict(self, X):
            salida = super().predict(X)
            self.predicciones.append(float(salida[0]))
            return salida
    tabla=pd.DataFrame({"DIA":pd.date_range("2020-01-01",periods=12),
        "a":np.arange(12.), "objetivo":np.arange(1.,13.)})
    s.evaluar_temporal(tabla,TimeSeriesSplit(3),Registrar(),{"metodo":"spearman","k":1})
    primera=Registrar.predicciones[0]
    tabla.loc[4:,"a"] = 999
    tabla.loc[3:,"objetivo"] = 999
    Registrar.predicciones=[]
    s.evaluar_temporal(tabla,TimeSeriesSplit(3),Registrar(),{"metodo":"spearman","k":1})
    assert Registrar.predicciones[0] == pytest.approx(primera)


def test_wrapper_preprocesamiento_solo_entrenamiento_interno(monkeypatch):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    class RegistrarImputacion(SimpleImputer):
        indices = []
        def fit(self, X, y=None):
            self.indices.append(list(X.index))
            return super().fit(X,y)
    monkeypatch.setattr(s, "aprender_reglas_de_limpieza", lambda t: {})
    monkeypatch.setattr(s, "limpiar", lambda t, r: t.copy())
    monkeypatch.setattr(s, "fitear_reglas_de_variables", lambda t: {"interacciones": {"variables_base": []}})
    monkeypatch.setattr(s, "transform_variables", lambda t, r: t.copy())
    t=pd.DataFrame({"DIA":pd.date_range("2020-01-01",periods=12), "a":np.arange(12.), "objetivo":np.arange(12.)})
    cfg={"metodo":"forward", "candidatas":["a"], "iniciales":[],
        "cantidad_objetivo":1,"max_pasos":1,"max_ajustes":2,"mejora_minima":None}
    s.aprender_seleccion(None,None,cfg,tabla_bruta=t,
        estimador=make_pipeline(RegistrarImputacion(),StandardScaler(),LinearRegression()),
        particion_interna=TimeSeriesSplit(2))
    assert RegistrarImputacion.indices == [list(range(4)), list(range(8))]


def test_wrapper_busca_solo_dentro_de_entrenamiento_externo(monkeypatch):
    recibidos=[]
    def buscar(tabla,cfg,componentes,estimador,particion):
        recibidos.append(tabla.index.tolist())
        return {"seleccionadas":["a"]}
    monkeypatch.setattr(s,"_buscar_secuencial",buscar)
    monkeypatch.setattr(s,"aprender_reglas_de_limpieza",lambda t:{})
    monkeypatch.setattr(s,"limpiar",lambda t,r:t.copy())
    monkeypatch.setattr(s,"fitear_reglas_de_variables",lambda t:{"interacciones":{"variables_base":[]}})
    monkeypatch.setattr(s,"transform_variables",lambda t,r:t.copy())
    t=pd.DataFrame({"DIA":pd.date_range("2020-01-01",periods=12),"a":np.arange(12.),"objetivo":np.arange(12.)})
    s.evaluar_temporal(t,TimeSeriesSplit(2),DummyRegressor(),{"metodo":"forward"},
        particion_interna=TimeSeriesSplit(2),estimador_selector=DummyRegressor())
    assert recibidos == [list(range(4)),list(range(8))]
