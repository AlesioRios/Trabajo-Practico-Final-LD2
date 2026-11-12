"""Pruebas de la elección de variables base para las interacciones."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from variables_interacciones import (agrupar_por_correlacion, aprender_reglas_de_interacciones, columnas_candidatas,
                                     crear_interacciones, elegir_por_grupos, elegir_voraz, matriz_spearman,
                                     nombres_de_interacciones, spearman_con_objetivo)


@pytest.fixture
def tabla():
    """100 días: a y b casi iguales (redundantes), c independiente pero parecida al objetivo, d ruido."""
    azar = np.random.default_rng(42)
    a = azar.normal(size=100)
    objetivo = a + azar.normal(scale=0.5, size=100)
    return pd.DataFrame({
        "a": a,
        "b": a + azar.normal(scale=0.01, size=100),               # copia casi exacta de a
        "c": objetivo + azar.normal(scale=1.0, size=100) - a,     # parecida al objetivo, distinta de a
        "d": azar.normal(size=100),                               # ruido
        "constante": 1.0,
        "texto": "x",
        "valido": True,                                           # columna de control
        "objetivo": objetivo})


def test_candidatas_excluyen_control_texto_y_constantes(tabla):
    assert columnas_candidatas(tabla.drop(columns=["objetivo"])) == ["a", "b", "c", "d"]


def test_agrupa_las_redundantes_juntas(tabla):
    correlaciones = matriz_spearman(tabla[["a", "b", "c", "d"]])
    grupos, _ = agrupar_por_correlacion(correlaciones, umbral=0.8)
    assert grupos["a"] == grupos["b"]
    assert grupos["a"] != grupos["d"]


def test_por_grupos_no_repite_redundantes(tabla):
    X = tabla[["a", "b", "c", "d"]]
    parecido = spearman_con_objetivo(X, tabla["objetivo"])
    grupos, _ = agrupar_por_correlacion(matriz_spearman(X), umbral=0.8)
    elegidas = elegir_por_grupos(parecido, grupos, cantidad=3)
    assert len({"a", "b"} & set(elegidas)) == 1                   # solo una de las dos copias


def test_voraz_no_repite_redundantes_y_respeta_el_orden(tabla):
    X = tabla[["a", "b", "c", "d"]]
    parecido = spearman_con_objetivo(X, tabla["objetivo"])
    elegidas = elegir_voraz(parecido, matriz_spearman(X), cantidad=3, umbral=0.8)
    assert len({"a", "b"} & set(elegidas)) == 1
    assert elegidas[0] == parecido.abs().idxmax()


def test_voraz_respeta_la_cantidad(tabla):
    X = tabla[["a", "b", "c", "d"]]
    parecido = spearman_con_objetivo(X, tabla["objetivo"])
    assert len(elegir_voraz(parecido, matriz_spearman(X), cantidad=2)) == 2


def test_spearman_ignora_dias_vacios(tabla):
    X = tabla[["a"]].copy()
    X.iloc[:10, 0] = np.nan                                       # 10 días sin valor
    esperado = tabla["a"].iloc[10:].corr(tabla["objetivo"].iloc[10:], method="spearman")
    assert spearman_con_objetivo(X, tabla["objetivo"])["a"] == pytest.approx(esperado)


def test_par_sin_datos_en_comun_queda_en_cero():
    X = pd.DataFrame({"p": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
                      "q": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0]})
    assert matriz_spearman(X).loc["p", "q"] == 0.0


def test_reglas_eligen_variables_no_redundantes(tabla):
    reglas = aprender_reglas_de_interacciones(tabla, cantidad=3)
    assert len(reglas["variables_base"]) == 3
    assert len({"a", "b"} & set(reglas["variables_base"])) == 1


def test_reglas_guardan_la_lista_y_el_umbral(tabla):
    reglas = aprender_reglas_de_interacciones(tabla, cantidad=2, umbral=0.5)
    assert set(reglas) == {"variables_base", "umbral_correlacion"}
    assert reglas["umbral_correlacion"] == 0.5


def test_crear_interacciones_calcula_los_productos_a_mano():
    tabla = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "z": [2.0, 2.0, 2.0], "otra": [9, 9, 9]})
    resultado = crear_interacciones(tabla, {"variables_base": ["x", "y", "z"]})
    assert list(resultado.columns) == ["x", "y", "z", "otra", "x x y", "x x z", "y x z"]
    assert resultado["x x y"].tolist() == [4.0, 10.0, 18.0]
    assert resultado["x x z"].tolist() == [2.0, 4.0, 6.0]
    assert resultado["y x z"].tolist() == [8.0, 10.0, 12.0]


def test_producto_con_un_factor_vacio_queda_vacio_no_cero():
    tabla = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, 6.0]})
    resultado = crear_interacciones(tabla, {"variables_base": ["x", "y"]})
    assert resultado["x x y"].isna().tolist() == [False, True, False]


def test_no_hay_cuadrados_y_la_cantidad_es_la_de_los_pares():
    tabla = pd.DataFrame(np.arange(40, dtype=float).reshape(4, 10), columns=list("abcdefghij"))
    resultado = crear_interacciones(tabla, {"variables_base": list("abcdefghij")})
    assert resultado.shape[1] == 10 + 45
    assert not any(" x " in c and c.split(" x ")[0] == c.split(" x ")[1] for c in resultado.columns)


def test_crear_interacciones_no_modifica_la_tabla_ni_el_indice():
    tabla = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, index=[10, 20])
    copia = tabla.copy()
    resultado = crear_interacciones(tabla, {"variables_base": ["x", "y"]})
    assert tabla.equals(copia)
    assert resultado.index.tolist() == [10, 20]


def test_crear_interacciones_falla_si_falta_una_columna_o_ya_existen():
    tabla = pd.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(ValueError):
        crear_interacciones(tabla, {"variables_base": ["x", "no_existe"]})
    con_interaccion = crear_interacciones(tabla, {"variables_base": ["x", "y"]})
    with pytest.raises(ValueError):
        crear_interacciones(con_interaccion, {"variables_base": ["x", "y"]})


def test_los_nombres_siguen_el_orden_de_polynomialfeatures():
    assert nombres_de_interacciones(["a", "b", "c"]) == ["a x b", "a x c", "b x c"]
