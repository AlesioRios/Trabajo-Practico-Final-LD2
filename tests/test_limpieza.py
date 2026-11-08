"""Pruebas de la limpieza (Fase 2, consigna 2.1) con tablas chicas inventadas.

Cada prueba arma una tabla de 20 días donde el resultado se puede calcular a mano.
No se leen los Excel de la cátedra.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from limpieza import aprender_reglas_de_limpieza, limpiar

DIAS = 20


def tabla_de_entrenamiento():
    """20 días con una columna de cada tipo que interesa a la limpieza."""
    dias = np.arange(1, DIAS + 1, dtype=float)
    tabla = pd.DataFrame({"DIA": pd.date_range("2022-01-01", periods=DIAS),
                          "valido": True,
                          "objetivo": dias * 10})
    tabla["Frio (Kw)"] = dias                                    # energía (sin ' | ' en el nombre)
    tabla["Consolidado KPI | EE Planta / Hl"] = dias             # razón de energía del KPI
    tabla["Agua | Caudal"] = dias                                # otra hoja: no es de energía
    tabla["Agua | Temp Tanque"] = dias                           # temperatura: puede ser negativa
    tabla["Agua | Constante"] = 5.0                              # siempre igual: se descarta
    tabla["Agua | Casi vacia"] = [1.0] * 5 + [np.nan] * 15       # 75 % de vacíos: se descarta
    return tabla


def test_descarta_columnas_constantes_y_casi_vacias():
    reglas = aprender_reglas_de_limpieza(tabla_de_entrenamiento())
    assert "Agua | Constante" not in reglas["columnas_conservadas"]
    assert "Agua | Casi vacia" not in reglas["columnas_conservadas"]
    assert "Agua | Caudal" in reglas["columnas_conservadas"]


def test_limpiar_mantiene_las_columnas_de_control_y_las_filas():
    tabla = tabla_de_entrenamiento()
    limpia = limpiar(tabla, aprender_reglas_de_limpieza(tabla))
    assert len(limpia) == len(tabla)                             # no se borra ninguna fila
    assert {"DIA", "valido", "objetivo"} <= set(limpia.columns)
    assert "Agua | Constante" not in limpia.columns


def test_dia_invalido_anula_solo_las_columnas_de_energia():
    tabla = tabla_de_entrenamiento()
    reglas = aprender_reglas_de_limpieza(tabla)
    nueva = tabla.copy()
    nueva.loc[3, "valido"] = False
    limpia = limpiar(nueva, reglas)
    assert np.isnan(limpia.loc[3, "Frio (Kw)"])
    assert np.isnan(limpia.loc[3, "Consolidado KPI | EE Planta / Hl"])
    assert limpia.loc[3, "Agua | Caudal"] == nueva.loc[3, "Agua | Caudal"]      # otra hoja: se conserva
    assert limpia.loc[3, "objetivo"] == nueva.loc[3, "objetivo"]                # el objetivo no se toca
    assert limpia["Frio (Kw)"].isna().sum() == 1                                 # solo ese día


def test_valor_sobre_cinco_veces_el_p99_se_anula_y_uno_menor_no():
    tabla = tabla_de_entrenamiento()
    reglas = aprender_reglas_de_limpieza(tabla)
    limite = reglas["limite_superior"]["Agua | Caudal"]
    nueva = tabla.copy()
    nueva.loc[0, "Agua | Caudal"] = limite * 1.01
    nueva.loc[1, "Agua | Caudal"] = limite * 0.99
    limpia = limpiar(nueva, reglas)
    assert np.isnan(limpia.loc[0, "Agua | Caudal"])
    assert limpia.loc[1, "Agua | Caudal"] == limite * 0.99


def test_el_limite_es_cinco_veces_el_p99_del_entrenamiento():
    tabla = tabla_de_entrenamiento()
    reglas = aprender_reglas_de_limpieza(tabla)
    esperado = 5 * np.quantile(np.arange(1, DIAS + 1), 0.99)     # p99 de 1..20
    assert reglas["limite_superior"]["Agua | Caudal"] == esperado


def test_negativo_se_anula_salvo_en_temperatura():
    tabla = tabla_de_entrenamiento()
    reglas = aprender_reglas_de_limpieza(tabla)
    nueva = tabla.copy()
    nueva.loc[2, ["Agua | Caudal", "Agua | Temp Tanque"]] = -3.0
    limpia = limpiar(nueva, reglas)
    assert np.isnan(limpia.loc[2, "Agua | Caudal"])
    assert limpia.loc[2, "Agua | Temp Tanque"] == -3.0


def test_usa_las_reglas_del_entrenamiento_y_no_las_de_la_tabla_que_limpia():
    tabla = tabla_de_entrenamiento()
    reglas = aprender_reglas_de_limpieza(tabla)
    otra = tabla.copy()
    otra["Agua | Caudal"] = 1000.0                               # si recalculara el p99, ninguno sería absurdo
    assert limpiar(otra, reglas)["Agua | Caudal"].isna().all()


def test_limpiar_dos_veces_da_lo_mismo_que_una():
    tabla = tabla_de_entrenamiento()
    tabla.loc[4, "Agua | Caudal"] = 5000.0
    tabla.loc[6, "valido"] = False
    reglas = aprender_reglas_de_limpieza(tabla)
    una_vez = limpiar(tabla, reglas)
    pdt.assert_frame_equal(limpiar(una_vez, reglas), una_vez)


def test_no_modifica_la_tabla_recibida():
    tabla = tabla_de_entrenamiento()
    tabla.loc[6, "valido"] = False
    reglas = aprender_reglas_de_limpieza(tabla)
    copia = tabla.copy()
    limpiar(tabla, reglas)
    pdt.assert_frame_equal(tabla, copia)


def test_si_falta_una_columna_necesaria_avisa_cual_falta():
    tabla = tabla_de_entrenamiento()
    reglas = aprender_reglas_de_limpieza(tabla)
    sin_caudal = tabla.drop(columns=["Agua | Caudal"])
    with pytest.raises(ValueError, match="Caudal"):            # el mensaje nombra la columna que falta
        limpiar(sin_caudal, reglas)


def test_si_falta_la_marca_de_dia_valido_avisa():
    tabla = tabla_de_entrenamiento()
    reglas = aprender_reglas_de_limpieza(tabla)
    with pytest.raises(ValueError, match="valido"):
        limpiar(tabla.drop(columns=["valido"]), reglas)


def test_ignora_las_columnas_de_mas():
    tabla = tabla_de_entrenamiento()
    reglas = aprender_reglas_de_limpieza(tabla)
    con_extra = tabla.assign(**{"Hoja nueva | Columna rara": 1.0})
    limpia = limpiar(con_extra, reglas)
    assert "Hoja nueva | Columna rara" not in limpia.columns
    pdt.assert_frame_equal(limpia, limpiar(tabla, reglas))
