"""Pruebas de src/variables_planta.py con tablas chicas armadas a mano."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from variables_planta import (HL_ENVASADOS, NOMBRES_NUEVAS, PLANTA_KW, aprender_reglas_de_planta,
                              crear_variables_planta)

AIRE = "Consolidado Aire | Aire Producido (M3)"
GAS = "Consolidado GasVapor | Gas Planta (Mj)"


def tabla_de_planta(frio, planta, servicios, hl, aire=None, gas=None):
    """Tabla con las columnas que leen las razones; aire y gas valen 1 en todos los días si no se dan."""
    n = len(frio)
    return pd.DataFrame({"Frio (Kw)": frio, PLANTA_KW: planta, "Servicios (Kw)": servicios,
                         HL_ENVASADOS: hl, AIRE: aire if aire is not None else [1.0] * n,
                         GAS: gas if gas is not None else [1.0] * n})


def entrenamiento_simple():
    """5 días: mediana de planta 100 y de hectolitros 10, es decir umbrales 10 y 1."""
    return tabla_de_planta(frio=[20.0] * 5, planta=[100.0] * 5, servicios=[50.0] * 5, hl=[10.0] * 5)


def test_el_umbral_es_una_fraccion_de_la_mediana_de_entrenamiento():
    reglas = aprender_reglas_de_planta(entrenamiento_simple())
    assert reglas["umbral_denominador"][PLANTA_KW] == pytest.approx(10.0)
    assert reglas["umbral_denominador"][HL_ENVASADOS] == pytest.approx(1.0)


def test_calcula_cada_razon_con_valores_conocidos():
    reglas = aprender_reglas_de_planta(entrenamiento_simple())
    tabla = tabla_de_planta(frio=[30.0], planta=[120.0], servicios=[60.0], hl=[6.0], aire=[12.0], gas=[18.0])
    nueva = crear_variables_planta(tabla, reglas).iloc[0]
    assert nueva["frio_sobre_planta"] == pytest.approx(0.25)
    assert nueva["frio_por_hl_envasado"] == pytest.approx(5.0)
    assert nueva["planta_por_hl_envasado"] == pytest.approx(20.0)
    assert nueva["aire_por_hl_envasado"] == pytest.approx(2.0)
    assert nueva["gas_por_hl_envasado"] == pytest.approx(3.0)
    assert nueva["servicios_sobre_planta"] == pytest.approx(0.5)


def test_denominador_cero_vacio_o_chico_da_vacio_y_nunca_infinito():
    reglas = aprender_reglas_de_planta(entrenamiento_simple())              # umbral de hectolitros: 1
    tabla = tabla_de_planta(frio=[20.0] * 4, planta=[100.0] * 4, servicios=[50.0] * 4,
                            hl=[0.0, 0.5, np.nan, 1.0])
    razon = crear_variables_planta(tabla, reglas)["frio_por_hl_envasado"]
    assert razon.isna().tolist() == [True, True, True, False]               # el umbral mismo (1,0) sí se acepta
    assert razon.iloc[3] == pytest.approx(20.0)
    assert not np.isinf(crear_variables_planta(tabla, reglas)[NOMBRES_NUEVAS]).any().any()


def test_un_numerador_vacio_da_razon_vacia():
    reglas = aprender_reglas_de_planta(entrenamiento_simple())
    tabla = tabla_de_planta(frio=[np.nan, 20.0], planta=[100.0, 100.0], servicios=[50.0, 50.0], hl=[10.0, 10.0])
    assert crear_variables_planta(tabla, reglas)["frio_sobre_planta"].isna().tolist() == [True, False]


def test_el_umbral_se_aprende_solo_con_entrenamiento():
    reglas = aprender_reglas_de_planta(entrenamiento_simple())
    otra = tabla_de_planta(frio=[20.0] * 3, planta=[1000.0] * 3, servicios=[50.0] * 3, hl=[500.0] * 3)
    crear_variables_planta(otra, reglas)                                    # aplicar a otra tabla no cambia las reglas
    assert reglas["umbral_denominador"][HL_ENVASADOS] == pytest.approx(1.0)


def test_no_modifica_la_original_ni_cambia_filas_ni_indice():
    reglas = aprender_reglas_de_planta(entrenamiento_simple())
    original = tabla_de_planta(frio=[30.0, 40.0], planta=[120.0, 130.0], servicios=[60.0, 65.0], hl=[6.0, 7.0])
    original.index = [10, 11]
    copia = original.copy()
    nueva = crear_variables_planta(original, reglas)
    pd.testing.assert_frame_equal(original, copia)
    assert nueva.index.tolist() == [10, 11]
    assert list(nueva.columns) == list(original.columns) + NOMBRES_NUEVAS


def test_aplicarla_dos_veces_da_lo_mismo():
    reglas = aprender_reglas_de_planta(entrenamiento_simple())
    una = crear_variables_planta(entrenamiento_simple(), reglas)
    dos = crear_variables_planta(una, reglas)
    pd.testing.assert_frame_equal(una, dos)


def test_falta_una_columna_necesaria():
    with pytest.raises(ValueError, match="Hl Cerveza Envasada"):
        aprender_reglas_de_planta(entrenamiento_simple().drop(columns=[HL_ENVASADOS]))
    reglas = aprender_reglas_de_planta(entrenamiento_simple())
    with pytest.raises(ValueError, match="Gas Planta"):
        crear_variables_planta(entrenamiento_simple().drop(columns=[GAS]), reglas)


def test_mediana_no_positiva_falla_con_mensaje():
    tabla = tabla_de_planta(frio=[20.0] * 3, planta=[100.0] * 3, servicios=[50.0] * 3, hl=[0.0] * 3)
    with pytest.raises(ValueError, match="no es positiva"):
        aprender_reglas_de_planta(tabla)
