"""Pruebas de src/variables_temporales.py con tablas chicas armadas a mano."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from variables_temporales import NOMBRES_NUEVAS, crear_variables_temporales


def tabla_de_frio(valores, inicio="2021-03-01"):
    """Tabla con un día por fila a partir de `inicio` y el frío indicado."""
    return pd.DataFrame({"DIA": pd.date_range(inicio, periods=len(valores)),
                         "Frio (Kw)": valores, "otra": range(len(valores))})


def test_rezagos_de_uno_dos_y_tres_dias():
    nueva = crear_variables_temporales(tabla_de_frio([10.0, 20.0, 30.0, 40.0, 50.0]))
    assert nueva["frio_ayer"].tolist()[1:] == [10.0, 20.0, 30.0, 40.0]
    assert nueva["frio_hace_2_dias"].tolist()[2:] == [10.0, 20.0, 30.0]
    assert nueva["frio_hace_3_dias"].tolist()[3:] == [10.0, 20.0]
    assert nueva["frio_ayer"].isna().tolist() == [True, False, False, False, False]
    assert nueva["frio_hace_3_dias"].isna().tolist() == [True, True, True, False, False]


def test_diferencia_entre_hoy_y_ayer():
    nueva = crear_variables_temporales(tabla_de_frio([10.0, 25.0, 20.0]))
    assert np.isnan(nueva["frio_hoy_menos_ayer"].iloc[0])
    assert nueva["frio_hoy_menos_ayer"].tolist()[1:] == [15.0, -5.0]


def test_media_y_desvio_de_siete_dias_incluyen_hoy():
    valores = [float(v) for v in range(1, 11)]                 # 1, 2, ..., 10
    nueva = crear_variables_temporales(tabla_de_frio(valores))
    # El 7.º día (fila 6) promedia los días 1 a 7; el 10.º promedia los días 4 a 10
    assert nueva["frio_media_7_dias"].iloc[6] == pytest.approx(4.0)
    assert nueva["frio_media_7_dias"].iloc[9] == pytest.approx(7.0)
    assert nueva["frio_desvio_7_dias"].iloc[9] == pytest.approx(np.std(range(4, 11), ddof=1))


def test_la_ventana_tolera_hasta_cuatro_faltantes():
    # En la fila 6 la ventana tiene 3 datos y 4 faltantes: alcanza. En la fila 7, con 2 datos, no.
    valores = [1.0, np.nan, np.nan, np.nan, np.nan, 2.0, 3.0, np.nan]
    nueva = crear_variables_temporales(tabla_de_frio(valores))
    assert nueva["frio_media_7_dias"].iloc[6] == pytest.approx(2.0)
    assert np.isnan(nueva["frio_media_7_dias"].iloc[5])        # solo 2 datos
    assert np.isnan(nueva["frio_media_7_dias"].iloc[4])        # solo 1 dato


def test_no_usa_informacion_del_futuro():
    base = tabla_de_frio([float(v) for v in range(1, 16)])
    cambiada = base.copy()
    cambiada.loc[10:, "Frio (Kw)"] = 999.0                     # se altera todo desde la fila 10
    a, b = crear_variables_temporales(base), crear_variables_temporales(cambiada)
    columnas_del_frio = [c for c in NOMBRES_NUEVAS if c.startswith("frio_")]
    pd.testing.assert_frame_equal(a.loc[:9, columnas_del_frio], b.loc[:9, columnas_del_frio])


def test_el_calendario_es_el_del_dia_siguiente():
    dias = ["2021-12-22", "2021-12-23", "2022-01-05", "2022-01-06"]
    tabla = pd.DataFrame({"DIA": pd.to_datetime(dias), "Frio (Kw)": [1.0, 2.0, 3.0, 4.0]})
    # Mañana es 23-dic (no), 24-dic (sí), 6-ene (sí, último día) y 7-ene (no)
    assert crear_variables_temporales(tabla)["es_fin_de_anio"].tolist() == [0, 1, 1, 0]


def test_fin_de_semana_segun_el_dia_siguiente():
    # 2021-03-04 es jueves: mañana viernes (0), viernes -> sábado (1), sábado -> domingo (1), domingo -> lunes (0)
    tabla = tabla_de_frio([1.0, 2.0, 3.0, 4.0], inicio="2021-03-04")
    assert crear_variables_temporales(tabla)["es_fin_de_semana"].tolist() == [0, 1, 1, 0]


def test_seno_y_coseno_del_dia_del_anio_siguiente():
    tabla = pd.DataFrame({"DIA": pd.to_datetime(["2021-12-31", "2022-01-01", "2022-06-30"]),
                          "Frio (Kw)": [1.0, 2.0, 3.0]})
    nueva = crear_variables_temporales(tabla)
    assert (nueva["dia_anio_seno"] ** 2 + nueva["dia_anio_coseno"] ** 2).tolist() == pytest.approx([1.0] * 3)
    # 31-dic y 1-ene quedan casi pegados en el círculo (mañana es 1-ene y 2-ene)
    assert abs(nueva["dia_anio_seno"].iloc[0] - nueva["dia_anio_seno"].iloc[1]) < 0.05
    assert nueva["dia_anio_coseno"].iloc[0] > 0.99


def test_no_modifica_la_original_ni_cambia_filas_ni_indice():
    original = tabla_de_frio([1.0, 2.0, 3.0, 4.0])
    original.index = [10, 11, 12, 13]
    copia = original.copy()
    nueva = crear_variables_temporales(original)
    pd.testing.assert_frame_equal(original, copia)
    assert nueva.index.tolist() == [10, 11, 12, 13]
    assert list(nueva.columns) == list(original.columns) + NOMBRES_NUEVAS


def test_aplicarla_dos_veces_da_lo_mismo():
    una = crear_variables_temporales(tabla_de_frio([float(v) for v in range(12)]))
    dos = crear_variables_temporales(una)
    pd.testing.assert_frame_equal(una, dos)


def test_falta_una_columna_necesaria():
    with pytest.raises(ValueError, match="Frio"):
        crear_variables_temporales(pd.DataFrame({"DIA": pd.date_range("2021-01-01", periods=3)}))


def test_filas_desordenadas_falla_con_mensaje():
    desordenada = tabla_de_frio([1.0, 2.0, 3.0]).iloc[[2, 0, 1]]
    with pytest.raises(ValueError, match="orden cronológico"):
        crear_variables_temporales(desordenada)
