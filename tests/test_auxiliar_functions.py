"""Pruebas de la unificación de los Excel con datos inventados (sin leer los .xlsx).

Los Excel de la cátedra no están en el repositorio, así que un test que los lea
fallaría en cualquier otra máquina. Acá se arman tablas de pocas filas que
reproducen los casos reales: un día repetido, un día cortado a las 23:00, un
consumo del orden de millones y un bloque presente en dos archivos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Los scripts del TP importan como `from auxiliar_functions import ...`, igual acá.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auxiliar_functions import (  # noqa: E402
    cierres_por_bloque,
    construir_diario,
    medir_efecto_por_grupo,
    numerar_bloques,
    unir_versiones,
)

HORAS = ["00:00:00", "12:00:00", "23:00:00", "23:59:00"]


def un_dia(fecha, consumos, horas=HORAS):
    """Un bloque diario: las lecturas `horas` de `fecha` con esos consumos de Frio."""
    return pd.DataFrame({"DIA": pd.Timestamp(fecha), "HORA": horas[: len(consumos)], "Frio (Kw)": consumos})


def archivo_viejo():
    return pd.concat([
        un_dia("2021-01-01", [0, 100, 200, 220]),
        un_dia("2021-01-02", [0, 150, 230, 250]),   # misma fecha, dos días distintos
        un_dia("2021-01-02", [0, 50, 90, 100]),
        un_dia("2021-01-03", [0, 120, 230]),         # cortado: termina a las 23:00
        un_dia("2021-01-04", [0, 1e6, 3e6, 5e6]),    # error del orden de millones
        un_dia("2021-01-05", [0, 130, 210, 240]),
    ], ignore_index=True)


def archivo_nuevo():
    return pd.concat([
        un_dia("2021-01-04", [0, 100, 200, 260]),    # el valor correcto del 04/01
        un_dia("2021-01-05", [0, 130, 210, 240]),    # idéntico al archivo viejo
        un_dia("2021-01-06", [0, 110, 190, 215]),
    ], ignore_index=True)


def test_numerar_bloques_cuenta_un_bloque_cada_vez_que_la_hora_retrocede():
    bloques = numerar_bloques(archivo_viejo())
    # 01/01 = 0, 02/01 (dos bloques) = 1 y 2, 03/01 = 3, 04/01 = 4, 05/01 = 5
    assert bloques.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 3 + [4] * 4 + [5] * 4


def test_cierres_por_bloque_toma_la_ultima_lectura_y_marca_la_calidad():
    cierres = cierres_por_bloque(archivo_viejo(), "viejo")

    assert len(cierres) == 6
    assert cierres["Frio (Kw)"].tolist() == [220, 250, 100, 230, 5e6, 240]
    # La fecha repetida se numera 0 y 1; las demás, 0.
    assert cierres["bloque"].tolist() == [0, 0, 1, 0, 0, 0]
    assert cierres["filas"].tolist() == [4, 4, 4, 3, 4, 4]
    assert cierres["completo"].tolist() == [True, True, True, False, True, True]
    assert cierres["plausible"].tolist() == [True, True, True, True, False, True]


def test_unir_versiones_elige_la_completa_y_plausible_y_si_empatan_la_mas_reciente():
    cierres = pd.concat([
        cierres_por_bloque(archivo_viejo(), "2020-2022"),
        cierres_por_bloque(archivo_nuevo(), "2021-2023"),
    ], ignore_index=True)

    mejores = unir_versiones(cierres)

    assert len(mejores) == 7  # 6 bloques del viejo + el 06/01, con el 04/01 y 05/01 repetidos
    elegidos = mejores.set_index([mejores["DIA"].dt.strftime("%d"), "bloque"])
    assert elegidos.loc[("04", 0), "Frio (Kw)"] == 260             # el millón pierde por implausible
    assert elegidos.loc[("04", 0), "archivo"] == "2021-2023"
    assert elegidos.loc[("05", 0), "archivo"] == "2021-2023"       # empate: gana el más reciente
    assert mejores["DIA"].is_monotonic_increasing                  # sale en orden cronológico


def test_construir_diario_marca_los_invalidos_sin_imputar_y_corre_el_objetivo():
    cierres = pd.concat([
        cierres_por_bloque(archivo_viejo(), "2020-2022"),
        cierres_por_bloque(archivo_nuevo(), "2021-2023"),
    ], ignore_index=True)

    diario = construir_diario(unir_versiones(cierres))

    assert diario["valido"].tolist() == [True, True, True, False, True, True, True]
    # El día cortado a las 23:00 queda vacío: no se inventa su consumo.
    assert np.isnan(diario.loc[3, "consumo_dia"])
    # objetivo = consumo del bloque siguiente. Vacío si el siguiente es inválido (el
    # segundo bloque del 02/01 mira al 03/01) o si no hay siguiente (el último).
    esperado = [250, 100, np.nan, 260, 240, 215, np.nan]
    np.testing.assert_array_equal(diario["objetivo"].to_numpy(dtype=float), np.array(esperado, dtype=float))


def test_medir_efecto_por_grupo_cuando_el_grupo_explica_casi_todo():
    # Promedio general 13; lunes promedia 11 y martes 15. Variación entre grupos = 16,
    # variación total = 20, así que los grupos explican el 80 %.
    tabla = pd.DataFrame({"dia": ["Lunes", "Lunes", "Martes", "Martes"], "consumo": [10, 12, 14, 16]})

    resumen, medidas = medir_efecto_por_grupo(tabla, "dia", "consumo", orden=["Martes", "Lunes"])

    assert resumen.index.tolist() == ["Martes", "Lunes"]           # respeta el orden pedido
    assert resumen.loc["Lunes", "Promedio"] == 11
    assert resumen.loc["Martes", ["Días", "Promedio", "Mediana"]].tolist() == [2, 15, 15]
    valores = medidas["Valor"].tolist()
    assert valores[0] == 4                                          # 15 - 11
    assert valores[1] == pytest.approx(100 * 4 / 13)
    assert valores[2] == pytest.approx(80)
    assert 0 < valores[3] < 1


def test_medir_efecto_por_grupo_cuando_el_grupo_no_explica_nada():
    # Los dos grupos promedian 13: la variación entre grupos es 0 y Kruskal-Wallis da p = 1.
    tabla = pd.DataFrame({"dia": ["Lunes", "Lunes", "Martes", "Martes"], "consumo": [10, 16, 12, 14]})

    _, medidas = medir_efecto_por_grupo(tabla, "dia", "consumo")

    valores = medidas["Valor"].tolist()
    assert valores[0] == 0
    assert valores[2] == pytest.approx(0)
    assert valores[3] == pytest.approx(1)


def test_medir_efecto_por_grupo_ignora_los_valores_vacios():
    tabla = pd.DataFrame({"dia": ["Lunes", "Lunes", "Lunes", "Martes", "Martes"],
                          "consumo": [10, np.nan, 12, 14, 16]})

    resumen, _ = medir_efecto_por_grupo(tabla, "dia", "consumo")

    assert resumen["Días"].to_dict() == {"Lunes": 2, "Martes": 2}   # el vacío no cuenta
