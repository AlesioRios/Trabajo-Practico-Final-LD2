"""Variables temporales (Fase 2, consigna 2.2).

`crear_variables_temporales(tabla)` recibe la tabla ya limpia, con todos los días en orden
cronológico (entrenamiento, validación y prueba concatenados), y devuelve la misma tabla con
columnas nuevas. Todas las columnas usan solo información de hoy o anterior:

  - del frío: rezagos de 1, 2 y 3 días, promedio y desvío estándar móviles de 7 días y la
    diferencia entre hoy y ayer;
  - del calendario del día que se predice (mañana, que ya se conoce hoy): seno y coseno del
    día del año, indicador de fin de año e indicador de fin de semana.

Los rezagos se calculan por fila y no por fecha: las filas son días consecutivos, mientras que
la columna DIA tiene fechas repetidas y faltantes por el desfase de fin de mes (ADR 0002).

Las justificaciones están en el notebook variables_temporales.ipynb y en el ADR 0012.
"""

import numpy as np
import pandas as pd

from limpieza import _exigir_columnas

COLUMNA_FRIO = "Frio (Kw)"
VENTANA_DIAS = 7                 # largo de la ventana móvil
MIN_DIAS_CON_DATO = 3            # una ventana de 7 días tolera hasta 4 faltantes
DIAS_PERIODO_ANUAL = 365.25      # período del seno y el coseno
INICIO_FIN_DE_ANIO = (12, 24)    # (mes, día) desde el que empieza la ventana de fin de año
FIN_FIN_DE_ANIO = (1, 6)         # (mes, día) hasta el que dura

NOMBRES_NUEVAS = ["frio_ayer", "frio_hace_2_dias", "frio_hace_3_dias",
                  "frio_media_7_dias", "frio_desvio_7_dias", "frio_hoy_menos_ayer",
                  "dia_anio_seno", "dia_anio_coseno", "es_fin_de_anio", "es_fin_de_semana"]

#----------------------------------------------------------------------------------------------------

def es_fin_de_anio(fechas):
    """Verdadero si la fecha cae entre el 24 de diciembre y el 6 de enero (ambos incluidos)."""
    mes_dia = fechas.dt.month * 100 + fechas.dt.day                 # 24 de diciembre -> 1224; 6 de enero -> 106
    desde = INICIO_FIN_DE_ANIO[0] * 100 + INICIO_FIN_DE_ANIO[1]
    hasta = FIN_FIN_DE_ANIO[0] * 100 + FIN_FIN_DE_ANIO[1]
    return (mes_dia >= desde) | (mes_dia <= hasta)


def crear_variables_temporales(tabla):
    """Devuelve una copia de `tabla` con las variables temporales agregadas al final.

    Requiere las columnas DIA y «Frio (Kw)» y que DIA esté en orden cronológico. No modifica la
    tabla recibida, no cambia sus filas ni su índice y no transforma ni escala nada (los valores
    quedan en kW). Los días sin dato de frío dan vacío en las columnas que dependen de él.
    """
    _exigir_columnas(tabla, ["DIA", COLUMNA_FRIO], "necesarias para las variables temporales")
    fechas = pd.to_datetime(tabla["DIA"])
    if not fechas.is_monotonic_increasing:
        raise ValueError("Las filas de la tabla deben estar en orden cronológico (DIA creciente): "
                         "los rezagos y los promedios móviles dependen de ese orden.")

    nueva = tabla.copy()
    frio = pd.to_numeric(nueva[COLUMNA_FRIO], errors="coerce")

    # Del frío: solo hoy y días anteriores (shift positivo mira hacia atrás)
    nueva["frio_ayer"] = frio.shift(1)
    nueva["frio_hace_2_dias"] = frio.shift(2)
    nueva["frio_hace_3_dias"] = frio.shift(3)
    ventana = frio.rolling(VENTANA_DIAS, min_periods=MIN_DIAS_CON_DATO)   # hoy y los 6 días anteriores
    nueva["frio_media_7_dias"] = ventana.mean()
    nueva["frio_desvio_7_dias"] = ventana.std()
    nueva["frio_hoy_menos_ayer"] = frio - frio.shift(1)

    # Del calendario del día que se predice: mañana, que ya se conoce hoy
    manana = fechas + pd.Timedelta(days=1)
    angulo = 2 * np.pi * manana.dt.dayofyear / DIAS_PERIODO_ANUAL
    nueva["dia_anio_seno"] = np.sin(angulo)
    nueva["dia_anio_coseno"] = np.cos(angulo)
    nueva["es_fin_de_anio"] = es_fin_de_anio(manana).astype(int)
    nueva["es_fin_de_semana"] = (manana.dt.dayofweek >= 5).astype(int)
    return nueva
