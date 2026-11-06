#Pipeline de preprocesamiento reproducible (Fase 2).
#Debe reutilizarse tal cual desde predict.py, sin reimplementarlo.

from pathlib import Path

import pandas as pd

from auxiliar_functions import (
    cierres_de_otras_hojas,
    cierres_por_bloque,
    construir_diario,
    leer_consolidado_ee,
    unir_versiones,
)

# Cortes de la división temporal (consigna 2.4): 70 % entrenamiento, 15 % validación, 15 % prueba
FRACCION_FIN_ENTRENAMIENTO = 0.70
FRACCION_FIN_VALIDACION = 0.85

#----------------------------------------------------------------------------------------------------

def construir_tabla_completa(carpeta_datos):
    """Une los tres Excel y todas sus hojas en una tabla con una fila por día."""
    archivos = sorted(Path(carpeta_datos).glob("Totalizadores Planta *.xlsx"))
    assert len(archivos) == 3, "Faltan Excel en data/ (ver README, sección 'Datos')."

    cierres = []
    for archivo in archivos:
        nombre = archivo.stem.replace("Totalizadores Planta ", "")       # p. ej. "2020-2022"
        with pd.ExcelFile(archivo) as libro:                              # se cierra solo al salir
            tabla_ee, _ = leer_consolidado_ee(libro)                      # hoja del objetivo
            de_ee = cierres_por_bloque(tabla_ee, nombre)                  # un día por fila, con el objetivo
            otras = {hoja: pd.read_excel(libro, sheet_name=hoja)          # las demás hojas, sin leer EE dos veces
                     for hoja in libro.sheet_names if hoja != "Consolidado EE"}
        de_otras = cierres_de_otras_hojas(otras)                          # una fila por bloque, ADR 0006
        assert len(de_otras) == len(de_ee), f"{nombre}: distinta cantidad de bloques"
        cierres.append(pd.concat([de_ee, de_otras], axis=1))              # pegadas por número de bloque

    # Una versión por día (ADR 0003) y marca de día válido (ADR 0004 y 0005)
    return construir_diario(unir_versiones(pd.concat(cierres, ignore_index=True)))


def dividir_por_fecha(tabla):
    """Divide la tabla del día, sin mezclarla, en entrenamiento (70 %), validación (15 %) y prueba (15 %)."""
    assert tabla["DIA"].is_monotonic_increasing, "La tabla tiene que estar en orden cronológico."
    filas = len(tabla)
    fin_entrenamiento = int(FRACCION_FIN_ENTRENAMIENTO * filas)    # 848 de 1.212
    fin_validacion = int(FRACCION_FIN_VALIDACION * filas)          # 1.030 de 1.212
    return (tabla.iloc[:fin_entrenamiento].copy(),                 # el tramo más antiguo
            tabla.iloc[fin_entrenamiento:fin_validacion].copy(),
            tabla.iloc[fin_validacion:].copy())                    # el más reciente: se toca una sola vez, al final
