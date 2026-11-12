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
from variables_interacciones import aprender_reglas_de_interacciones, crear_interacciones
from variables_planta import aprender_reglas_de_planta, crear_variables_planta
from variables_temporales import crear_variables_temporales

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


def cargar_tabla_completa(carpeta_datos, usar_cache=True):
    """Devuelve la tabla completa; la guarda en data/interim/ para no reconstruirla (~6 min) cada vez.

    El archivo guardado se descarta solo si algún Excel o el código que arma la tabla
    (este archivo y auxiliar_functions.py) es más nuevo que él.
    """
    carpeta_datos = Path(carpeta_datos)
    archivo_cache = carpeta_datos / "interim" / "tabla_completa.pkl"
    fuentes = list(carpeta_datos.glob("Totalizadores Planta *.xlsx")) + [Path(__file__), Path(__file__).with_name("auxiliar_functions.py")]

    if usar_cache and archivo_cache.exists():
        ultima_modificacion = max(f.stat().st_mtime for f in fuentes)
        if archivo_cache.stat().st_mtime > ultima_modificacion:           # nada cambió desde que se guardó
            return pd.read_pickle(archivo_cache)

    tabla = construir_tabla_completa(carpeta_datos)                       # camino lento
    archivo_cache.parent.mkdir(exist_ok=True)
    tabla.to_pickle(archivo_cache)
    return tabla


def dividir_por_fecha(tabla):
    """Divide la tabla del día, sin mezclarla, en entrenamiento (70 %), validación (15 %) y prueba (15 %)."""
    assert tabla["DIA"].is_monotonic_increasing, "La tabla tiene que estar en orden cronológico."
    filas = len(tabla)
    fin_entrenamiento = int(FRACCION_FIN_ENTRENAMIENTO * filas)    # 848 de 1.212
    fin_validacion = int(FRACCION_FIN_VALIDACION * filas)          # 1.030 de 1.212
    return (tabla.iloc[:fin_entrenamiento].copy(),                 # el tramo más antiguo
            tabla.iloc[fin_entrenamiento:fin_validacion].copy(),
            tabla.iloc[fin_validacion:].copy())                    # el más reciente: se toca una sola vez, al final


#----------------------------------------------------------------------------------------------------
# Variables nuevas (consigna 2.2): se aprenden reglas con entrenamiento y se aplican a cualquier tabla.

def fitear_reglas_de_variables(entrenamiento_limpio):
    """Aprende, solo con el tramo de entrenamiento ya limpio, todo lo que hace falta para crear las variables.

    Devuelve un diccionario con dos reglas, que `transform_variables` recibe tal cual:
      - `planta`: umbral de cada denominador de las razones (ADR 0013).
      - `interacciones`: las 10 variables base elegidas y el umbral de correlación (ADR 0014).
    Las variables temporales no necesitan reglas: solo miran hoy y los días anteriores.
    """
    reglas_planta = aprender_reglas_de_planta(entrenamiento_limpio)
    # Las variables base se eligen con temporales y razones ya creadas, así que se crean primero en entrenamiento.
    # Mirar solo hacia atrás da el mismo valor en el entrenamiento solo que en la tabla completa.
    con_variables = crear_variables_planta(crear_variables_temporales(entrenamiento_limpio), reglas_planta)
    return {"planta": reglas_planta, "interacciones": aprender_reglas_de_interacciones(con_variables)}


def transform_variables(tabla_limpia, reglas):
    """Agrega a la tabla limpia las variables temporales, las razones de planta y las interacciones, en ese orden.

    Recibe la tabla limpia completa y en orden cronológico (los rezagos de validación necesitan los últimos
    días de entrenamiento), y las `reglas` que devolvió `fitear_reglas_de_variables`. No aprende nada:
    no mira el contenido de la tabla para decidir umbrales ni variables. Misma cantidad de filas y mismo
    índice; las columnas originales no cambian y las nuevas se agregan al final.
    """
    con_temporales = crear_variables_temporales(tabla_limpia)
    con_planta = crear_variables_planta(con_temporales, reglas["planta"])
    return crear_interacciones(con_planta, reglas["interacciones"])
