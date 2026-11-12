"""Variables de planta: razones entre áreas (Fase 2, consigna 2.2).

Se hace en dos pasos, igual que la limpieza, para no mirar nunca validación ni prueba al decidir:
  1. `aprender_reglas_de_planta(entrenamiento_limpio)` mide, solo con entrenamiento, a partir de qué
     valor un denominador se considera demasiado chico para dividir por él.
  2. `crear_variables_planta(tabla, reglas)` calcula las razones en cualquier tabla ya limpia
     (entrenamiento, validación, prueba o datos nuevos en predict.py), sin recalcular nada.

Cada razón usa solo mediciones del mismo día, en sus unidades reales (kW, hectolitros, m3, MJ): no se
transforma ni se escala nada (ADR 0010). Si el denominador es vacío, cero o menor al umbral, la razón
queda vacía (NaN), nunca cero ni infinito; la imputación se hace después, dentro del `Pipeline`.

Las justificaciones están en el notebook variables_planta.ipynb y en el ADR 0013.
"""

import pandas as pd

from limpieza import _exigir_columnas

# Un denominador menor a esta fracción de su mediana de entrenamiento (día casi parado) no sirve para dividir
FRACCION_DE_LA_MEDIANA = 0.10

PLANTA_KW = "Planta (Kw)"
HL_ENVASADOS = "Consolidado Produccion | Hl Cerveza Envasada"

# (nombre de la variable nueva, numerador, denominador)
RAZONES = [
    ("frio_sobre_planta", "Frio (Kw)", PLANTA_KW),
    ("frio_por_hl_envasado", "Frio (Kw)", HL_ENVASADOS),
    ("planta_por_hl_envasado", PLANTA_KW, HL_ENVASADOS),
    ("aire_por_hl_envasado", "Consolidado Aire | Aire Producido (M3)", HL_ENVASADOS),
    ("gas_por_hl_envasado", "Consolidado GasVapor | Gas Planta (Mj)", HL_ENVASADOS),
    ("servicios_sobre_planta", "Servicios (Kw)", PLANTA_KW),
]
NOMBRES_NUEVAS = [nombre for nombre, _, _ in RAZONES]
DENOMINADORES = sorted({denominador for _, _, denominador in RAZONES})

#----------------------------------------------------------------------------------------------------

def _columnas_necesarias():
    """Todas las columnas que las razones leen (numeradores y denominadores), sin repetir."""
    return sorted({columna for _, numerador, denominador in RAZONES for columna in (numerador, denominador)})


def aprender_reglas_de_planta(entrenamiento_limpio):
    """Mide, solo con el tramo de entrenamiento ya limpio, el umbral de cada denominador.

    Devuelve un diccionario con `umbral_denominador`: para cada denominador, FRACCION_DE_LA_MEDIANA
    veces su mediana de entrenamiento. Falla si algún denominador no tiene mediana positiva.
    """
    _exigir_columnas(entrenamiento_limpio, _columnas_necesarias(), "necesarias para las variables de planta")
    umbrales = {}
    for denominador in DENOMINADORES:
        mediana = pd.to_numeric(entrenamiento_limpio[denominador], errors="coerce").median()
        if not mediana > 0:                                       # también cubre NaN
            raise ValueError(f"La mediana de entrenamiento de «{denominador}» no es positiva ({mediana}): "
                             "no se puede fijar un umbral para dividir por esa columna.")
        umbrales[denominador] = FRACCION_DE_LA_MEDIANA * mediana
    return {"umbral_denominador": umbrales}


def crear_variables_planta(tabla, reglas):
    """Devuelve una copia de `tabla` con las razones de planta agregadas al final.

    La tabla debe estar ya limpia. No modifica la recibida, no cambia sus filas ni su índice y no
    transforma ni escala nada. Si falta una columna necesaria levanta un ValueError que dice cuáles.
    """
    _exigir_columnas(tabla, _columnas_necesarias(), "necesarias para las variables de planta")
    nueva = tabla.copy()
    valores = {c: pd.to_numeric(nueva[c], errors="coerce") for c in _columnas_necesarias()}
    for nombre, numerador, denominador in RAZONES:
        divisor = valores[denominador]
        divisor_valido = divisor.where(divisor >= reglas["umbral_denominador"][denominador])   # NaN si es vacío, cero o chico
        nueva[nombre] = valores[numerador] / divisor_valido
    return nueva
