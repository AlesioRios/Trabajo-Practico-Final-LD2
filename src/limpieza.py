"""Limpieza del dataset (Fase 2, consigna 2.1).

Se hace en dos pasos para no mirar nunca validación ni prueba al decidir:
  1. `aprender_reglas_de_limpieza(entrenamiento)` mide todo con el tramo de entrenamiento.
  2. `limpiar(tabla, reglas)` aplica esas reglas a cualquier tabla (entrenamiento, validación,
     prueba o datos nuevos en predict.py), sin recalcular nada.

Las decisiones y sus justificaciones están en el notebook preprocesamiento.ipynb (sección 3)
y en docs/decisions/0008-politica-de-limpieza.md.
"""

from math import isfinite

import pandas as pd

from auxiliar_functions import marcar_columnas_descartables, perfilar_columnas

# Columnas que no son mediciones de la planta: fecha, hora, controles del día y el objetivo
COLUMNAS_DE_CONTROL = ["DIA", "HORA", "bloque", "filas", "archivo", "completo", "plausible",
                       "lectura_inicial", "mayor_incremento", "menor_incremento",
                       "tipo_falla", "valido", "consumo_dia", "objetivo"]

# Una celda es absurda si supera este múltiplo del percentil 99 de su columna (medido en entrenamiento)
FACTOR_SOBRE_P99 = 5

#----------------------------------------------------------------------------------------------------

def _exigir_columnas(tabla, columnas, que_es):
    """Si a la tabla le falta alguna de las columnas pedidas, falla diciendo cuáles (nunca con un KeyError sin explicación)."""
    faltan = [c for c in columnas if c not in tabla.columns]
    if faltan:
        ejemplos = ", ".join(f"«{c}»" for c in faltan[:10])
        resto = f" (y {len(faltan) - 10} más)" if len(faltan) > 10 else ""
        raise ValueError(f"A la tabla le faltan {len(faltan)} columnas {que_es}: {ejemplos}{resto}. "
                         "Se buscan por nombre; revisar que el archivo tenga las hojas y columnas esperadas.")


def _es_columna_de_energia(columna):
    """Consolidado EE (sin ' | ' en el nombre) y las razones 'EE ... / Hl' del KPI, derivadas de esos kW."""
    return " | " not in columna or columna.startswith("Consolidado KPI | EE")


def anular_dias_invalidos(tabla, columnas):
    """Devuelve una copia con vacío (NaN) en `columnas` para los días marcados como no válidos."""
    tabla = tabla.copy()
    tabla.loc[~tabla["valido"], columnas] = float("nan")
    return tabla


def marcar_valores_absurdos(tabla, columnas, limite_superior, columnas_de_temperatura):
    """Devuelve dos tablas de verdadero/falso: celdas negativas (salvo temperaturas) y celdas sobre el límite."""
    valores = tabla[columnas].apply(pd.to_numeric, errors="coerce")
    negativas = (valores < 0) & ~valores.columns.isin(columnas_de_temperatura)
    limite_superior = pd.Series(limite_superior, dtype=float).reindex(columnas)
    muy_altas = valores > limite_superior            # comparar con NaN da falso: no se marca
    return negativas, muy_altas


def aprender_reglas_de_limpieza(entrenamiento):
    """Mide, solo con el tramo de entrenamiento, todo lo que necesita `limpiar`.

    Devuelve un diccionario serializable a JSON estándar (None indica ausencia de límite), con:
      - columnas_conservadas: las que pasan los criterios del ADR 0006 (paso 1);
      - columnas_de_energia: las que se anulan en los días inválidos (paso 2);
      - columnas_de_temperatura: las que pueden ser negativas (paso 3);
      - limite_superior: FACTOR_SOBRE_P99 veces el percentil 99 de cada columna (paso 3).
    """
    _exigir_columnas(entrenamiento, ["valido"], "de control que la limpieza necesita")
    candidatas = [c for c in entrenamiento.columns if c not in COLUMNAS_DE_CONTROL]
    perfil = marcar_columnas_descartables(perfilar_columnas(entrenamiento[candidatas]))
    conservadas = [c for c, se_conserva in zip(candidatas, perfil["Se conserva"]) if se_conserva]

    de_energia = [c for c in conservadas if _es_columna_de_energia(c)]
    # El límite se mide después de anular los días inválidos, igual que al limpiar
    sin_invalidos = anular_dias_invalidos(entrenamiento, de_energia)
    valores = sin_invalidos[conservadas].apply(pd.to_numeric, errors="coerce")
    limite = FACTOR_SOBRE_P99 * valores.quantile(0.99)
    return {
        "columnas_conservadas": conservadas,
        "columnas_de_energia": de_energia,
        "columnas_de_temperatura": [c for c in conservadas if "Temp" in c],
        "limite_superior": {c: float(v) if pd.notna(v) and isfinite(v) and v > 0 else None
                            for c, v in limite.items()},     # None: sin límite; JSON estándar
    }


def limpiar(tabla, reglas):
    """Aplica las reglas de limpieza: descarta columnas, anula días inválidos y valores absurdos.

    No borra filas ni imputa nada. No modifica la tabla recibida. Las columnas de más se ignoran;
    si falta alguna columna necesaria levanta un ValueError que dice cuáles faltan.
    Admite límites JSON (dict con None) y los límites históricos representados como Series.
    """
    conservadas = reglas["columnas_conservadas"]
    _exigir_columnas(tabla, ["valido"], "de control que la limpieza necesita")
    _exigir_columnas(tabla, conservadas, "de medición que el modelo necesita")
    mantener = [c for c in tabla.columns if c in COLUMNAS_DE_CONTROL or c in set(conservadas)]
    limpia = anular_dias_invalidos(tabla[mantener], reglas["columnas_de_energia"])
    negativas, muy_altas = marcar_valores_absurdos(limpia, conservadas, reglas["limite_superior"],
                                                   reglas["columnas_de_temperatura"])
    limpia[conservadas] = limpia[conservadas].apply(pd.to_numeric, errors="coerce").mask(negativas | muy_altas)
    return limpia
