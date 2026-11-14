"""Elección de las variables base para las interacciones (Fase 2, consigna 2.2).

`PolynomialFeatures` multiplica de a pares las variables que recibe, así que con ~280 columnas
crearía decenas de miles. Se eligen pocas variables base (por defecto 10) con dos criterios a la vez:
que se parezcan al frío de mañana y que no digan lo mismo entre sí. Todo se mide con correlación de
Spearman (por orden de los valores, no por su valor) y solo con el tramo de entrenamiento.

Hay dos métodos que se comparan:
  1. Por grupos: se agrupan las variables que se parecen entre sí (agrupamiento jerárquico con
     enlace completo sobre 1 - |Spearman|) y se toma la mejor de cada grupo.
  2. Voraz: se recorren las variables de mayor a menor parecido con el objetivo y se agrega cada una
     solo si su correlación con las ya elegidas es menor al umbral.

Después se hace en dos pasos, igual que la limpieza y las razones de planta:
  1. `aprender_reglas_de_interacciones(entrenamiento)` elige las variables base solo con entrenamiento.
  2. `crear_interacciones(tabla, reglas)` agrega, en cualquier tabla, los productos de a pares de esas
     variables con `PolynomialFeatures(degree=2, interaction_only=True)`: una columna por par, sin cuadrados.
Si alguno de los dos factores está vacío, el producto queda vacío (NaN); la imputación va en el `Pipeline`.

Las justificaciones están en el notebook variables_interacciones.ipynb y en el ADR 0014.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sklearn.preprocessing import PolynomialFeatures

from limpieza import COLUMNAS_DE_CONTROL, _exigir_columnas

UMBRAL_CORRELACION = 0.8       # dos variables con |Spearman| >= 0.8 se consideran redundantes
CANTIDAD_BASE = 10             # cuántas variables base se eligen
COLUMNA_OBJETIVO = "objetivo"  # el frío de mañana

#----------------------------------------------------------------------------------------------------

def columnas_candidatas(tabla):
    """Columnas que pueden ser variables base: numéricas, sin las de control y no constantes."""
    candidatas = []
    for columna in tabla.columns:
        if columna in COLUMNAS_DE_CONTROL or columna == "DIA":
            continue
        if not (pd.api.types.is_numeric_dtype(tabla[columna]) or pd.api.types.is_bool_dtype(tabla[columna])):
            continue
        if tabla[columna].nunique() > 1:                           # una columna constante no tiene correlación
            candidatas.append(columna)
    return candidatas


def spearman_con_objetivo(X, objetivo):
    """Spearman de cada columna de X con el objetivo, usando solo los días donde ambos tienen valor."""
    return X.apply(lambda columna: columna.corr(objetivo, method="spearman"))


def matriz_spearman(X):
    """Matriz de Spearman entre columnas (por pares de días con valor). Un par sin datos suficientes
    queda en 0 (sin relación conocida), para que el agrupamiento no falle con vacíos."""
    return X.corr(method="spearman").fillna(0.0)


def agrupar_por_correlacion(correlaciones, umbral=UMBRAL_CORRELACION):
    """Agrupa las variables con enlace completo sobre la distancia 1 - |Spearman|.

    Con enlace completo, dentro de un grupo TODOS los pares tienen |Spearman| >= umbral.
    Devuelve (serie con el número de grupo de cada variable, matriz de enlace).
    """
    distancia = 1 - correlaciones.abs().to_numpy()
    np.fill_diagonal(distancia, 0.0)
    distancia = (distancia + distancia.T) / 2                      # fuerza simetría exacta
    enlace = linkage(squareform(distancia, checks=False), method="complete")
    grupos = fcluster(enlace, t=1 - umbral, criterion="distance")
    return pd.Series(grupos, index=correlaciones.index, name="grupo"), enlace


def elegir_por_grupos(parecido, grupos, cantidad=CANTIDAD_BASE):
    """Método 1: la variable de mayor |Spearman| con el objetivo de cada grupo, y de ellas las `cantidad` mejores."""
    tabla = pd.DataFrame({"parecido": parecido, "grupo": grupos})
    tabla["abs"] = tabla["parecido"].abs()
    mejores = tabla.sort_values("abs", ascending=False).groupby("grupo").head(1)
    return list(mejores.sort_values("abs", ascending=False).index[:cantidad])


def elegir_voraz(parecido, correlaciones, cantidad=CANTIDAD_BASE, umbral=UMBRAL_CORRELACION):
    """Método 2: de mayor a menor |Spearman| con el objetivo, se agrega la variable solo si su |Spearman|
    con todas las ya elegidas es menor al umbral."""
    elegidas = []
    for columna in parecido.abs().sort_values(ascending=False).index:
        if all(abs(correlaciones.loc[columna, otra]) < umbral for otra in elegidas):
            elegidas.append(columna)
        if len(elegidas) == cantidad:
            break
    return elegidas


#----------------------------------------------------------------------------------------------------

def aprender_reglas_de_interacciones(entrenamiento, cantidad=CANTIDAD_BASE, umbral=UMBRAL_CORRELACION):
    """Elige, solo con el tramo de entrenamiento, las variables base de las interacciones.

    `entrenamiento` es la tabla de entrenamiento ya limpia y con las variables temporales y de planta
    (incluye la columna `objetivo`). Usa el método voraz (ADR 0014). Devuelve un diccionario con
    `variables_base` (lista ordenada de mayor a menor parecido con el objetivo) y el umbral usado.
    """
    _exigir_columnas(entrenamiento, [COLUMNA_OBJETIVO], "necesarias para elegir las variables base")
    X = entrenamiento[columnas_candidatas(entrenamiento.drop(columns=[COLUMNA_OBJETIVO]))].astype(float)
    parecido = spearman_con_objetivo(X, entrenamiento[COLUMNA_OBJETIVO])
    base = elegir_voraz(parecido, matriz_spearman(X), cantidad, umbral)
    return {"variables_base": base, "umbral_correlacion": umbral}


def nombres_de_interacciones(variables_base):
    """Nombre de cada producto, en el mismo orden en que `PolynomialFeatures` los genera: «A x B»."""
    n = len(variables_base)
    return [f"{variables_base[i]} x {variables_base[j]}" for i in range(n) for j in range(i + 1, n)]


def construir_mapa_interacciones(reglas):
    """Obtiene componentes desde reglas de variables o de interacciones, sin aprender.

    Usa variables_base y el mismo orden/nombres que crear_interacciones. Devuelve
    un diccionario de listas serializable a JSON; no interpreta nombres de columnas.
    """
    if not isinstance(reglas, dict):
        raise ValueError("Faltan reglas de interacciones con variables_base.")
    interacciones = reglas.get("interacciones", reglas)
    if not isinstance(interacciones, dict) or "variables_base" not in interacciones:
        raise ValueError("Falta variables_base en las reglas de interacciones.")
    base = interacciones["variables_base"]
    if not isinstance(base, list) or any(not isinstance(c, str) or not c for c in base):
        raise ValueError("variables_base debe ser una lista de nombres no vacíos.")
    if len(set(base)) != len(base):
        raise ValueError("variables_base contiene nombres repetidos.")
    if any(c in COLUMNAS_DE_CONTROL for c in base):
        raise ValueError("Una variable base es objetivo o control.")
    nombres = nombres_de_interacciones(base)
    if len(set(nombres)) != len(nombres) or set(nombres) & set(base):
        raise ValueError("Los nombres de interacciones son ambiguos o colisionan con sus bases.")
    return {nombre: list(par) for nombre, par in zip(nombres, combinations(base, 2))}


def crear_interacciones(tabla, reglas):
    """Devuelve una copia de `tabla` con una columna nueva por cada par de variables base (su producto).

    Usa `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)`, que no acepta vacíos:
    se calcula con los vacíos en cero y después se anula (NaN) todo producto en que alguno de los dos
    factores estaba vacío, para no inventar un valor. No cambia filas, índice ni columnas originales.
    Adjunta el mapa JSON en attrs["componentes_interacciones"] para el selector.
    """
    componentes = construir_mapa_interacciones(reglas)
    base = reglas["variables_base"]
    _exigir_columnas(tabla, base, "necesarias para las interacciones")
    nuevas = nombres_de_interacciones(base)
    repetidas = [c for c in nuevas if c in tabla.columns]
    if repetidas:
        raise ValueError(f"La tabla ya tiene columnas de interacción (por ejemplo «{repetidas[0]}»).")

    valores = tabla[base].apply(pd.to_numeric, errors="coerce").astype(float)
    vacios = valores.isna().to_numpy()
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    resultado = poly.fit_transform(valores.fillna(0.0).to_numpy())
    n = len(base)
    productos = resultado[:, n:]                                # las primeras n columnas son las originales
    for k, potencias in enumerate(poly.powers_[n:]):            # cada fila de powers_ marca los dos factores
        i, j = np.flatnonzero(potencias)
        productos[vacios[:, i] | vacios[:, j], k] = np.nan
    nueva = pd.concat([tabla, pd.DataFrame(productos, index=tabla.index, columns=nuevas)], axis=1)
    nueva.attrs = tabla.attrs.copy()
    nueva.attrs["componentes_interacciones"] = componentes
    return nueva
