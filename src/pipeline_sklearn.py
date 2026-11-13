"""Variantes de preprocesamiento para comparar con Ridge y exportar datos."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from auxiliar_functions import guardar_huellas
from preprocessing_pipeline import dividir_por_fecha
from variables_planta import NOMBRES_NUEVAS

VARIANTE_POR_DEFECTO = "A"


def separar_xy(tabla, columnas_elegidas):
    """Devuelve copias de las entradas y el objetivo para las filas etiquetadas."""
    filas = tabla["objetivo"].notna()
    return tabla.loc[filas, columnas_elegidas].copy(), tabla.loc[filas, "objetivo"].copy()


def aprender_reglas_de_grupos(entrenamiento, columnas):
    """Devuelve grupos por nombre y por valores observados en entrenamiento."""
    grupos = {nombre: [] for nombre in ("binarias", "ciclicas", "razones", "numericas")}
    for columna in columnas:
        valores = entrenamiento[columna].dropna()
        if columna.startswith("dia_anio_"):
            grupo = "ciclicas"
        elif columna in NOMBRES_NUEVAS:
            grupo = "razones"
        elif len(valores) and valores.isin([0, 1]).all():
            grupo = "binarias"
        else:
            grupo = "numericas"
        grupos[grupo].append(columna)
    return grupos


def _mediana(indicador=False):
    return SimpleImputer(strategy="median", add_indicator=indicador, keep_empty_features=True)


def _yeo(estandarizar=True):
    return PowerTransformer(method="yeo-johnson", standardize=estandarizar)


def armar_pipeline(variante=VARIANTE_POR_DEFECTO, columnas=None, reglas_grupos=None):
    """Devuelve un pipeline sin ajustar. E recibe los grupos de entrenamiento."""
    if not isinstance(variante, str):
        numericas, binarias = list(variante), list(columnas)
        columnas = numericas + binarias
        variante, reglas_grupos = "F", {"binarias": binarias}
    if columnas is None:
        raise ValueError("Indicá las columnas de entrada.")
    if variante == "E":
        if reglas_grupos is None:
            raise ValueError("La variante E requiere grupos de entrenamiento.")
        bloques = [
            ("binarias", "passthrough", reglas_grupos["binarias"]),
            ("ciclicas", _mediana(), reglas_grupos["ciclicas"]),
            ("razones", Pipeline([
                ("imputar", _mediana(True)), ("transformar", _yeo(False)),
                ("escalar", RobustScaler()),
            ]), reglas_grupos["razones"]),
            ("numericas", Pipeline([
                ("imputar", _mediana()), ("transformar", _yeo()),
            ]), reglas_grupos["numericas"]),
        ]
    else:
        pasos = {
            "A": [("imputar", _mediana()), ("escalar", StandardScaler())],
            "B": [("imputar", _mediana()), ("transformar", _yeo())],
            "C": [("imputar", _mediana(True)), ("transformar", _yeo())],
            "D": [("imputar", _mediana()), ("transformar", _yeo(False)),
                  ("escalar", RobustScaler())],
            "F": [("escalar_distancias", StandardScaler()),
                  ("imputar", KNNImputer(n_neighbors=5, keep_empty_features=True)),
                  ("transformar", _yeo(False)), ("escalar", StandardScaler())],
        }
        if variante not in pasos:
            raise ValueError(f"Variante desconocida: {variante}")
        binarias = reglas_grupos["binarias"] if variante == "F" and reglas_grupos else []
        numericas = [columna for columna in columnas if columna not in binarias]
        bloques = [("numericas", Pipeline(pasos[variante]), numericas),
                   ("binarias", "passthrough", binarias)]
    return Pipeline([("preprocessor", ColumnTransformer(bloques)), ("regressor", Ridge())])


def guardar_dataset_final(tabla_con_variables, columnas, carpeta_salida):
    """Escribe las filas etiquetadas en sus unidades originales y guarda la huella."""
    salida = Path(carpeta_salida)
    salida.mkdir(parents=True, exist_ok=True)
    campos = ["DIA", *columnas, "objetivo"]
    nombres = ("entrenamiento", "validacion", "prueba")
    partes = [
        tramo.loc[tramo["objetivo"].notna(), campos].assign(tramo=nombre)
        for nombre, tramo in zip(nombres, dividir_por_fecha(tabla_con_variables))
    ]
    ruta = salida / "dataset_final.csv"
    pd.concat(partes).to_csv(ruta, index=False, lineterminator="\n")
    guardar_huellas([ruta], salida.parent / "checksums.json", salida.parent)
    return ruta


def escribir_linaje(carpeta_salida, reglas, columnas, shapes):
    """Escribe el origen, las reglas, las columnas y las dimensiones del dataset."""
    salida = Path(carpeta_salida)
    salida.mkdir(parents=True, exist_ok=True)
    ruta = salida / "data_lineage.json"
    datos = {
        "script": "src/pipeline_sklearn.py", "fecha": date.today().isoformat(),
        "reglas": reglas, "columnas": list(columnas), "shapes": shapes,
    }
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return ruta
