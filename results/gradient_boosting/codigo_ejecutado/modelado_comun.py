"""Piezas comunes de la Fase 3: cargar el dataset final, medir los baselines y registrar experimentos.

Todos los modelos usan estas funciones para que se midan sobre las mismas filas y se registren igual
(ADR 0018). Reglas que este módulo hace cumplir:
  - El tramo de prueba NO se carga ni se registra salvo que se pida de forma explícita: se mira una sola vez,
    al final, con el modelo ya elegido por su error en validación.
  - Los modelos se comparan con los baselines sobre las MISMAS filas (las que tienen los dos baselines).
  - Cada ejecución queda como una fila de `results/experiment_logs.csv`, sin pisar las anteriores.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from auxiliar_functions import SEMILLA

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATASET_FINAL = RAIZ / "data" / "processed" / "dataset_final.csv"
RUTA_LINAJE = RAIZ / "data" / "processed" / "data_lineage.json"
RUTA_LOG = RAIZ / "results" / "experiment_logs.csv"

TRAMOS = ("entrenamiento", "validacion", "prueba")
TRAMOS_DE_TRABAJO = ("entrenamiento", "validacion")      # los que cualquiera puede usar para elegir modelos
COLUMNAS_DEL_LOG = ["fecha_hora", "modelo", "tramo", "filas", "hiperparametros", "mae", "rmse", "r2", "semilla",
                    "commit", "arbol_sin_commitear", "notas"]


def cargar_dataset_final(ruta=RUTA_DATASET_FINAL, ruta_linaje=RUTA_LINAJE, incluir_prueba=False):
    """Lee `dataset_final.csv` y devuelve un diccionario con un elemento por tramo.

    Cada tramo es un diccionario con: `X` (las variables elegidas, en sus unidades originales y con sus vacíos),
    `y` (el objetivo: el consumo de frío del día siguiente), `fechas`, `persistencia` y `media_movil` (los dos
    baselines, una predicción por fila). Los índices son los de las filas del CSV: únicos, en orden cronológico.
    La columna `DIA` no es única, así que nunca se usa como clave.

    El tramo de prueba solo se incluye con `incluir_prueba=True`: usarlo es una decisión de la evaluación final.
    """
    linaje = json.loads(Path(ruta_linaje).read_text(encoding="utf-8"))
    tabla = pd.read_csv(ruta, parse_dates=["DIA"])
    columnas = linaje["columnas"]
    persistencia, media_movil = linaje["baselines"]["persistencia"], linaje["baselines"]["media_movil"]
    faltan = [c for c in [*columnas, persistencia, media_movil, "objetivo", "tramo"] if c not in tabla.columns]
    if faltan:
        raise ValueError(f"Al dataset final le faltan columnas que pide el linaje: {faltan}")
    desconocidos = sorted(set(tabla["tramo"]) - set(TRAMOS))
    if desconocidos:
        raise ValueError(f"El dataset final trae tramos desconocidos: {desconocidos}")

    datos = {}
    for tramo in (TRAMOS if incluir_prueba else TRAMOS_DE_TRABAJO):
        parte = tabla.loc[tabla["tramo"] == tramo]
        datos[tramo] = {"X": parte[columnas], "y": parte["objetivo"], "fechas": parte["DIA"],
                        "persistencia": parte[persistencia], "media_movil": parte[media_movil]}
    return datos


def filas_comunes(tramo):
    """Máscara de las filas del tramo donde existen el objetivo y las dos predicciones de los baselines.

    Es el conjunto sobre el que se comparan todos los modelos con los baselines: evaluar un modelo en más
    filas que los baselines (o en otras) haría que los MAE no se puedan comparar.
    """
    return tramo["y"].notna() & tramo["persistencia"].notna() & tramo["media_movil"].notna()


def evaluar_baselines(tramo):
    """MAE de la persistencia y de la media móvil sobre las filas comunes del tramo."""
    filas = filas_comunes(tramo)
    y = tramo["y"][filas]
    return {"filas": int(filas.sum()),
            "persistencia": float((y - tramo["persistencia"][filas]).abs().mean()),
            "media_movil": float((y - tramo["media_movil"][filas]).abs().mean())}


def calcular_metricas(y_real, y_predicho):
    """MAE, RMSE y R² de una predicción. Falla si hay vacíos o si las longitudes no coinciden."""
    y_real, y_predicho = np.asarray(y_real, dtype=float), np.asarray(y_predicho, dtype=float)
    if y_real.shape != y_predicho.shape or y_real.ndim != 1 or len(y_real) == 0:
        raise ValueError("y_real e y_predicho tienen que ser vectores no vacíos de la misma longitud.")
    if np.isnan(y_real).any() or np.isnan(y_predicho).any():
        raise ValueError("Hay valores vacíos en y_real o en y_predicho: se evalúa solo donde hay predicción.")
    return {"mae": float(mean_absolute_error(y_real, y_predicho)),
            "rmse": float(np.sqrt(mean_squared_error(y_real, y_predicho))),
            "r2": float(r2_score(y_real, y_predicho))}


def _estado_de_git(carpeta=RAIZ):
    """Hash del commit actual y si hay cambios sin commitear. Si no se puede saber, devuelve ('desconocido', None).

    El hash (código único del último commit) dice con qué versión del código se corrió el experimento. Los
    cambios sin commitear son archivos editados que todavía no están en ningún commit: si los hay, el hash no
    alcanza para reconstruir lo que se ejecutó. El propio registro de experimentos no cuenta como cambio, porque
    cada experimento lo modifica al agregar su fila.
    """
    sin_el_registro = ["--", ".", ":(exclude)results/experiment_logs.csv"]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=carpeta, capture_output=True, text=True,
                                timeout=20, check=True).stdout.strip()
        sucio = bool(subprocess.run(["git", "status", "--porcelain", *sin_el_registro], cwd=carpeta,
                                    capture_output=True, text=True, timeout=20, check=True).stdout.strip())
        return commit, sucio
    except (OSError, subprocess.SubprocessError):
        return "desconocido", None


def registrar_experimento(modelo, hiperparametros, y_real, y_predicho, tramo, notas="", ruta=RUTA_LOG,
                          evaluacion_final=False):
    """Agrega una fila a `results/experiment_logs.csv` con las métricas de una ejecución y devuelve esa fila.

    `modelo` es un nombre corto ("xgboost", "ridge", "persistencia"...); `hiperparametros`, un diccionario con lo
    que se usó; `tramo`, el tramo donde se midió. Registrar el tramo de prueba exige `evaluacion_final=True`.
    Nunca pisa las filas anteriores. Si un modelo se ejecuta dos veces se registran dos filas.
    """
    if tramo not in TRAMOS:
        raise ValueError(f"Tramo desconocido {tramo!r}; los válidos son {TRAMOS}.")
    if tramo == "prueba" and not evaluacion_final:
        raise ValueError("El tramo de prueba solo se registra en la evaluación final del modelo ya elegido "
                         "(evaluacion_final=True): elegir un modelo mirando la prueba se penaliza.")
    commit, sucio = _estado_de_git()
    fila = {"fecha_hora": datetime.now().isoformat(timespec="seconds"), "modelo": modelo, "tramo": tramo,
            "filas": len(np.asarray(y_real)),
            "hiperparametros": json.dumps(hiperparametros, sort_keys=True, default=str, ensure_ascii=False),
            **calcular_metricas(y_real, y_predicho), "semilla": SEMILLA, "commit": commit,
            "arbol_sin_commitear": sucio, "notas": notas}
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        encabezado = ruta.read_text(encoding="utf-8").splitlines()[0].split(",")
        if encabezado != COLUMNAS_DEL_LOG:
            raise ValueError(f"{ruta} tiene otras columnas que las esperadas: {encabezado}")
    pd.DataFrame([fila], columns=COLUMNAS_DEL_LOG).to_csv(ruta, mode="a", header=not ruta.exists(), index=False,
                                                          lineterminator="\n")
    return fila


def validacion_cruzada_temporal(pliegues=5):
    """Divisor de pliegues cronológicos que usan TODOS los modelos para ajustar hiperparámetros en entrenamiento.

    `TimeSeriesSplit` nunca mezcla el orden: cada pliegue evalúa sobre días posteriores a los que ajusta. Con
    `gap=1` se descarta la última fila de ajuste, porque su objetivo (el consumo de mañana) es el consumo de hoy de
    la primera fila de evaluación. Se usa con `divisor.split(X)` sobre `datos["entrenamiento"]["X"]` y, dentro de
    cada pliegue, se mide solo sobre las filas comunes (`filas_comunes`).
    """
    return TimeSeriesSplit(n_splits=pliegues, gap=1)


def registrar_baselines(datos, tramos=TRAMOS_DE_TRABAJO, ruta=RUTA_LOG):
    """Registra la persistencia y la media móvil de cada tramo como los primeros experimentos del log.

    Es idempotente: si el log ya tiene un baseline de ese tramo con la misma marca, no lo repite, así que
    se puede volver a ejecutar el notebook sin llenar el archivo de filas iguales. Devuelve las filas nuevas.
    """
    marca = "baseline sobre las filas comunes"
    previos = pd.read_csv(ruta) if Path(ruta).exists() else pd.DataFrame(columns=COLUMNAS_DEL_LOG)
    ya_registrados = set(map(tuple, previos.loc[previos["notas"] == marca, ["modelo", "tramo"]].values.tolist()))
    filas = []
    for tramo in tramos:
        parte, comunes = datos[tramo], filas_comunes(datos[tramo])
        for nombre, hiperparametros in (("persistencia", {}), ("media_movil", {"ventana": 7, "minimo_dias": 4})):
            if (nombre, tramo) in ya_registrados:
                continue
            filas.append(registrar_experimento(
                nombre, hiperparametros, parte["y"][comunes], parte[nombre][comunes], tramo,
                notas=marca, ruta=ruta, evaluacion_final=(tramo == "prueba")))
    return filas
