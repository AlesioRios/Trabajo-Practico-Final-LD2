"""Variantes de preprocesamiento para comparar con Ridge y exportar datos."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from auxiliar_functions import calcular_baselines, guardar_huellas
from preprocessing_pipeline import dividir_por_fecha
from variables_planta import NOMBRES_NUEVAS

VARIANTE_POR_DEFECTO = "A"
COLUMNAS_CICLICAS = ("dia_anio_seno", "dia_anio_coseno")   # ya están entre -1 y 1: no se transforman ni escalan
ALPHAS_POR_DEFECTO = (0.1, 1, 10, 100)           # grilla acordada para la regularización de Ridge
ORDEN_DE_SIMPLICIDAD = ("A", "B", "D", "C", "E", "F")   # de la que menos pasos tiene a la que más


def separar_xy(tabla, columnas_elegidas):
    """Devuelve copias de las entradas y el objetivo para las filas etiquetadas."""
    filas = tabla["objetivo"].notna()
    return tabla.loc[filas, columnas_elegidas].copy(), tabla.loc[filas, "objetivo"].copy()


def aprender_reglas_de_grupos(entrenamiento, columnas):
    """Devuelve grupos por nombre y por valores observados en entrenamiento.

    Cíclicas son solo las dos columnas exactas del seno y el coseno del día del año: un producto como
    "dia_anio_coseno x Agua" empieza igual pero tiene la escala del agua y va con las numéricas.
    """
    grupos = {nombre: [] for nombre in ("binarias", "ciclicas", "razones", "numericas")}
    for columna in columnas:
        valores = entrenamiento[columna].dropna()
        if columna in COLUMNAS_CICLICAS:
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


#----------------------------------------------------------------------------------------------------
# Comparación de variantes del Pipeline con Ridge como juez (no es el modelo final: ese lo elige la Fase 3).

def _error_medio(variante, alpha, columnas, X_ajuste, y_ajuste, X_evaluacion, y_evaluacion):
    """Ajusta la variante solo con las filas de ajuste y devuelve el MAE sobre las de evaluación."""
    grupos = aprender_reglas_de_grupos(X_ajuste, columnas)                  # los grupos también salen del ajuste
    modelo = armar_pipeline(variante, columnas, grupos)
    modelo.set_params(regressor__alpha=alpha).fit(X_ajuste, y_ajuste)
    return float((y_evaluacion - modelo.predict(X_evaluacion)).abs().mean())


def comparar_variantes(entrenamiento, columnas, variantes=ORDEN_DE_SIMPLICIDAD,
                       alphas=ALPHAS_POR_DEFECTO, pliegues=5):
    """Mide cada variante y cada alpha en pliegues cronológicos del tramo de entrenamiento.

    Usa `TimeSeriesSplit` (nunca mezcla el orden). Devuelve dos tablas:
      - `detalle`: una fila por variante, alpha y pliegue, con su MAE en kW;
      - `referencias`: por pliegue, filas, fechas, consumo y MAE de los dos baselines, medidos sobre
        exactamente las mismas filas de evaluación que usa Ridge.
    La última fila del entrenamiento se descarta: su objetivo es el consumo del primer día de validación.
    Si la última fila de ajuste es la anterior a la primera de evaluación, también se descarta: su objetivo
    es justamente el consumo de ese primer día de evaluación, que ya es entrada de esa fila.
    """
    tabla = entrenamiento.iloc[:-1]
    X, y = separar_xy(tabla, columnas)
    base = calcular_baselines(entrenamiento)
    posiciones = tabla.index.get_indexer(X.index)             # lugar de cada fila en la tabla, con cualquier índice
    detalle, referencias = [], []
    for numero, (ajuste, evaluacion) in enumerate(TimeSeriesSplit(pliegues).split(X), 1):
        if posiciones[ajuste[-1]] + 1 == posiciones[evaluacion[0]]:
            ajuste = ajuste[:-1]
        disponibles = base.loc[X.index[evaluacion]].dropna().index        # filas donde existen ambos baselines
        X_ev, y_ev = X.loc[disponibles], y.loc[disponibles]
        referencias.append({
            "pliegue": numero, "filas_ajuste": len(ajuste), "filas_evaluacion": len(y_ev),
            "inicio": tabla.loc[disponibles, "DIA"].min(), "fin": tabla.loc[disponibles, "DIA"].max(),
            "consumo_medio": y_ev.mean(), "desvio_consumo": y_ev.std(),
            "persistencia": (y_ev - base.loc[disponibles, "persistencia"]).abs().mean(),
            "media_movil": (y_ev - base.loc[disponibles, "media_movil"]).abs().mean(),
        })
        for variante in variantes:
            for alpha in alphas:
                detalle.append({"variante": variante, "alpha": alpha, "pliegue": numero,
                                "mae": _error_medio(variante, alpha, columnas, X.iloc[ajuste], y.iloc[ajuste],
                                                    X_ev, y_ev)})
    return pd.DataFrame(detalle), pd.DataFrame(referencias).set_index("pliegue")


def resumir_comparacion(detalle):
    """Deja una fila por variante, con el alpha de menor MAE promedio y el desvío entre pliegues."""
    por_alpha = detalle.groupby(["variante", "alpha"])["mae"].agg(["mean", "std"]).reset_index()
    mejores = por_alpha.sort_values("mean").drop_duplicates("variante").set_index("variante")
    return mejores.rename(columns={"mean": "mae_promedio", "std": "desvio"}).sort_index()


def elegir_variante(resumen):
    """Regla de un desvío: la variante más simple cuyo MAE promedio no supera al mejor más su desvío.

    Devuelve la variante elegida y el límite que se aplicó.
    """
    mejor = resumen["mae_promedio"].idxmin()
    limite = float(resumen.loc[mejor, "mae_promedio"] + resumen.loc[mejor, "desvio"])
    for variante in ORDEN_DE_SIMPLICIDAD:
        if variante in resumen.index and resumen.loc[variante, "mae_promedio"] <= limite:
            return variante, limite
    raise ValueError("Ninguna variante entra en el límite: el resumen está vacío.")


def evaluar_en_validacion(entrenamiento, validacion, columnas, variante, alpha):
    """Ajusta con todo el entrenamiento y mide en validación junto con los baselines. Solo informa.

    Los baselines se calculan sobre entrenamiento y validación juntos para que la media móvil de los primeros
    días de validación tenga su historia. La última fila de cada tramo se descarta (su objetivo es del tramo
    siguiente), igual que en `comparar_variantes`.
    """
    X, y = separar_xy(entrenamiento.iloc[:-1], columnas)
    X_val, y_val = separar_xy(validacion.iloc[:-1], columnas)
    base = calcular_baselines(pd.concat([entrenamiento, validacion])).loc[X_val.index].dropna()
    X_val, y_val = X_val.loc[base.index], y_val.loc[base.index]
    error = _error_medio(variante, alpha, columnas, X, y, X_val, y_val)
    persistencia = float((y_val - base["persistencia"]).abs().mean())
    media_movil = float((y_val - base["media_movil"]).abs().mean())
    mejor_baseline = min(persistencia, media_movil)
    return {"filas": len(y_val), "mae": error, "persistencia": persistencia, "media_movil": media_movil,
            "mejora_relativa": 100 * (mejor_baseline - error) / mejor_baseline}
