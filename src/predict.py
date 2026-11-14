"""Predicción del consumo de Frio (kW) del día siguiente (Fase 4).

Uso: python src/predict.py <ruta_al_excel>
Genera results/predicciones.csv con una fila por día: `fecha` es el día D y `prediccion_frio_kw`, el consumo
previsto para el día D+1.

Cómo funciona (ADR 0022):
  1. `load_model` busca la versión en models/model_registry.json y carga el Pipeline completo (imputación,
     escalado si corresponde, y modelo) con joblib.
  2. `load_and_preprocess` reusa preprocessing_pipeline.py con las reglas aprendidas en entrenamiento
     (data/processed/data_lineage.json). Lee del Excel solo las hojas y columnas que las variables del modelo
     necesitan, siempre por nombre; lo demás se ignora. Si falta algo, lo dice con `ErrorDeEntrada`.
  3. `predict_consumption` predice todos los días. Los que tienen alguna variable vacía (por ejemplo, los
     primeros días del archivo, sin historia para los rezagos) se predicen igual y se informan aparte.

Las funciones no imprimen ni preguntan nada: los avisos van solo en `main`, para poder reutilizarlas desde otro programa.
No usa internet.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

from preprocessing_pipeline import ErrorDeEntrada, preparar_para_prediccion

RAIZ = Path(__file__).resolve().parents[1]
RUTA_REGISTRO = RAIZ / "models" / "model_registry.json"
RUTA_LINAJE = RAIZ / "data" / "processed" / "data_lineage.json"
RUTA_PREDICCIONES = RAIZ / "results" / "predicciones.csv"


class ErrorDeModelo(RuntimeError):
    """No se pudo cargar el modelo o sus reglas: el mensaje dice qué archivo o versión falla."""


#----------------------------------------------------------------------------------------------------
# Modelo y reglas

def _version_como_tupla(clave):
    """'1.10.0' o 'v1.10.0' -> (1, 10, 0). Devuelve None si la clave no es una versión semántica."""
    coincide = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", str(clave).strip())
    return tuple(int(parte) for parte in coincide.groups()) if coincide else None


def _leer_json(ruta, que_es):
    if not Path(ruta).is_file():
        raise ErrorDeModelo(f"No se encontró {que_es}: {str(ruta)!r}.")
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ErrorDeModelo(f"{que_es.capitalize()} {str(ruta)!r} no es un JSON válido: {error}") from error


def elegir_version(registro, version="latest"):
    """Devuelve la clave del registro que corresponde a `version`.

    'latest' es la versión más alta, comparada como números (1.10.0 es mayor que 1.9.0; como texto sería al revés).
    Cualquier otro valor se busca como clave exacta, con o sin 'v' adelante.
    """
    versiones = {clave: _version_como_tupla(clave) for clave in registro if _version_como_tupla(clave)}
    if version == "latest":
        if not versiones:
            raise ErrorDeModelo("El registro de modelos no tiene ninguna versión con formato X.Y.Z.")
        return max(versiones, key=versiones.get)
    if version in registro:
        return version
    buscada = _version_como_tupla(version)
    for clave, tupla in versiones.items():
        if buscada and tupla == buscada:
            return clave
    raise ErrorDeModelo(f"La versión {version!r} no está en el registro. Versiones: {sorted(registro)}.")


def _cargar_modelo_y_entrada(version="latest", ruta_registro=None):
    """Devuelve (modelo, clave de la versión, entrada del registro)."""
    ruta_registro = Path(ruta_registro or RUTA_REGISTRO)
    registro = _leer_json(ruta_registro, "el registro de modelos")
    clave = elegir_version(registro, version)
    entrada = registro[clave]
    if "path" not in entrada:
        raise ErrorDeModelo(f"La versión {clave!r} del registro no tiene el campo 'path'.")
    ruta_modelo = Path(entrada["path"])
    if not ruta_modelo.is_absolute():
        ruta_modelo = RAIZ / ruta_modelo                           # las rutas del registro son relativas a la raíz del repo
    if not ruta_modelo.is_file():
        raise ErrorDeModelo(f"El registro apunta a {str(ruta_modelo)!r} (versión {clave}) pero ese archivo no existe.")
    return joblib.load(ruta_modelo), clave, entrada


def load_model(version="latest"):
    """Carga el modelo (el Pipeline completo) de la versión pedida desde models/model_registry.json."""
    return _cargar_modelo_y_entrada(version)[0]


def columnas_del_modelo(modelo, entrada=None):
    """Nombres y orden de las columnas con que se entrenó el modelo.

    Se leen del propio modelo (`feature_names_in_`), así `predict.py` sirve igual con 14 o con más variables. Si el modelo
    no las guardó, se usan las del registro (campo `columnas`).
    """
    columnas = getattr(modelo, "feature_names_in_", None)
    if columnas is None and entrada:
        columnas = entrada.get("columnas")
    if columnas is None:
        raise ErrorDeModelo("No se pueden saber las columnas del modelo: no tiene feature_names_in_ y el registro no "
                            "tiene el campo 'columnas'.")
    return [str(c) for c in columnas]


def cargar_reglas(ruta_linaje=None):
    """Reglas de limpieza y de variables aprendidas en entrenamiento (data/processed/data_lineage.json)."""
    reglas = _leer_json(ruta_linaje or RUTA_LINAJE, "el linaje de los datos")["reglas"]
    if not {"limpieza", "variables"} <= set(reglas):
        raise ErrorDeModelo("El linaje no tiene las reglas de 'limpieza' y 'variables'.")
    return reglas


#----------------------------------------------------------------------------------------------------
# Datos nuevos y predicción

def load_and_preprocess(filepath, columnas=None):
    """Lee el Excel y aplica el preprocesamiento de entrenamiento. Devuelve (X, fechas).

    `X` tiene una fila por día y solo las columnas que pide el modelo, en su orden, en unidades originales y
    con los vacíos que haya (el Pipeline guardado los imputa). `fechas` es la fecha D de cada fila.
    Sin `columnas` se usan las del modelo más reciente del registro.
    """
    if columnas is None:
        modelo, _, entrada = _cargar_modelo_y_entrada()
        columnas = columnas_del_modelo(modelo, entrada)
    X, fechas = preparar_para_prediccion(filepath, columnas, cargar_reglas())
    return X.reset_index(drop=True), fechas.reset_index(drop=True)


def predict_consumption(filepath, version="latest"):
    """Predice el consumo de Frio (kW) del día siguiente para cada día del archivo.

    Devuelve un DataFrame con `fecha` (día D) y `prediccion_frio_kw` (consumo previsto de D+1). Predice todos los días.
    Los que tienen alguna variable vacía quedan listados en `resultado.attrs["dias_incompletos"]`
    (columnas `fecha` y `variables_vacias`), sin agregar columnas al resultado.
    """
    modelo, _, entrada = _cargar_modelo_y_entrada(version)
    X, fechas = load_and_preprocess(filepath, columnas_del_modelo(modelo, entrada))
    if X.empty:
        raise ErrorDeEntrada(f"No se encontraron días en {Path(filepath).name!r}: no hay nada para predecir.")
    resultado = pd.DataFrame({"fecha": fechas, "prediccion_frio_kw": modelo.predict(X)})
    vacias = X.isna()
    incompletos = vacias.any(axis=1)
    resultado.attrs["dias_incompletos"] = pd.DataFrame({
        "fecha": fechas[incompletos],
        "variables_vacias": [", ".join(X.columns[fila]) for fila in vacias[incompletos].to_numpy()]})
    return resultado


#----------------------------------------------------------------------------------------------------
# Línea de comandos

def main(argumentos=None):
    """Predice sobre el Excel indicado, avisa qué días tenían datos incompletos y guarda results/predicciones.csv.

    Devuelve 0 si salió bien y 1 si el archivo o el modelo tienen un problema (el mensaje va a stderr y no se escribe el CSV).
    """
    analizador = argparse.ArgumentParser(description="Predice el consumo de Frio (kW) del día siguiente.")
    analizador.add_argument("excel", help="ruta al archivo Excel con los datos de la planta")
    analizador.add_argument("--salida", default=str(RUTA_PREDICCIONES), help="CSV de salida (por defecto results/predicciones.csv)")
    opciones = analizador.parse_args(argumentos)
    for flujo in (sys.stdout, sys.stderr):          # una consola con otra codificación no debe hacer fallar un aviso
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    try:
        resultado = predict_consumption(opciones.excel)
    except (ErrorDeEntrada, ErrorDeModelo) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:                      # lo no previsto también termina con un mensaje, sin CSV a medias
        print(f"Error inesperado al procesar {opciones.excel!r} ({type(error).__name__}): {error}", file=sys.stderr)
        return 1

    incompletos = resultado.attrs["dias_incompletos"]
    if len(incompletos):
        print(f"Atención: {len(incompletos)} de {len(resultado)} días tienen datos incompletos "
              "(se predicen igual, pero son menos confiables):")
        for _, fila in incompletos.iterrows():
            faltan = fila["variables_vacias"].split(", ")
            resumen = ", ".join(faltan[:3]) + (f" y {len(faltan) - 3} más" if len(faltan) > 3 else "")
            cuantas = "1 variable vacía" if len(faltan) == 1 else f"{len(faltan)} variables vacías"
            print(f"  {fila['fecha']:%Y-%m-%d}: {cuantas} ({resumen})")
        print()
    print(resultado.to_string(index=False))

    salida = Path(opciones.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(salida, index=False, lineterminator="\n")
    print(f"\nPredicciones generadas en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
