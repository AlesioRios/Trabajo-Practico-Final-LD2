"""Contrato oficial de Fase 2.3: cargar y aplicar Spearman sin selección ni ajuste."""
from pathlib import Path
import hashlib
import json
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
ARTEFACTO = RAIZ / "results/seleccion/comparacion_validation_congelada/seleccion_definitiva.json"


def cargar_seleccion_definitiva(ruta=ARTEFACTO):
    """Valida integridad, orden y contrato antes de devolver reglas JSON independientes."""
    ruta = Path(ruta)
    huellas = json.loads((ruta.parent / "sha256.json").read_text(encoding="utf-8"))
    if hashlib.sha256(ruta.read_bytes()).hexdigest() != huellas[ruta.name]:
        raise ValueError("Huella incompatible del conjunto definitivo")
    reglas = json.loads(ruta.read_text(encoding="utf-8"))
    columnas = reglas["variables"]
    prep = reglas["preprocesamiento"]
    if (reglas["metodo"] != "spearman" or len(columnas) != 14
            or len(set(columnas)) != 14 or prep["columnas"] != columnas
            or reglas["cantidad_variables"] != 14
            or reglas["columnas_efectivas"] != len(columnas) + len(prep["indicadores"])
            or not set(prep["indicadores"]).issubset(columnas)
            or reglas["test_utilizado"]):
        raise ValueError("Contrato Spearman inconsistente")
    return reglas


def aplicar_conjunto_definitivo(X, reglas=None):
    """Extrae nombres y orden oficiales de una tabla con ingeniería reconstruida."""
    reglas = cargar_seleccion_definitiva() if reglas is None else reglas
    if not X.columns.is_unique:
        raise ValueError("Las columnas de entrada deben ser únicas")
    faltan = set(reglas["variables"]) - set(X.columns)
    if faltan:
        raise ValueError(f"Faltan variables definitivas: {sorted(faltan)}")
    return X.loc[:, reglas["variables"]].copy()


def reconstruir_conjunto_definitivo(historia, reglas=None):
    """Reutiliza limpieza e ingeniería congeladas sin aprender ni construir etiquetas.

    Recibe bloques crudos ordenados hasta la emisión, con sus controles. Conservar
    historia antes de quitar filas. Aplica la política estricta de lecturas mínimas.
    """
    from limpieza import limpiar
    from preprocessing_pipeline import transform_variables
    reglas = cargar_seleccion_definitiva() if reglas is None else reglas
    tabla = historia.copy()
    incompletos = tabla["filas"].lt(24)
    tabla.loc[incompletos, "valido"] = False
    tabla.loc[incompletos, "consumo_dia"] = np.nan
    creada = transform_variables(limpiar(tabla, reglas["limpieza"]), reglas["ingenieria"])
    return aplicar_conjunto_definitivo(creada, reglas)


def preprocesar_conjunto_definitivo(X, reglas=None):
    """Aplica medianas, escala e indicadores guardados de Train, sin fit.

    Reproduce las 27 columnas del comparador; no entrena un modelo definitivo.
    """
    from experimentos_seleccion import aplicar_preparacion
    reglas = cargar_seleccion_definitiva() if reglas is None else reglas
    return aplicar_preparacion(aplicar_conjunto_definitivo(X, reglas), reglas["preprocesamiento"])
