"""Contrato oficial de Fase 2.3: cargar y aplicar Spearman sin selecciÃ³n ni ajuste."""
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
    # ExtensiÃ³n manual posterior: conservar intacta la evidencia automÃ¡tica de 14 variables.
    reglas["variables"] = [*columnas, "es_fin_de_anio"]
    reglas["cantidad_variables"] = 15
    reglas["columnas_efectivas"] += 1
    reglas["incorporacion_manual"] = {"variable": "es_fin_de_anio", "adr": "0015",
                                    "evaluada": False}
    reglas["mae_validation_original_14"] = reglas.pop("mae_validation")
    return reglas


def aplicar_conjunto_definitivo(X, reglas=None):
    """Extrae nombres y orden oficiales de una tabla con ingenierÃ­a reconstruida."""
    reglas = cargar_seleccion_definitiva() if reglas is None else reglas
    if not X.columns.is_unique:
        raise ValueError("Las columnas de entrada deben ser únicas")
    faltan = set(reglas["variables"]) - set(X.columns)
    if faltan:
        raise ValueError(f"Faltan variables definitivas: {sorted(faltan)}")
    return X.loc[:, reglas["variables"]].copy()


def reconstruir_conjunto_definitivo(historia, reglas=None):
    """Reutiliza limpieza e ingenierÃ­a congeladas sin aprender ni construir etiquetas.

    Recibe bloques crudos ordenados hasta la emisiÃ³n, con sus controles. Conservar
    historia antes de quitar filas. Aplica la polÃ­tica estricta de lecturas mÃ­nimas.
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

    Conserva las 27 columnas originales y agrega el indicador binario (28); no entrena un modelo definitivo.
    """
    from experimentos_seleccion import aplicar_preparacion
    reglas = cargar_seleccion_definitiva() if reglas is None else reglas
    seleccion = aplicar_conjunto_definitivo(X, reglas)
    preparada = aplicar_preparacion(seleccion, reglas["preprocesamiento"])
    # El indicador binario no necesita aprender mediana ni escala.
    return np.column_stack([preparada, seleccion["es_fin_de_anio"].to_numpy(dtype=float)])
