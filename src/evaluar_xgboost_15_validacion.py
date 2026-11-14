"""Única evaluación independiente del ganador XGBoost de 15 variables, sin optimizar."""
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor
from modelado_comun import (RAIZ, RUTA_LOG, _estado_de_git, cargar_dataset_final,
                            filas_comunes, registrar_experimento)
from modelado_xgboost import PARAMS_FIJOS

SALIDA = RAIZ / "results" / "xgboost_15_validacion"
REFERENCIA = "origin/feature/modeling-rf-lineales-experimentos"


def evaluar_validacion():
    """Entrena una sola vez; si la evidencia existe, solo la lee."""
    if (SALIDA / "evaluacion.json").exists():
        return json.loads((SALIDA / "evaluacion.json").read_text(encoding="utf-8")), pd.read_csv(SALIDA / "comparacion.csv")
    commit, sucio = _estado_de_git()
    if sucio is not False or commit == "desconocido":
        raise RuntimeError("Versionar el código y dejar limpio el árbol antes de evaluar.")
    resumen = json.loads((RAIZ / "results/xgboost_15/resumen.json").read_text(encoding="utf-8"))
    detalle = json.loads((RAIZ / "results/xgboost_15/detalle.json").read_text(encoding="utf-8"))
    ganador = next(t for t in detalle["trials"] if t["trial"] == resumen["mejor_trial_provisional"])
    contrato = detalle["configuracion"]["configuracion"]
    if ganador["estado"] != "COMPLETE" or ganador["parametros"] != resumen["parametros_provisionales"]:
        raise ValueError("La configuración ganadora no coincide con el estudio conservado.")
    if PARAMS_FIJOS != contrato["fijos"]:
        raise ValueError("Los parámetros fijos cambiaron respecto de la búsqueda.")
    datos = cargar_dataset_final()  # no incluye prueba
    train, val = datos["entrenamiento"], datos["validacion"]
    comunes = filas_comunes(val)
    if len(train["X"]) != 830 or train["X"].shape[1] != 15 or int(comunes.sum()) != 171:
        raise ValueError("No coincide el contrato de 830 filas, 15 predictoras y 171 filas comunes.")
    if list(train["X"].columns) != contrato["predictoras"]:
        raise ValueError("Las predictoras no coinciden con el estudio de 15 variables.")
    huella = hashlib.sha256((RAIZ / "data/processed/dataset_final.csv").read_bytes()).hexdigest()
    if huella != contrato["dataset_sha256"]:
        raise ValueError("El dataset cambió desde la búsqueda.")
    ref = subprocess.check_output(["git", "rev-parse", REFERENCIA], cwd=RAIZ, text=True).strip()
    remoto = subprocess.check_output(["git", "show", f"{ref}:results/experiment_logs.csv"], cwd=RAIZ).decode("utf-8-sig")
    registros = pd.read_csv(io.StringIO(remoto))
    comparables = registros.loc[registros["tramo"].eq("validacion") & registros["modelo"].isin(["ridge", "lasso", "random_forest"])
        & registros["notas"].str.startswith("15 variables con es_fin_de_anio; Optuna", na=False)].copy()
    if len(comparables) != 3 or not comparables["modelo"].is_unique or not comparables["filas"].eq(171).all():
        raise ValueError("No se identifican inequívocamente las evaluaciones comparables.")
    anteriores = RUTA_LOG.read_bytes()
    parametros = {**PARAMS_FIJOS, **ganador["parametros"]}
    modelo = XGBRegressor(**parametros)
    modelo.fit(train["X"], train["y"])
    pred = modelo.predict(val["X"].loc[comunes])
    marca = f"15 variables con es_fin_de_anio; validacion independiente; Optuna {resumen['estudio']}; trial {ganador['trial']}; ajuste 830 filas; sin nueva optimizacion"
    registro = registrar_experimento("xgboost", parametros, val["y"].loc[comunes], pred, "validacion", notas=marca)
    if not RUTA_LOG.read_bytes().startswith(anteriores):
        raise RuntimeError("Se debe preservar el log anterior.")
    SALIDA.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"indice_csv": val["X"].index[comunes], "DIA": val["fechas"].loc[comunes],
                  "real": val["y"].loc[comunes], "predicho": pred}).to_csv(SALIDA / "predicciones.csv", index=False)
    columnas = ["modelo", "filas", "mae", "rmse", "r2", "commit"]
    comparacion = pd.concat([comparables[columnas], pd.DataFrame([registro])[columnas]], ignore_index=True).sort_values("mae")
    comparacion.to_csv(SALIDA / "comparacion.csv", index=False)
    comparables.to_csv(SALIDA / "referencias_companera.csv", index=False)
    resultado = {"estudio": resumen["estudio"], "trial": ganador["trial"], "predictoras": list(train["X"].columns),
                 "filas_ajuste": 830, "filas_evaluacion": 171, "dataset_sha256": huella,
                 "referencia_companera": ref, "registro": registro,
                 "alcance": "Validación independiente; sin test, sin nueva optimización, sin modelo de producción"}
    (SALIDA / "evaluacion.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado, comparacion
