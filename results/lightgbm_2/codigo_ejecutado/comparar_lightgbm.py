"""Comparación descriptiva de ganadores; nunca propone ni ajusta hiperparámetros."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelado_comun import RAIZ, calcular_metricas, cargar_dataset_final, filas_comunes


def comparar_busquedas():
    """Reutiliza modelos guardados y verifica las mismas filas y predicciones de validación."""
    salida = RAIZ / "results/lightgbm_2"
    datos = cargar_dataset_final()
    train, val = datos["entrenamiento"], datos["validacion"]
    comunes_train, comunes_val = filas_comunes(train), filas_comunes(val)
    filas, parametros, limites = [], [], []
    for numero, nombre in [(1, "lightgbm"), (2, "lightgbm_2")]:
        carpeta = RAIZ / "results" / nombre
        evaluacion = json.loads((carpeta / "validacion.json").read_text(encoding="utf-8"))
        resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
        contrato = evaluacion["configuracion"]
        for clave, archivo in [("dataset_sha256", "dataset_final.csv"), ("linaje_sha256", "data_lineage.json")]:
            if contrato[clave] != hashlib.sha256((RAIZ / "data/processed" / archivo).read_bytes()).hexdigest():
                raise ValueError("Los datos cambiaron desde la búsqueda.")
        if contrato["predictoras"] != list(train["X"].columns):
            raise ValueError("Las predictoras no coinciden.")
        modelo = joblib.load(RAIZ / evaluacion["modelo"])
        guardadas = pd.read_csv(carpeta / "predicciones_validacion.csv")
        np.testing.assert_array_equal(guardadas["indice_csv"], val["X"].index[comunes_val])
        pred_val = modelo.predict(val["X"].loc[comunes_val])
        np.testing.assert_allclose(pred_val, guardadas["predicho"], rtol=1e-14)
        metricas_val = calcular_metricas(val["y"].loc[comunes_val], pred_val)
        for clave, valor in metricas_val.items():
            np.testing.assert_allclose(valor, evaluacion["metricas"][clave], rtol=1e-12)
        metricas_train = calcular_metricas(train["y"].loc[comunes_train],
                                           modelo.predict(train["X"].loc[comunes_train]))
        filas.append({"busqueda": numero, "trial": evaluacion["trial"],
                      "mae_optuna": evaluacion["mae_cv"],
                      **{f"{k}_validacion": v for k, v in metricas_val.items()},
                      **{f"{k}_entrenamiento": v for k, v in metricas_train.items()},
                      "brecha_mae_val_train": metricas_val["mae"] - metricas_train["mae"],
                      "filas_ajuste": evaluacion["filas_ajuste"],
                      "filas_metricas_entrenamiento": int(comunes_train.sum()),
                      "filas_validacion": int(comunes_val.sum()), "estados": resumen["estados"]})
        for nombre_param, valor in evaluacion["parametros"].items():
            parametros.append({"busqueda": numero, "hiperparametro": nombre_param, "valor": valor})
            if numero != 2 or nombre_param not in contrato["espacio"]:
                continue
            rango = contrato["espacio"][nombre_param]
            low, high = rango["low"], rango["high"]
            if nombre_param == "num_leaves":
                high = min(31, 2 ** evaluacion["parametros"]["max_depth"])
            transformar = np.log10 if rango.get("log", False) else float
            # Con profundidad 3, num_leaves solo puede ser 8: no es un límite explorado.
            posicion = None if high == low else (transformar(valor) - transformar(low)) / (transformar(high) - transformar(low))
            limites.append({"hiperparametro": nombre_param, "valor": valor, "minimo": low,
                            "maximo": high, "posicion_relativa": None if posicion is None else float(posicion),
                            "cerca_limite": "fijo por condición" if posicion is None else "inferior" if posicion <= .05 else "superior" if posicion >= .95 else "no"})
    tabla = pd.DataFrame(filas)
    primero, segundo = filas
    diferencias = {}
    for metrica in ["mae_optuna", "mae_validacion", "rmse_validacion", "r2_validacion"]:
        delta = segundo[metrica] - primero[metrica]
        diferencias[metrica] = {"segunda_menos_primera": delta,
                               "variacion_porcentual": 100 * delta / primero[metrica]}
    resultado = {"busquedas": filas, "diferencias": diferencias,
                 "parametros": parametros, "limites_segunda_busqueda": limites,
                 "criterio_cerca_limite": "5% del ancho; escala log10 para regularización",
                 "limitacion": "Comparación descriptiva: no demuestra significación estadística. La validación ya se usó para comparar modelos y seleccionar variables; no reemplaza test."}
    (salida / "comparacion.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    tabla.drop(columns="estados").to_csv(salida / "comparacion.csv", index=False)
    pd.DataFrame(parametros).pivot(index="hiperparametro", columns="busqueda", values="valor").to_csv(salida / "parametros_comparados.csv")
    pd.DataFrame(limites).to_csv(salida / "limites.csv", index=False)
    return resultado


if __name__ == "__main__":
    print(json.dumps(comparar_busquedas(), ensure_ascii=False, indent=2))
