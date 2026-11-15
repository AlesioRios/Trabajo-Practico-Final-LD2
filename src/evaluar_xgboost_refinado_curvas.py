"""Una evaluación del ganador refinado con curvas por árbol, sin early stopping."""
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from modelado_comun import RAIZ, RUTA_LOG, _estado_de_git, cargar_dataset_final, filas_comunes, registrar_experimento

SALIDA = RAIZ / "results/xgboost_15_refinado_validacion"


def evaluar_con_curvas():
    """Ajusta una vez con 830 filas; consultas posteriores solo leen evidencia."""
    if (SALIDA / "evaluacion.json").exists():
        return json.loads((SALIDA / "evaluacion.json").read_text(encoding="utf-8"))
    commit, sucio = _estado_de_git()
    if sucio is not False or commit == "desconocido":
        raise RuntimeError("Versionar el código antes del único entrenamiento adicional.")
    carpeta = RAIZ / "results/xgboost_15_refinado"
    resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
    detalle = json.loads((carpeta / "detalle.json").read_text(encoding="utf-8"))
    contrato = detalle["configuracion"]["configuracion"]
    parametros = {**contrato["fijos"], **resumen["parametros_provisionales"]}
    if resumen["total"] != 60 or "early_stopping_rounds" in parametros:
        raise ValueError("Se requiere el estudio terminado, sin early stopping.")
    datos = cargar_dataset_final()
    train, val = datos["entrenamiento"], datos["validacion"]
    comunes = filas_comunes(val)
    if len(train["X"]) != 830 or int(comunes.sum()) != 171 or list(train["X"].columns) != contrato["predictoras"]:
        raise ValueError("No coincide el contrato de datos de la evaluación.")
    if hashlib.sha256((RAIZ / "data/processed/dataset_final.csv").read_bytes()).hexdigest() != contrato["dataset_sha256"]:
        raise ValueError("El dataset cambió desde la segunda búsqueda.")
    previos = RUTA_LOG.read_bytes()
    modelo = XGBRegressor(**parametros)
    modelo.fit(train["X"], train["y"],
               eval_set=[(train["X"], train["y"]), (val["X"].loc[comunes], val["y"].loc[comunes])], verbose=False)
    historia = modelo.evals_result()
    curvas = pd.DataFrame({"arboles": np.arange(1, parametros["n_estimators"] + 1),
                           "mae_entrenamiento": historia["validation_0"]["mae"],
                           "mae_validacion_independiente": historia["validation_1"]["mae"]})
    pred = modelo.predict(val["X"].loc[comunes])
    registro = registrar_experimento("xgboost", parametros, val["y"].loc[comunes], pred, "validacion",
        notas=f"15 features; estudio refinado {resumen['estudio']}; trial {resumen['mejor_trial_provisional']}; ajuste 830 filas; eval_set train y validacion independiente 171; sin early stopping ni reajuste")
    if not RUTA_LOG.read_bytes().startswith(previos):
        raise RuntimeError("Se debe conservar el registro anterior.")
    SALIDA.mkdir(parents=True, exist_ok=False)
    curvas.to_csv(SALIDA / "mae_por_arbol.csv", index=False)
    pd.DataFrame({"indice_csv": val["X"].index[comunes], "DIA": val["fechas"].loc[comunes],
                  "real": val["y"].loc[comunes], "predicho": pred}).to_csv(SALIDA / "predicciones.csv", index=False)
    figura, eje = plt.subplots(figsize=(12, 6), layout="constrained")
    eje.plot(curvas["arboles"], curvas["mae_entrenamiento"], label="Entrenamiento (830 filas)", color="#1b6ca8")
    eje.plot(curvas["arboles"], curvas["mae_validacion_independiente"], label="Validación independiente (171 filas comunes)", color="#c45a17")
    eje.set(xlabel="Cantidad de árboles", ylabel="MAE (kW)",
            title="XGBoost con 15 features — MAE por iteración de boosting", xlim=(1, parametros["n_estimators"]))
    eje.grid(alpha=.25)
    eje.legend()
    figura.savefig(SALIDA / "mae_entrenamiento_validacion_por_arbol.png", dpi=220, facecolor="white")
    plt.close(figura)
    resultado = {"estudio": resumen["estudio"], "trial": resumen["mejor_trial_provisional"],
                 "configuracion_completa": parametros, "filas_ajuste": 830, "filas_validacion": 171,
                 "early_stopping": False, "arboles_entrenados": modelo.get_booster().num_boosted_rounds(),
                 "registro": registro, "dataset_sha256": contrato["dataset_sha256"],
                 "png": "results/xgboost_15_refinado_validacion/mae_entrenamiento_validacion_por_arbol.png",
                 "alcance": "Configuración fijada antes de validación; curvas descriptivas, sin cambio de hiperparámetros; sin prueba"}
    (SALIDA / "evaluacion.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado
