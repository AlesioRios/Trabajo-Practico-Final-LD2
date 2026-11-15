"""BÃºsqueda temporal de LightGBM con Optuna; no entrena un modelo definitivo."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import zipfile
import joblib
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor

from auxiliar_functions import SEMILLA
from modelado_comun import (RAIZ, RUTA_DATASET_FINAL, RUTA_LINAJE, RUTA_LOG, _estado_de_git,
                            calcular_metricas, cargar_dataset_final, filas_comunes,
                            registrar_experimento, validacion_cruzada_temporal)

NOMBRE_ESTUDIO = "lightgbm-temporal-v1"
RUTA_ESTUDIO = RAIZ / "results" / "optuna" / "lightgbm" / "estudio.sqlite3"
ESPACIO = {
    "n_estimators": {"tipo": "int", "low": 100, "high": 500},
    "learning_rate": {"tipo": "float", "low": 0.02, "high": 0.10},
    "max_depth": {"tipo": "int", "low": 3, "high": 6},
    "num_leaves": {"tipo": "int", "low": 8, "high": 31},
    "min_child_samples": {"tipo": "int", "low": 20, "high": 60},
    "subsample": {"tipo": "float", "low": 0.75, "high": 1.0},
    "colsample_bytree": {"tipo": "float", "low": 0.75, "high": 1.0},
    "reg_alpha": {"tipo": "float", "low": 0.001, "high": 10.0, "log": True},
    "reg_lambda": {"tipo": "float", "low": 0.001, "high": 10.0, "log": True},
}
PARAMS_FIJOS = {"subsample_freq": 1, "random_state": SEMILLA, "verbosity": -1, "n_jobs": 1}
CONFIG_PRUNER = {"n_startup_trials": 5, "n_warmup_steps": 2, "interval_steps": 1, "n_min_trials": 3}


def proponer_parametros(trial):
    """Propone exactamente los seis parÃ¡metros acordados, en orden estable."""
    parametros = {}
    for nombre, rango in ESPACIO.items():
        opciones = {k: v for k, v in rango.items() if k != "tipo"}
        if nombre == "num_leaves":
            opciones["high"] = min(31, 2 ** parametros["max_depth"])
        sugerir = trial.suggest_int if rango["tipo"] == "int" else trial.suggest_float
        parametros[nombre] = sugerir(nombre, **opciones)
    return parametros


def cargar_entrenamiento():
    """Selecciona solo entrenamiento del cargador comÃºn; no consulta validaciÃ³n ni prueba."""
    return cargar_dataset_final()["entrenamiento"]


def crear_objective(entrenamiento, fabrica_modelo=None):
    """Cierra sobre entrenamiento Ãºnicamente; permite modelos pequeÃ±os de prueba."""
    X, y = entrenamiento["X"], entrenamiento["y"]
    prohibidas = {"objetivo", "consumo_dia", "baseline_media_movil", "tramo", "DIA"}
    if X.shape[1] != 15 or prohibidas.intersection(X.columns):
        raise ValueError("Se requieren las 15 variables seleccionadas sin columnas auxiliares.")
    if not X.index.is_unique or not X.index.is_monotonic_increasing:
        raise ValueError("Se requieren los Ã­ndices Ãºnicos y ordenados del CSV.")
    if not entrenamiento["fechas"].is_monotonic_increasing or y.isna().any():
        raise ValueError("El entrenamiento debe estar ordenado y tener objetivo completo.")
    if not all(v.index.equals(X.index) for v in entrenamiento.values()):
        raise ValueError("Los Ã­ndices del entrenamiento deben coincidir.")
    fabrica_modelo = fabrica_modelo or LGBMRegressor

    def objective(trial):
        """MAE conjunto de las predicciones fuera de muestra de cinco pliegues."""
        reales, predichos, indices, pliegues = [], [], [], []
        try:
            parametros = {**PARAMS_FIJOS, **proponer_parametros(trial)}
            for paso, (ajuste, evaluacion) in enumerate(validacion_cruzada_temporal().split(X)):
                parte = {k: v.iloc[evaluacion] for k, v in entrenamiento.items()}
                comunes = filas_comunes(parte)
                if comunes.sum() < 2:
                    raise ValueError("El pliegue necesita al menos dos filas comunes para RÂ².")
                modelo = fabrica_modelo(**parametros)
                modelo.fit(X.iloc[ajuste], y.iloc[ajuste])
                real = parte["y"].loc[comunes].to_numpy()
                pred = modelo.predict(parte["X"].loc[comunes])
                reales.extend(real.tolist())
                predichos.extend(np.asarray(pred).tolist())
                indices.extend(parte["X"].index[comunes].tolist())
                acumuladas = calcular_metricas(reales, predichos)
                pliegues.append({"paso": paso, "filas_ajuste": len(ajuste), "filas_comunes": len(real),
                                 "ultimo_indice_ajuste": int(X.index[ajuste[-1]]),
                                 "primer_indice_evaluacion": int(parte["X"].index[0]),
                                 **calcular_metricas(real, pred), "mae_acumulado": acumuladas["mae"]})
                trial.set_user_attr("pliegues", pliegues)
                trial.report(acumuladas["mae"], paso)
                if trial.should_prune():
                    motivo = f"MedianPruner al completar el pliegue {paso + 1}"
                    trial.set_user_attr("motivo", motivo)
                    raise optuna.TrialPruned(motivo)
            trial.set_user_attr("metricas", acumuladas)
            # Conservar las predicciones permite recuperar el log tras una interrupciÃ³n.
            trial.set_user_attr("predicciones", {"indices": indices, "real": reales, "predicho": predichos})
            return acumuladas["mae"]
        except optuna.TrialPruned:
            raise
        except Exception as error:
            trial.set_user_attr("motivo", f"{type(error).__name__}: {error}")
            raise

    return objective


def configuracion():
    """Contrato que debe coincidir al reanudar, incluyendo datos, cÃ³digo y versiones."""
    huella = lambda ruta: hashlib.sha256(Path(ruta).read_bytes()).hexdigest()
    return {"version": 1, "direccion": "minimize", "metrica": "mae", "max_trials": 100,
            "espacio": ESPACIO, "fijos": PARAMS_FIJOS, "pruner": CONFIG_PRUNER,
            "sampler": "TPESampler(seed=42)", "num_leaves_max": "min(31, 2 ** max_depth)",
            "referencia_optuna": "XGBoost primera busqueda, b964060",
            "predictoras": list(cargar_entrenamiento()["X"].columns), "pliegues": 5, "gap": 1,
            "dataset_sha256": huella(RUTA_DATASET_FINAL), "linaje_sha256": huella(RUTA_LINAJE),
            "codigo_sha256": huella(__file__),
            "comunes_sha256": {nombre: huella(RAIZ / "src" / nombre)
                               for nombre in ("modelado_comun.py", "auxiliar_functions.py")},
            "requirements_sha256": huella(RAIZ / "requirements.txt"), "python": platform.python_version(),
            "versiones": {p: importlib.metadata.version(p) for p in ("optuna", "lightgbm", "numpy", "pandas", "scikit-learn")}}


def abrir_estudio(ruta=RUTA_ESTUDIO, contrato=None):
    """Abre o crea SQLite y rechaza cambios incompatibles o trials sin terminar."""
    ruta = Path(ruta).resolve()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contrato = configuracion() if contrato is None else contrato
    estudio = optuna.create_study(study_name=NOMBRE_ESTUDIO, storage=f"sqlite:///{ruta.as_posix()}",
                                  direction="minimize", load_if_exists=True,
                                  sampler=optuna.samplers.TPESampler(seed=SEMILLA),
                                  pruner=optuna.pruners.MedianPruner(**CONFIG_PRUNER))
    anterior = estudio.user_attrs.get("configuracion")
    if anterior is None and not estudio.trials:
        estudio.set_user_attr("configuracion", contrato)
    elif anterior != contrato or estudio.direction != optuna.study.StudyDirection.MINIMIZE:
        raise ValueError("La configuraciÃ³n del estudio persistido es incompatible; no se continÃºa.")
    if any(not t.state.is_finished() for t in estudio.trials):
        raise ValueError("Hay trials RUNNING/WAITING: revisar la interrupciÃ³n antes de continuar.")
    # SQLite conserva resultados pero no el generador aleatorio. Reproducimos las propuestas
    # en memoria para recuperar el estado de TPE sin pickle ni volver a entrenar modelos.
    if estudio.trials:
        replay = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMILLA))
        for anterior in estudio.trials:
            actual = replay.ask()
            if proponer_parametros(actual) != anterior.params:
                raise ValueError("No se pudo reproducir la secuencia del sampler; no se continÃºa.")
            for paso, valor in anterior.intermediate_values.items():
                actual.report(valor, paso)
            replay.tell(actual, anterior.value if anterior.state == optuna.trial.TrialState.COMPLETE else None,
                        state=anterior.state)
        # Optuna 5.0 liga esta cachÃ© al identificador del estudio en memoria.
        # Se descarta solo la cachÃ© reconstruible; se conserva el estado aleatorio.
        # Esta integraciÃ³n interna queda cubierta por la prueba de reanudaciÃ³n.
        replay.sampler._search_space = optuna.search_space.IntersectionSearchSpace(include_pruned=True)
        estudio.sampler = replay.sampler
    return estudio


def sincronizar_registro(estudio, ruta_log=RUTA_LOG):
    """Registra solo COMPLETE; recupera faltantes sin duplicar filas tras reanudar."""
    previas = set(pd.read_csv(ruta_log)["notas"].dropna()) if Path(ruta_log).exists() else set()
    for trial in estudio.trials:
        marca = f"Optuna {estudio.study_name}; trial={trial.number}; CV temporal 5 pliegues; solo entrenamiento"
        if trial.state != optuna.trial.TrialState.COMPLETE or marca in previas:
            continue
        pred = trial.user_attrs["predicciones"]
        registrar_experimento("lightgbm", {**PARAMS_FIJOS, **trial.params}, pred["real"], pred["predicho"],
                              "entrenamiento", notas=marca, ruta=ruta_log)
        previas.add(marca)


def ejecutar_busqueda(estudio, objective, total=100, ruta_log=RUTA_LOG):
    """ContinÃºa hasta el lÃ­mite TOTAL; un Ãºnico proceso, un trial a la vez."""
    if not 0 <= total <= 100:
        raise ValueError("El mÃ¡ximo total debe estar entre 0 y 100.")
    sincronizar_registro(estudio, ruta_log)
    restantes = max(0, total - len(estudio.trials))
    estudio.optimize(objective, n_trials=restantes, n_jobs=1,
                      callbacks=[lambda s, t: (sincronizar_registro(s, ruta_log), mostrar_progreso(s, t, total, ruta_log))], catch=(ValueError, RuntimeError))
    return estudio


def exportar_resultados(estudio, carpeta):
    """Exporta estados, parÃ¡metros y detalle por pliegue sin mÃ©tricas finales inventadas."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = [{"trial": t.number, "estado": t.state.name, **t.params,
              **t.user_attrs.get("metricas", {}), "pliegues_completados": len(t.user_attrs.get("pliegues", [])),
              "motivo": t.user_attrs.get("motivo", "")} for t in estudio.trials]
    pd.DataFrame(filas).to_csv(carpeta / "trials.csv", index=False, lineterminator="\n")
    detalle = [{"trial": t.number, "estado": t.state.name, "parametros": t.params,
                "intermedios": t.intermediate_values, **t.user_attrs} for t in estudio.trials]
    (carpeta / "detalle.json").write_text(json.dumps({"configuracion": estudio.user_attrs,
                                                     "trials": detalle}, ensure_ascii=False, indent=2), encoding="utf-8")


def guardar_evidencia(estudio, carpeta=None):
    """Copia consistente de SQLite y resultados para versionar al terminar la búsqueda."""
    carpeta = RAIZ / "results" / "lightgbm" if carpeta is None else Path(carpeta)
    exportar_resultados(estudio, carpeta)
    copia = RUTA_ESTUDIO.parent / "copia.sqlite3"
    with sqlite3.connect(RUTA_ESTUDIO) as origen, sqlite3.connect(copia) as destino:
        origen.backup(destino)
    with zipfile.ZipFile(carpeta / "estudio.sqlite3.zip", "w", compression=zipfile.ZIP_DEFLATED) as archivo:
        archivo.write(copia, "estudio.sqlite3")
    mejor = estudio.best_trial if any(t.state.name == "COMPLETE" for t in estudio.trials) else None
    resumen = {"estudio": estudio.study_name, "total": len(estudio.trials),
               "estados": dict(Counter(t.state.name for t in estudio.trials)),
               "commit_codigo": estudio.user_attrs.get("commit_codigo"),
               "mejor_trial_provisional": mejor.number if mejor else None,
               "parametros_provisionales": mejor.params if mejor else None,
               "metricas_provisionales": mejor.user_attrs["metricas"] if mejor else None,
               "sqlite_sha256": hashlib.sha256(copia.read_bytes()).hexdigest()}
    (carpeta / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")


def mostrar_progreso(estudio, trial, total, ruta_log=RUTA_LOG):
    """Informa cada estado; el MAE de un podado es parcial, nunca final."""
    completos = [t for t in estudio.trials if t.state.name == "COMPLETE"]
    mejor = estudio.best_trial if completos else None
    mensaje = {"trial": trial.number, "avance": f"{len(estudio.trials)}/{total}",
               "estado": trial.state.name,
               "mae": trial.value if trial.state.name == "COMPLETE" else None,
               "mae_parcial": trial.value if trial.state.name == "PRUNED" else None,
               "mejor_mae": mejor.value if mejor else None,
               "mejor_trial": mejor.number if mejor else None}
    if mejor and mejor.number == trial.number:
        mensaje["nuevos_mejores_parametros"] = mejor.params
    print(json.dumps(mensaje, ensure_ascii=False), flush=True)
    ruta = Path(ruta_log).parent / "lightgbm" / "progreso.jsonl"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(mensaje, ensure_ascii=False) + "\n")


def evaluar_ganador(estudio, carpeta):
    """Ajusta solo entrenamiento y evalúa una vez en validación independiente."""
    carpeta = Path(carpeta)
    destino = carpeta / "validacion.json"
    if destino.exists():
        anterior = json.loads(destino.read_text(encoding="utf-8"))
        if anterior["trial"] != estudio.best_trial.number or anterior["configuracion"] != estudio.user_attrs["configuracion"]:
            raise ValueError("La evaluación guardada corresponde a otro ganador o contrato.")
        return anterior
    datos = cargar_dataset_final()  # incluir_prueba=False: nunca se entrega test
    train, val = datos["entrenamiento"], datos["validacion"]
    comunes = filas_comunes(val)
    if train["fechas"].max() >= val["fechas"].min():
        raise ValueError("Entrenamiento y validación deben estar separados temporalmente.")
    parametros = {**PARAMS_FIJOS, **estudio.best_params}
    modelo = LGBMRegressor(**parametros)
    modelo.fit(train["X"], train["y"])
    pred = modelo.predict(val["X"].loc[comunes])
    marca = f"Optuna {estudio.study_name}; ganador trial={estudio.best_trial.number}; validacion independiente; solo ajuste entrenamiento"
    registro = registrar_experimento("lightgbm", parametros, val["y"].loc[comunes], pred,
                                    "validacion", notas=marca)
    modelos = RAIZ / "models" / "lightgbm"
    modelos.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, modelos / "modelo_validacion.joblib")
    modelo.booster_.save_model(str(modelos / "modelo_validacion.txt"))
    pd.DataFrame({"indice_csv": val["X"].index[comunes], "DIA": val["fechas"].loc[comunes],
                  "real": val["y"].loc[comunes], "predicho": pred}).to_csv(carpeta / "predicciones_validacion.csv", index=False)
    resultado = {"trial": estudio.best_trial.number, "mae_cv": estudio.best_value,
                 "parametros": parametros, "metricas": {k: registro[k] for k in ("mae", "rmse", "r2")},
                 "filas_ajuste": len(train["X"]), "filas_validacion": int(comunes.sum()),
                 "registro": registro, "configuracion": estudio.user_attrs["configuracion"],
                 "modelo": "models/lightgbm/modelo_validacion.joblib",
                 "alcance": "Modelo para validación; sin reentrenar con validación y sin evaluar test"}
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def generar_graficos(estudio, carpeta):
    """Mismos gráficos matplotlib de XGBoost, con f-ANOVA reproducible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances
    from optuna.visualization.matplotlib import (plot_optimization_history,
                                                plot_param_importances, plot_parallel_coordinate)
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    huella = hashlib.sha256(RUTA_ESTUDIO.read_bytes()).hexdigest()
    plt.rcParams.update({"font.size": 11, "figure.dpi": 120, "savefig.dpi": 220})
    importancia = get_param_importances(estudio, evaluator=FanovaImportanceEvaluator(seed=42))
    (carpeta / "importancias.json").write_text(json.dumps(importancia, indent=2), encoding="utf-8")
    graficos = [
        (lambda: plot_optimization_history(estudio), "historial_optimizacion.png", "Historial de optimización", (12, 6)),
        (lambda: plot_param_importances(estudio, evaluator=FanovaImportanceEvaluator(seed=42)),
         "importancia_hiperparametros_fanova.png", "Importancia de hiperparámetros · f-ANOVA", (12, 6)),
        (lambda: plot_parallel_coordinate(estudio), "coordenadas_paralelas.png", "Coordenadas paralelas", (19, 7)),
    ]
    for generar, nombre, titulo, tamano in graficos:
        eje = generar()
        fig = eje.figure
        fig.set_size_inches(*tamano)
        eje.set_title(f"{titulo} · LightGBM, 15 variables", pad=25)
        fig.tight_layout()
        fig.savefig(carpeta / nombre, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    if hashlib.sha256(RUTA_ESTUDIO.read_bytes()).hexdigest() != huella:
        raise RuntimeError("La visualización modificó SQLite.")


def main():
    """Primera búsqueda completa, evidencia, validación del ganador y tres PNG."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = abrir_estudio()
    commit, sucio = _estado_de_git()
    estudio.set_user_attr("commit_codigo", estudio.user_attrs.get("commit_codigo", commit))
    estudio.set_user_attr("arbol_sin_commitear", sucio)
    salida = RAIZ / "results" / "lightgbm"
    ejecutar_busqueda(estudio, crear_objective(cargar_entrenamiento()))
    guardar_evidencia(estudio, salida)
    evaluacion = evaluar_ganador(estudio, salida)
    generar_graficos(estudio, salida / "graficos")
    print(json.dumps({"validacion": evaluacion["metricas"], "trial": estudio.best_trial.number}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

