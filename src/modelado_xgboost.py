"""BÃºsqueda temporal de XGBoost con Optuna; no entrena un modelo definitivo."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from auxiliar_functions import SEMILLA
from modelado_comun import (RAIZ, RUTA_DATASET_FINAL, RUTA_LINAJE, RUTA_LOG, _estado_de_git,
                            calcular_metricas, cargar_dataset_final, filas_comunes,
                            registrar_experimento, validacion_cruzada_temporal)

NOMBRE_ESTUDIO = "xgboost-temporal-v1"
RUTA_ESTUDIO = RAIZ / "results" / "optuna" / "xgboost" / "estudio.sqlite3"
ESPACIO = {
    "n_estimators": {"tipo": "int", "low": 100, "high": 2000},
    "learning_rate": {"tipo": "float", "low": 0.001, "high": 0.3, "log": True},
    "max_depth": {"tipo": "int", "low": 3, "high": 10},
    "min_child_weight": {"tipo": "float", "low": 0.1, "high": 10.0, "log": True},
    "subsample": {"tipo": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"tipo": "float", "low": 0.5, "high": 1.0},
}
PARAMS_FIJOS = {"reg_alpha": 0, "reg_lambda": 1, "gamma": 0, "objective": "reg:squarederror",
                "eval_metric": "mae", "tree_method": "hist", "random_state": SEMILLA, "n_jobs": 1}
CONFIG_PRUNER = {"n_startup_trials": 5, "n_warmup_steps": 2, "interval_steps": 1, "n_min_trials": 3}


def proponer_parametros(trial):
    """Propone exactamente los seis parÃ¡metros acordados, en orden estable."""
    parametros = {}
    for nombre, rango in ESPACIO.items():
        opciones = {k: v for k, v in rango.items() if k != "tipo"}
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
    fabrica_modelo = fabrica_modelo or XGBRegressor

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
            "sampler": "TPESampler(seed=42)", "pliegues": 5, "gap": 1,
            "dataset_sha256": huella(RUTA_DATASET_FINAL), "linaje_sha256": huella(RUTA_LINAJE),
            "codigo_sha256": huella(__file__),
            "comunes_sha256": {nombre: huella(RAIZ / "src" / nombre)
                               for nombre in ("modelado_comun.py", "auxiliar_functions.py")},
            "requirements_sha256": huella(RAIZ / "requirements.txt"), "python": platform.python_version(),
            "versiones": {p: importlib.metadata.version(p) for p in ("optuna", "xgboost", "numpy", "pandas", "scikit-learn")}}


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
        registrar_experimento("xgboost", {**PARAMS_FIJOS, **trial.params}, pred["real"], pred["predicho"],
                              "entrenamiento", notas=marca, ruta=ruta_log)
        previas.add(marca)


def ejecutar_busqueda(estudio, objective, total=100, ruta_log=RUTA_LOG):
    """ContinÃºa hasta el lÃ­mite TOTAL; un Ãºnico proceso, un trial a la vez."""
    if not 0 <= total <= 100:
        raise ValueError("El mÃ¡ximo total debe estar entre 0 y 100.")
    sincronizar_registro(estudio, ruta_log)
    restantes = max(0, total - len(estudio.trials))
    estudio.optimize(objective, n_trials=restantes, n_jobs=1,
                      callbacks=[lambda s, t: sincronizar_registro(s, ruta_log)], catch=(ValueError, RuntimeError))
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
    carpeta = RAIZ / "results" / "xgboost" if carpeta is None else Path(carpeta)
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


def main():
    """Entrada explÃ­cita de bÃºsqueda; la notebook solo consulta por defecto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=100, help="MÃ¡ximo TOTAL, incluidos trials previos (hasta 100)")
    parser.add_argument("--guardar-evidencia", action="store_true", help="Exporta copia versionable al terminar")
    args = parser.parse_args()
    commit, sucio = _estado_de_git()
    if sucio is not False or commit == "desconocido":
        raise RuntimeError("VersionÃ¡ el cÃ³digo y dejÃ¡ limpio el Ã¡rbol antes de ejecutar experimentos definitivos.")
    estudio = abrir_estudio()
    estudio.set_user_attr("commit_codigo", estudio.user_attrs.get("commit_codigo", commit))
    ejecutar_busqueda(estudio, crear_objective(cargar_entrenamiento()), total=args.total)
    exportar_resultados(estudio, RUTA_ESTUDIO.parent)
    if args.guardar_evidencia:
        guardar_evidencia(estudio)
    print(estudio.trials_dataframe(attrs=("number", "value", "state")))


if __name__ == "__main__":
    main()
