"""Búsqueda temporal de Gradient Boosting con Optuna; no entrena un modelo definitivo."""

import argparse
from datetime import datetime, timezone
from time import perf_counter
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
from sklearn.ensemble import GradientBoostingRegressor
from pipeline_sklearn import armar_pipeline

from auxiliar_functions import SEMILLA
from modelado_comun import (RAIZ, RUTA_DATASET_FINAL, RUTA_LINAJE, RUTA_LOG, _estado_de_git,
                            calcular_metricas, cargar_dataset_final, filas_comunes,
                            registrar_experimento, validacion_cruzada_temporal)

NOMBRE_ESTUDIO = "gradient_boosting-temporal-v1"
RUTA_ESTUDIO = RAIZ / "results" / "optuna" / "gradient_boosting" / "estudio.sqlite3"
ESPACIO = {
    "n_estimators": {"tipo": "int", "low": 100, "high": 600, "step": 50},
    "learning_rate": {"tipo": "float", "low": 0.01, "high": 0.15, "log": True},
    "max_depth": {"tipo": "int", "low": 2, "high": 4},
    "min_samples_leaf": {"tipo": "int", "low": 3, "high": 20},
    "subsample": {"tipo": "float", "low": 0.6, "high": 1.0, "step": 0.1},
    "max_features": {"tipo": "float", "low": 0.6, "high": 1.0, "step": 0.1},
    "loss": {"tipo": "categorical", "choices": ["squared_error", "absolute_error", "huber"]},
    "alpha": {"tipo": "float", "low": 0.75, "high": 0.95, "step": 0.05},
}
PARAMS_FIJOS = {"random_state": SEMILLA, "n_iter_no_change": None}
CONFIG_PRUNER = {"n_startup_trials": 5, "n_warmup_steps": 2, "interval_steps": 1, "n_min_trials": 3}


def proponer_parametros(trial):
    """Propone el espacio acordado, con alpha condicionado a huber."""
    parametros = {}
    for nombre, rango in ESPACIO.items():
        if nombre == "alpha" and parametros["loss"] != "huber":
            continue
        opciones = {k: v for k, v in rango.items() if k != "tipo"}
        sugerir = {"int": trial.suggest_int, "float": trial.suggest_float,
                   "categorical": trial.suggest_categorical}[rango["tipo"]]
        parametros[nombre] = sugerir(nombre, **opciones)
    return parametros


def crear_pipeline(columnas, **parametros):
    """Reutiliza A sin modificar la Fase 2 ni aprender fuera de fit."""
    return armar_pipeline("A", columnas=list(columnas)).set_params(
        regressor=GradientBoostingRegressor(**parametros))


def cargar_entrenamiento():
    """Selecciona solo entrenamiento del cargador común; no consulta validación ni prueba."""
    return cargar_dataset_final()["entrenamiento"]


def crear_objective(entrenamiento, fabrica_modelo=None):
    """Cierra sobre entrenamiento únicamente; permite modelos pequeÃ±os de prueba."""
    X, y = entrenamiento["X"], entrenamiento["y"]
    prohibidas = {"objetivo", "consumo_dia", "baseline_media_movil", "tramo", "DIA"}
    if X.shape[1] != 15 or prohibidas.intersection(X.columns):
        raise ValueError("Se requieren las 15 variables seleccionadas sin columnas auxiliares.")
    if not X.index.is_unique or not X.index.is_monotonic_increasing:
        raise ValueError("Se requieren los índices únicos y ordenados del CSV.")
    if not entrenamiento["fechas"].is_monotonic_increasing or y.isna().any():
        raise ValueError("El entrenamiento debe estar ordenado y tener objetivo completo.")
    if not all(v.index.equals(X.index) for v in entrenamiento.values()):
        raise ValueError("Los índices del entrenamiento deben coincidir.")
    fabrica_modelo = fabrica_modelo or (lambda **p: crear_pipeline(X.columns, **p))

    def objective(trial):
        """MAE conjunto de las predicciones fuera de muestra de cinco pliegues."""
        reales, predichos, indices, pliegues = [], [], [], []
        try:
            parametros = {**PARAMS_FIJOS, **proponer_parametros(trial)}
            for paso, (ajuste, evaluacion) in enumerate(validacion_cruzada_temporal().split(X)):
                parte = {k: v.iloc[evaluacion] for k, v in entrenamiento.items()}
                comunes = filas_comunes(parte)
                if comunes.sum() < 2:
                    raise ValueError("El pliegue necesita al menos dos filas comunes para R².")
                inicio = datetime.now(timezone.utc).isoformat()
                reloj = perf_counter()
                modelo = fabrica_modelo(**parametros)
                modelo.fit(X.iloc[ajuste], y.iloc[ajuste])
                real = parte["y"].loc[comunes].to_numpy()
                pred = modelo.predict(parte["X"].loc[comunes])
                reales.extend(real.tolist())
                predichos.extend(np.asarray(pred).tolist())
                indices.extend(parte["X"].index[comunes].tolist())
                acumuladas = calcular_metricas(reales, predichos)
                pliegues.append({"inicio_utc": inicio, "fin_utc": datetime.now(timezone.utc).isoformat(),
                                 "duracion_segundos": perf_counter() - reloj, "paso": paso, "filas_ajuste": len(ajuste), "filas_comunes": len(real),
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
            # Conservar las predicciones permite recuperar el log tras una interrupción.
            trial.set_user_attr("predicciones", {"indices": indices, "real": reales, "predicho": predichos})
            return acumuladas["mae"]
        except optuna.TrialPruned:
            raise
        except Exception as error:
            trial.set_user_attr("motivo", f"{type(error).__name__}: {error}")
            raise

    return objective


def configuracion():
    """Contrato que debe coincidir al reanudar, incluyendo datos, código y versiones."""
    huella = lambda ruta: hashlib.sha256(Path(ruta).read_bytes()).hexdigest()
    return {"version": 1, "direccion": "minimize", "metrica": "mae", "max_trials": 100,
            "espacio": ESPACIO, "fijos": PARAMS_FIJOS, "pruner": CONFIG_PRUNER,
            "sampler": "TPESampler(seed=42)", "pipeline": "armar_pipeline(A), mediana + StandardScaler + GradientBoostingRegressor",
            "referencia_optuna": "XGBoost primera busqueda, b964060",
            "predictoras": list(cargar_entrenamiento()["X"].columns), "pliegues": 5, "gap": 1,
            "dataset_sha256": huella(RUTA_DATASET_FINAL), "linaje_sha256": huella(RUTA_LINAJE),
            "codigo_sha256": huella(__file__),
            "comunes_sha256": {nombre: huella(RAIZ / "src" / nombre)
                               for nombre in ("modelado_comun.py", "auxiliar_functions.py", "pipeline_sklearn.py")},
            "requirements_sha256": huella(RAIZ / "requirements.txt"), "python": platform.python_version(),
            "versiones": {p: importlib.metadata.version(p) for p in ("optuna", "joblib", "numpy", "pandas", "scikit-learn")}}


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
        raise ValueError("La configuración del estudio persistido es incompatible; no se continúa.")
    if any(not t.state.is_finished() for t in estudio.trials):
        raise ValueError("Hay trials RUNNING/WAITING: revisar la interrupción antes de continuar.")
    # SQLite conserva resultados pero no el generador aleatorio. Reproducimos las propuestas
    # en memoria para recuperar el estado de TPE sin pickle ni volver a entrenar modelos.
    if estudio.trials:
        replay = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMILLA))
        for anterior in estudio.trials:
            actual = replay.ask()
            if proponer_parametros(actual) != anterior.params:
                raise ValueError("No se pudo reproducir la secuencia del sampler; no se continúa.")
            for paso, valor in anterior.intermediate_values.items():
                actual.report(valor, paso)
            replay.tell(actual, anterior.value if anterior.state == optuna.trial.TrialState.COMPLETE else None,
                        state=anterior.state)
        # Optuna 5.0 liga esta caché al identificador del estudio en memoria.
        # Se descarta solo la caché reconstruible; se conserva el estado aleatorio.
        # Esta integración interna queda cubierta por la prueba de reanudación.
        replay.sampler._search_space = optuna.search_space.IntersectionSearchSpace(include_pruned=True)
        estudio.sampler = replay.sampler
    return estudio


def sincronizar_registro(estudio, ruta_log=RUTA_LOG):
    """Agrega cada ajuste y estado al esquema común, sin duplicar ni inventar métricas."""
    from modelado_comun import COLUMNAS_DEL_LOG
    previas = set(pd.read_csv(ruta_log)["notas"].dropna()) if Path(ruta_log).exists() else set()
    for trial in estudio.trials:
        parametros = {**PARAMS_FIJOS, **trial.params}
        for fold in trial.user_attrs.get("pliegues", []):
            marca = f"Optuna {estudio.study_name}; trial={trial.number}; fold={fold['paso']}; estado_ajuste=COMPLETE; estado_trial={trial.state.name}"
            if marca in previas:
                continue
            fila = dict(zip(COLUMNAS_DEL_LOG, [None] * len(COLUMNAS_DEL_LOG)))
            fila.update(fecha_hora=fold["fin_utc"], modelo="gradient_boosting", tramo="entrenamiento",
                        filas=fold["filas_comunes"], hiperparametros=json.dumps(parametros, sort_keys=True),
                        **{k: fold[k] for k in ("mae", "rmse", "r2")}, semilla=SEMILLA,
                        commit=estudio.user_attrs.get("commit_codigo", _estado_de_git()[0]),
                        arbol_sin_commitear=estudio.user_attrs.get("arbol_sin_commitear", True), notas=marca)
            Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([fila], columns=COLUMNAS_DEL_LOG).to_csv(
                ruta_log, mode="a", header=not Path(ruta_log).exists(), index=False, lineterminator="\n")
            previas.add(marca)
        marca = f"Optuna {estudio.study_name}; trial={trial.number}; estado={trial.state.name}; CV temporal 5 pliegues; solo entrenamiento"
        if marca in previas or not trial.state.is_finished():
            continue
        metricas = trial.user_attrs.get("metricas", {}) if trial.state.name == "COMPLETE" else {}
        fila = {"fecha_hora": trial.datetime_complete.isoformat(), "modelo": "gradient_boosting",
                "tramo": "entrenamiento", "filas": len(trial.user_attrs.get("predicciones", {}).get("real", [])),
                "hiperparametros": json.dumps(parametros, sort_keys=True),
                **{k: metricas.get(k) for k in ("mae", "rmse", "r2")}, "semilla": SEMILLA,
                "commit": estudio.user_attrs.get("commit_codigo", _estado_de_git()[0]),
                "arbol_sin_commitear": estudio.user_attrs.get("arbol_sin_commitear", True), "notas": marca}
        Path(ruta_log).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([fila], columns=COLUMNAS_DEL_LOG).to_csv(
            ruta_log, mode="a", header=not Path(ruta_log).exists(), index=False, lineterminator="\n")
        previas.add(marca)


def ejecutar_busqueda(estudio, objective, total=100, ruta_log=RUTA_LOG):
    """ContinÃºa hasta el lÃ­mite TOTAL; un Ãºnico proceso, un trial a la vez."""
    if not 0 <= total <= 100:
        raise ValueError("El mÃ¡ximo total debe estar entre 0 y 100.")
    sincronizar_registro(estudio, ruta_log)
    if len(estudio.trials) > total:
        raise ValueError("El estudio ya supera el total solicitado; no se modifica.")
    restantes = total - len(estudio.trials)
    estudio.optimize(objective, n_trials=restantes, n_jobs=1,
                      callbacks=[lambda s, t: (sincronizar_registro(s, ruta_log), mostrar_progreso(s, t, total, ruta_log),
                          exportar_resultados(s, Path(ruta_log).parent / "gradient_boosting"))], catch=(ValueError, RuntimeError))
    return estudio


def exportar_resultados(estudio, carpeta):
    """Exporta estados, parÃ¡metros y detalle por pliegue sin mÃ©tricas finales inventadas."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = [{"trial": t.number, "estado": t.state.name, **t.params,
              **t.user_attrs.get("metricas", {}), "pliegues_completados": len(t.user_attrs.get("pliegues", [])),
              "motivo": t.user_attrs.get("motivo", ""),
              "inicio": t.datetime_start.isoformat(), "fin": t.datetime_complete.isoformat() if t.datetime_complete else None,
              "duracion_segundos": t.duration.total_seconds() if t.duration else None} for t in estudio.trials]
    pd.DataFrame(filas).to_csv(carpeta / "trials.csv", index=False, lineterminator="\n")
    detalle = [{"trial": t.number, "estado": t.state.name, "parametros": t.params,
                "intermedios": t.intermediate_values, "inicio": t.datetime_start.isoformat(),
                "fin": t.datetime_complete.isoformat() if t.datetime_complete else None, **t.user_attrs} for t in estudio.trials]
    (carpeta / "detalle.json").write_text(json.dumps({"configuracion": estudio.user_attrs,
                                                     "trials": detalle}, ensure_ascii=False, indent=2), encoding="utf-8")


def guardar_evidencia(estudio, carpeta=None):
    """Copia consistente de SQLite y resultados para versionar al terminar la búsqueda."""
    carpeta = RAIZ / "results" / "gradient_boosting" if carpeta is None else Path(carpeta)
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
               "duracion_segundos": trial.duration.total_seconds() if trial.duration else None,
               "nuevo_mejor": bool(mejor and mejor.number == trial.number),
               "mae": trial.value if trial.state.name == "COMPLETE" else None,
               "mae_parcial": trial.value if trial.state.name == "PRUNED" else None,
               "mejor_mae": mejor.value if mejor else None,
               "mejor_trial": mejor.number if mejor else None}
    if mejor and mejor.number == trial.number:
        mensaje["nuevos_mejores_parametros"] = mejor.params
    print(json.dumps(mensaje, ensure_ascii=False), flush=True)
    ruta = Path(ruta_log).parent / "gradient_boosting" / "progreso.jsonl"
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
    if len(estudio.trials) != 100 or any(not t.state.is_finished() for t in estudio.trials):
        raise ValueError("La evaluación requiere exactamente 100 trials terminados.")
    parametros = {**PARAMS_FIJOS, **estudio.best_params}
    modelo = crear_pipeline(train["X"].columns, **parametros)
    inicio = datetime.now(timezone.utc).isoformat()
    reloj = perf_counter()
    modelo.fit(train["X"], train["y"])
    pred = modelo.predict(val["X"].loc[comunes])
    marca = f"Optuna {estudio.study_name}; ganador trial={estudio.best_trial.number}; validacion independiente; solo ajuste entrenamiento"
    registro = registrar_experimento("gradient_boosting", parametros, val["y"].loc[comunes], pred,
                                    "validacion", notas=marca + "; estado=COMPLETE")
    modelos = RAIZ / "models" / "gradient_boosting"
    modelos.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, modelos / "modelo_validacion.joblib")

    pd.DataFrame({"indice_csv": val["X"].index[comunes], "DIA": val["fechas"].loc[comunes],
                  "real": val["y"].loc[comunes], "predicho": pred}).to_csv(carpeta / "predicciones_validacion.csv", index=False)
    resultado = {"trial": estudio.best_trial.number, "mae_cv": estudio.best_value,
                 "parametros": parametros, "metricas": {k: registro[k] for k in ("mae", "rmse", "r2")},
                 "inicio_utc": inicio, "fin_utc": datetime.now(timezone.utc).isoformat(),
                 "duracion_segundos": perf_counter() - reloj, "filas_ajuste": len(train["X"]), "filas_validacion": int(comunes.sum()),
                 "registro": registro, "configuracion": estudio.user_attrs["configuracion"],
                 "modelo": "models/gradient_boosting/modelo_validacion.joblib",
                 "alcance": "Modelo para validación; sin reentrenar con validación y sin evaluar test"}
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def main():
    """Cien trials, evaluación independiente y diagnóstico completo del candidato."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recuperar-interrumpidos", action="store_true",
                        help="Solo sin otro proceso activo: marcar RUNNING como FAIL, conservar su número y continuar hasta 100.")
    args = parser.parse_args()
    if args.recuperar_interrumpidos and RUTA_ESTUDIO.exists():
        previo = optuna.load_study(study_name=NOMBRE_ESTUDIO, storage=f"sqlite:///{RUTA_ESTUDIO.as_posix()}")
        for trial in previo.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                previo.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = abrir_estudio()
    commit, sucio = _estado_de_git()
    estudio.set_user_attr("commit_codigo", estudio.user_attrs.get("commit_codigo", commit))
    estudio.set_user_attr("arbol_sin_commitear", sucio)
    salida = RAIZ / "results" / "gradient_boosting"
    ejecutar_busqueda(estudio, crear_objective(cargar_entrenamiento()))
    guardar_evidencia(estudio, salida)
    evaluacion = evaluar_ganador(estudio, salida)
    from diagnostico_gradient_boosting import generar_diagnostico
    generar_diagnostico(estudio)
    print(json.dumps({"validacion": evaluacion["metricas"], "trial": estudio.best_trial.number}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

