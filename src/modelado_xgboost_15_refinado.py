"""Segunda búsqueda independiente: cuatro hiperparámetros, 60 trials, 15 predictoras."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import modelado_xgboost as comun
from modelado_comun import RAIZ, RUTA_LOG, _estado_de_git

NOMBRE = "xgboost-temporal-15-features-refinado-v1"
SQLITE = RAIZ / "results/optuna/xgboost_15_refinado/estudio_60.sqlite3"
SALIDA = RAIZ / "results/xgboost_15_refinado"
ESPACIO = {
    "n_estimators": {"tipo": "int", "low": 1750, "high": 3000},
    "min_child_weight": {"tipo": "float", "low": 4.0, "high": 5.0, "log": True},
    "subsample": {"tipo": "float", "low": 0.87, "high": 0.97},
    "colsample_bytree": {"tipo": "float", "low": 0.80, "high": 1.0},
}


def main():
    """Optimiza solo entrenamiento y exporta evidencia nueva sin tocar estudios anteriores."""
    commit, sucio = _estado_de_git()
    if sucio is not False or commit == "desconocido":
        raise RuntimeError("El código debe estar versionado y el árbol limpio.")
    fuente = RAIZ / "results/xgboost_15/detalle.json"
    detalle = json.loads(fuente.read_text(encoding="utf-8"))
    top = sorted([t for t in detalle["trials"] if t["estado"] == "COMPLETE"], key=lambda t: t["metricas"]["mae"])[:10]
    ganador = top[0]
    fijos = {**comun.PARAMS_FIJOS, "learning_rate": ganador["parametros"]["learning_rate"],
             "max_depth": ganador["parametros"]["max_depth"]}
    anteriores = [p for carpeta in ("xgboost", "xgboost_15", "xgboost_15_validacion")
                  for p in (RAIZ / "results" / carpeta).rglob("*") if p.is_file()]
    anteriores += [p for p in (RAIZ / "results/optuna/xgboost/estudio.sqlite3",
                                RAIZ / "results/optuna/xgboost_15/estudio_15.sqlite3") if p.exists()]
    huellas = {str(p.relative_to(RAIZ)): hashlib.sha256(p.read_bytes()).hexdigest() for p in anteriores}
    log_previo = RUTA_LOG.read_bytes()
    entrenamiento = comun.cargar_entrenamiento()
    original = detalle["configuracion"]["configuracion"]
    if list(entrenamiento["X"].columns) != original["predictoras"]:
        raise ValueError("Las predictoras cambiaron respecto del estudio anterior.")
    comun.NOMBRE_ESTUDIO, comun.RUTA_ESTUDIO = NOMBRE, SQLITE
    comun.ESPACIO, comun.PARAMS_FIJOS = ESPACIO, fijos
    contrato = comun.configuracion()
    if contrato["dataset_sha256"] != original["dataset_sha256"]:
        raise ValueError("El dataset cambió respecto de la primera búsqueda de 15 variables.")
    contrato.update({"max_trials": 60, "cantidad_predictoras": 15, "predictoras": original["predictoras"],
                     "ejecutor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "fuente_top10_sha256": hashlib.sha256(fuente.read_bytes()).hexdigest()})
    estudio = comun.abrir_estudio(SQLITE, contrato)
    estudio.set_user_attr("commit_codigo", estudio.user_attrs.get("commit_codigo", commit))
    estudio.set_user_attr("seleccion", "15 features; refinamiento de cuatro hiperparámetros; ADR 0019")
    print(f"{NOMBRE}; 60 trials totales; espacio={ESPACIO}; fijos={fijos}", flush=True)
    comun.ejecutar_busqueda(estudio, comun.crear_objective(entrenamiento), total=60)
    comun.guardar_evidencia(estudio, SALIDA)
    pd.DataFrame([{"trial": t["trial"], "mae": t["metricas"]["mae"], **t["parametros"]} for t in top]).to_csv(SALIDA / "top10_estudio_anterior.csv", index=False)
    mae_anterior, nuevo = ganador["metricas"]["mae"], estudio.best_value
    comparacion = {"estudio_anterior": "xgboost-temporal-15-features-v1", "estudio_nuevo": NOMBRE,
                   "mae_anterior": mae_anterior, "mae_nuevo": nuevo,
                   "diferencia_nuevo_menos_anterior": nuevo - mae_anterior,
                   "mejora_porcentual": (mae_anterior - nuevo) / mae_anterior * 100,
                   "fijos": fijos, "rangos": ESPACIO,
                   "justificacion": "Top 10: árboles 1757–1982, peso 4.09035–4.97287, filas 0.87423–0.95912, columnas 0.80861–0.98423. Redondeo exterior y extensión de árboles hasta 3000.",
                   "alcance": "CV temporal de entrenamiento; sin validación independiente ni prueba; resultado provisional"}
    for nombre, huella in huellas.items():
        if hashlib.sha256((RAIZ / nombre).read_bytes()).hexdigest() != huella:
            raise RuntimeError(f"Se modificó un resultado anterior: {nombre}")
    if not RUTA_LOG.read_bytes().startswith(log_previo):
        raise RuntimeError("Se debe conservar el prefijo íntegro del log anterior.")
    (SALIDA / "comparacion.json").write_text(json.dumps(comparacion, ensure_ascii=False, indent=2), encoding="utf-8")
    (SALIDA / "preservacion_originales.json").write_text(json.dumps(huellas, indent=2), encoding="utf-8")
    print(json.dumps(comparacion, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
