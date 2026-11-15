"""Estudio independiente de XGBoost con 15 predictoras; conserva el estudio original."""
import hashlib
import json
from pathlib import Path

import modelado_xgboost as comun
from modelado_comun import RAIZ, RUTA_LOG, _estado_de_git

NOMBRE = "xgboost-temporal-15-features-v1"
CARPETA = RAIZ / "results" / "optuna" / "xgboost_15"
SQLITE = CARPETA / "estudio_15.sqlite3"
SALIDA = RAIZ / "results" / "xgboost_15"


def main():
    """Ejecuta hasta 100 trials totales con la misma metodología del estudio original."""
    commit, sucio = _estado_de_git()
    if sucio is not False or commit == "desconocido":
        raise RuntimeError("El código debe estar versionado y el árbol limpio.")
    anteriores = list((RAIZ / "results" / "xgboost").rglob("*"))
    originales = {str(p.relative_to(RAIZ)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in anteriores if p.is_file()}
    viejo_sqlite = comun.RUTA_ESTUDIO
    if viejo_sqlite.exists():
        originales[str(viejo_sqlite.relative_to(RAIZ))] = hashlib.sha256(viejo_sqlite.read_bytes()).hexdigest()
    log_anterior = RUTA_LOG.read_bytes()
    entrenamiento = comun.cargar_entrenamiento()
    if entrenamiento["X"].shape[1] != 15 or "es_fin_de_anio" not in entrenamiento["X"]:
        raise ValueError("Se requieren las 15 predictoras, incluido es_fin_de_anio.")
    comun.NOMBRE_ESTUDIO = NOMBRE
    comun.RUTA_ESTUDIO = SQLITE
    contrato = comun.configuracion()
    contrato.update({"predictoras": list(entrenamiento["X"].columns), "cantidad_predictoras": 15,
                     "ejecutor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    estudio = comun.abrir_estudio(SQLITE, contrato)
    estudio.set_user_attr("commit_codigo", estudio.user_attrs.get("commit_codigo", commit))
    estudio.set_user_attr("seleccion", "15 features, incluye es_fin_de_anio; ADR 0015")
    print(f"Estudio: {NOMBRE}; predictoras: 15; previos: {len(estudio.trials)}; máximo total: 100", flush=True)
    comun.ejecutar_busqueda(estudio, comun.crear_objective(entrenamiento), total=100)
    comun.guardar_evidencia(estudio, SALIDA)
    anterior = json.loads((RAIZ / "results" / "xgboost" / "resumen.json").read_text())
    mae_anterior = anterior["metricas_provisionales"]["mae"]
    mae_nuevo = estudio.best_value
    comparacion = {"estudio_14": anterior["estudio"], "estudio_15": NOMBRE,
                   "mae_14": mae_anterior, "mae_15": mae_nuevo,
                   "diferencia_mae_15_menos_14": mae_nuevo - mae_anterior,
                   "mejora_porcentual": 100 * (mae_anterior - mae_nuevo) / mae_anterior,
                   "mejora": mae_nuevo < mae_anterior,
                   "alcance": "MAE provisional de CV temporal en entrenamiento; búsquedas independientes, no evaluación final"}
    # Verificación de conservación de resultados; no ejecuta pruebas ni modelos adicionales.
    for nombre, huella in originales.items():
        if hashlib.sha256((RAIZ / nombre).read_bytes()).hexdigest() != huella:
            raise RuntimeError(f"Cambió un artefacto anterior: {nombre}")
    if not RUTA_LOG.read_bytes().startswith(log_anterior):
        raise RuntimeError("El log previo debe conservarse íntegro como prefijo.")
    (SALIDA / "comparacion_14_vs_15.json").write_text(json.dumps(comparacion, ensure_ascii=False, indent=2), encoding="utf-8")
    (SALIDA / "preservacion_originales.json").write_text(json.dumps(originales, indent=2), encoding="utf-8")
    print(json.dumps(comparacion, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
