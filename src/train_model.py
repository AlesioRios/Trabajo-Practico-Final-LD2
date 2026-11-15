"""Cierre Lasso: entrenamiento y evaluación local explícita, única y auditable."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from auxiliar_functions import SEMILLA
from modelado_comun import calcular_metricas, filas_comunes, registrar_experimento
from pipeline_sklearn import armar_pipeline

RAIZ = Path(__file__).resolve().parents[1]
PARAMETROS = {"alpha": 75.0, "max_iter": 20000, "random_state": SEMILLA}
VALIDACION = {"mae": 2215.796375483132, "rmse": 2835.803833786694, "r2": 0.4676090070547668}


def ahora():
    """Timestamp explícito para metadatos nuevos."""
    return datetime.now(timezone.utc).isoformat()


def huella(ruta):
    """SHA-256 de bytes, sin interpretar observaciones."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def leer_json(ruta):
    """Lee metadatos."""
    return json.loads(Path(ruta).read_text(encoding="utf-8-sig"))


def guardar_json(ruta, datos):
    """Publica metadatos completos mediante reemplazo atómico."""
    ruta = Path(ruta)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporal.replace(ruta)


def columnas_ganadoras(raiz=RAIZ):
    """Nombres y orden históricos; no selecciona variables de nuevo."""
    columnas = leer_json(raiz / "data/processed/seleccion_original_14/data_lineage.json")["columnas"]
    if len(columnas) != 14 or len(set(columnas)) != 14 or "es_fin_de_anio" in columnas:
        raise ValueError("Se requieren las 14 entradas históricas ganadoras.")
    return columnas


def cargar_tramos(ruta, tramos, columnas):
    """Filtra antes de convertir valores; entrenamiento no materializa filas test.

    Conserva el índice físico del CSV. No cambia limpieza, etiquetas ni fronteras.
    """
    elegidas, indices = [], []
    with Path(ruta).open(encoding="utf-8-sig", newline="") as archivo:
        for indice, fila in enumerate(csv.DictReader(archivo)):
            if fila["tramo"] in tramos:
                elegidas.append(fila); indices.append(indice)
    tabla = pd.DataFrame(elegidas, index=indices)
    if tabla.empty:
        raise ValueError("No hay observaciones en los tramos solicitados.")
    campos = [*columnas, "objetivo", "consumo_dia", "baseline_media_movil"]
    tabla[campos] = tabla[campos].apply(pd.to_numeric, errors="raise")
    tabla["DIA"] = pd.to_datetime(tabla["DIA"], errors="raise")
    if not tabla.DIA.is_monotonic_increasing or not np.isfinite(tabla.objetivo).all():
        raise ValueError("Orden temporal u objetivos inválidos.")
    if np.isinf(tabla[columnas].to_numpy(dtype=float)).any():
        raise ValueError("Predictores infinitos.")
    return tabla


def fabricar_modelo(columnas):
    """Reutiliza exactamente pipeline A con el estimador ganador."""
    return armar_pipeline("A", columnas).set_params(regressor=Lasso(**PARAMETROS))


def siguiente_version(registro, carpeta):
    """Primera versión patch disponible sin sobrescribir historia."""
    patch = 0
    while f"1.0.{patch}" in registro or (carpeta / f"modelo_v1.0.{patch}.pkl").exists():
        patch += 1
    return f"1.0.{patch}"


def entrenar(raiz=RAIZ):
    """Ajusta y verifica dos ajustes idénticos solo con train+validation."""
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("El cierre requiere Python 3.12, entorno cervecera_env.")
    salida = raiz / "results/lasso_final"
    salida.mkdir(parents=True, exist_ok=True)
    if (salida / "entrenamiento.json").exists():
        raise FileExistsError("Ya existe entrenamiento final: recuperar, no reajustar.")
    rutas = ["src", "tests", "requirements.txt", "data/processed", "docs/decisions/0024-cierre-lasso.md"]
    estado = subprocess.check_output(["git", "status", "--porcelain", "--", *rutas], cwd=raiz, text=True)
    if estado.strip():
        raise RuntimeError("Guardar código y contrato en un commit antes del ajuste final.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=raiz, text=True).strip()
    columnas = columnas_ganadoras(raiz)
    ruta_datos = raiz / "data/processed/dataset_final.csv"
    datos = cargar_tramos(ruta_datos, {"entrenamiento", "validacion"}, columnas)
    if list(datos.tramo.drop_duplicates()) != ["entrenamiento", "validacion"]:
        raise ValueError("Se requiere entrenamiento seguido por validación.")
    X, y = datos[columnas], datos.objetivo
    inicio = ahora()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        modelo = fabricar_modelo(columnas).fit(X, y)
        replica = fabricar_modelo(columnas).fit(X, y)
    pred = modelo.predict(X)
    np.testing.assert_allclose(replica.predict(X), pred, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(replica.named_steps["regressor"].coef_, modelo.named_steps["regressor"].coef_)
    ruta_registry = raiz / "models/model_registry.json"
    registro = leer_json(ruta_registry) if ruta_registry.exists() else {}
    version = siguiente_version(registro, raiz / "models")
    relativa = f"models/modelo_v{version}.pkl"
    modelo.cierre_metadata_ = {"version": version, "columnas": columnas, "commit_entrenamiento": commit,
                              "dataset_sha256": huella(ruta_datos)}
    with (raiz / relativa).open("xb") as archivo:
        joblib.dump(modelo, archivo, compress=3)
    recuperado = joblib.load(raiz / relativa)
    np.testing.assert_array_equal(recuperado.predict(X), pred)
    pre = modelo.named_steps["preprocessor"].named_transformers_["numericas"]
    np.testing.assert_allclose(pre.named_steps["imputar"].statistics_, X.median().to_numpy(), rtol=0, atol=1e-10)
    coef = modelo.named_steps["regressor"].coef_
    pd.DataFrame({"variable": columnas, "coeficiente_estandarizado": coef, "magnitud_absoluta": np.abs(coef),
        "escala_train_validation": pre.named_steps["escalar"].scale_}).sort_values("magnitud_absoluta", ascending=False).to_csv(
            salida / "coeficientes.csv", index=False)
    pd.DataFrame({"indice_csv": datos.index, "DIA": datos.DIA, "real": y, "predicho": pred}).to_csv(
        salida / "predicciones_ajuste.csv", index=False)
    evento = registrar_experimento(f"lasso_v{version}", PARAMETROS, y, pred, "entrenamiento",
        notas=f"version={version}; evento=reentrenamiento_final; ajuste=train+validation; metricas dentro de muestra; validacion original conservada",
        ruta=raiz / "results/experiment_logs.csv")
    meta = {"version": version, "modelo": "Lasso", "pipeline": "A", "hiperparametros": PARAMETROS,
        "columnas": columnas, "metricas_validacion_original": VALIDACION,
        "fuente_seleccion": "results/comparacion_modelos_fuentes.csv", "path": relativa, "commit": commit,
        "fecha_entrenamiento": inicio, "fin_entrenamiento": ahora(), "filas_train_validation": len(datos),
        "filas_por_tramo": datos.tramo.value_counts().to_dict(), "indices_ajuste": datos.index.tolist(),
        "ultima_fecha_ajuste": datos.DIA.max().date().isoformat(), "dataset_sha256": huella(ruta_datos),
        "linaje_sha256": huella(raiz / "data/processed/data_lineage.json"), "modelo_sha256": huella(raiz / relativa),
        "python": platform.python_version(),
        "dependencias": {p: importlib.metadata.version(p) for p in ["scikit-learn", "numpy", "pandas", "joblib"]},
        "reproducibilidad": {"dos_ajustes_coinciden": True, "joblib_reproduce_predicciones": True,
                             "medianas_solo_train_validation": True},
        "metricas_ajuste_no_validacion": calcular_metricas(y, pred), "evento_log": evento,
        "estado": "congelado_pendiente_evaluacion_unica"}
    guardar_json(salida / "entrenamiento.json", meta)
    registro[version] = meta
    guardar_json(ruta_registry, registro)
    return meta


def validar_archivo_test(salida):
    """Audita resultados archivados sin cargar modelo ni dataset de prueba."""
    resultado = leer_json(salida / "evaluacion_test.json")
    if huella(salida / "predicciones_test.csv") != resultado["predicciones_sha256"]:
        raise ValueError("Cambió la evidencia de test archivada.")
    tabla = pd.read_csv(salida / "predicciones_test.csv")
    for nombre, campo in [("lasso", "predicho"), ("persistencia", "persistencia"), ("media_movil", "media_movil")]:
        for k, v in calcular_metricas(tabla.real, tabla[campo]).items():
            if not np.isclose(v, resultado["metricas"][nombre][k], rtol=1e-12, atol=1e-9):
                raise ValueError("Métricas y predicciones archivadas no coinciden.")
    return resultado


def verificar_entrenamiento(raiz=RAIZ):
    """Reproduce el ajuste histórico sin tocar test, artefacto, registry ni logs."""
    meta = leer_json(raiz / "results/lasso_final/entrenamiento.json")
    tabla = cargar_tramos(raiz / "data/processed/dataset_final.csv", {"entrenamiento", "validacion"}, meta["columnas"])
    if huella(raiz / "data/processed/dataset_final.csv") != meta["dataset_sha256"]:
        raise ValueError("Dataset diferente del usado para entrenar.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        replica = fabricar_modelo(meta["columnas"]).fit(tabla[meta["columnas"]], tabla.objetivo)
    referencia = pd.read_csv(raiz / "results/lasso_final/predicciones_ajuste.csv", float_precision="round_trip")
    np.testing.assert_allclose(replica.predict(tabla[meta["columnas"]]), referencia.predicho, rtol=0, atol=1e-10)
    return {"reproduce_train_validation": True, "filas": len(tabla), "test_usado": False}


def evaluar_test_una_vez(raiz=RAIZ):
    """Predice test una sola vez. El marcador exclusivo precede toda lectura test.

    Si se interrumpe la evaluación no se borra el marcador ni se reintenta. Una
    ejecución posterior completa solo consulta las predicciones ya archivadas.
    """
    salida = raiz / "results/lasso_final"
    if (salida / "evaluacion_test.json").exists():
        return validar_archivo_test(salida)
    meta = leer_json(salida / "entrenamiento.json")
    ruta_modelo = raiz / meta["path"]
    if huella(ruta_modelo) != meta["modelo_sha256"] or huella(raiz / "data/processed/dataset_final.csv") != meta["dataset_sha256"]:
        raise ValueError("El modelo o dataset difieren de los congelados.")
    with (raiz / "results/experiment_logs.csv").open(encoding="utf-8") as archivo:
        if any(r["tramo"] == "prueba" for r in csv.DictReader(archivo)):
            raise RuntimeError("Ya hay evaluación de prueba registrada: recuperar y auditar.")
    modelo = joblib.load(ruta_modelo)
    marcador = salida / "test_consumido.json"
    with marcador.open("x", encoding="utf-8") as archivo:
        json.dump({"inicio": ahora(), "version": meta["version"], "estado": "iniciado_no_repetir"}, archivo)
    test = cargar_tramos(raiz / "data/processed/dataset_final.csv", {"prueba"}, meta["columnas"])
    if min(test.index) <= max(meta["indices_ajuste"]) or test.DIA.min() < pd.Timestamp(meta["ultima_fecha_ajuste"]):
        raise ValueError("Frontera temporal inválida.")
    comunes = filas_comunes({"y": test.objetivo, "persistencia": test.consumo_dia, "media_movil": test.baseline_media_movil})
    evaluadas = test.loc[comunes]
    pred = modelo.predict(evaluadas[meta["columnas"]])  # única predicción sobre test local
    tabla = pd.DataFrame({"indice_csv": evaluadas.index, "DIA": evaluadas.DIA, "real": evaluadas.objetivo,
        "predicho": pred, "persistencia": evaluadas.consumo_dia, "media_movil": evaluadas.baseline_media_movil})
    tabla["residuo"] = tabla.real - tabla.predicho
    tabla["error_absoluto"] = tabla.residuo.abs()
    tabla.to_csv(salida / "predicciones_test.csv", index=False)
    metricas, eventos = {}, {}
    for nombre, campo in [("lasso", "predicho"), ("persistencia", "persistencia"), ("media_movil", "media_movil")]:
        metricas[nombre] = calcular_metricas(tabla.real, tabla[campo])
        eventos[nombre] = registrar_experimento(f"{nombre}_v{meta['version']}",
            PARAMETROS if nombre == "lasso" else ({} if nombre == "persistencia" else {"ventana": 7, "minimo_dias": 4}),
            tabla.real, tabla[campo], "prueba", evaluacion_final=True,
            notas=f"version={meta['version']}; evento=evaluacion_final_unica; soporte_comun_test_local; modelo_sha256={meta['modelo_sha256']}",
            ruta=raiz / "results/experiment_logs.csv")
    m = metricas["lasso"]["mae"]
    resultado = {"version": meta["version"], "timestamp": ahora(), "filas_test_etiquetadas": len(test),
        "filas_evaluadas": len(tabla), "indices_evaluados": tabla.indice_csv.tolist(),
        "indices_excluidos": test.index[~comunes].tolist(), "metricas": metricas,
        "comparaciones": {b: {"diferencia_mae": metricas[b]["mae"]-m,
            "mejora_porcentual": 100*(metricas[b]["mae"]-m)/metricas[b]["mae"]} for b in ["persistencia", "media_movil"]},
        "supera_ambos_baselines": all(m < metricas[b]["mae"] for b in ["persistencia", "media_movil"]),
        "mae_local_menor_3000": m < 3000, "modelo_sha256": meta["modelo_sha256"],
        "predicciones_sha256": huella(salida / "predicciones_test.csv"), "eventos_log": eventos,
        "evaluaciones_predict_test": 1, "advertencia": "Test local: no demuestra el umbral del test oculto de la cátedra."}
    guardar_json(salida / "evaluacion_test.json", resultado)
    registro = leer_json(raiz / "models/model_registry.json")
    registro[meta["version"]].update({"metricas_test_local": metricas, "filas_test": len(tabla),
        "evaluacion_test": "results/lasso_final/evaluacion_test.json", "estado": "evaluado_test_local"})
    guardar_json(raiz / "models/model_registry.json", registro)
    guardar_json(marcador, {"inicio": leer_json(marcador)["inicio"], "fin": ahora(), "estado": "completo_no_repetir",
        "version": meta["version"], "predicciones_sha256": resultado["predicciones_sha256"]})
    return validar_archivo_test(salida)


def main():
    """Sin argumentos no se entrena ni se evalúa."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accion", choices=["entrenar", "evaluar-test-una-vez", "consultar", "verificar-entrenamiento"])
    args = parser.parse_args()
    if args.accion == "entrenar": resultado = entrenar()
    elif args.accion == "evaluar-test-una-vez": resultado = evaluar_test_una_vez()
    elif args.accion == "verificar-entrenamiento": resultado = verificar_entrenamiento()
    else: resultado = validar_archivo_test(RAIZ / "results/lasso_final")
    print(json.dumps(resultado, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
