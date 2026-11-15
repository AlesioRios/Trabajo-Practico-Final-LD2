"""Diagnóstico offline de LightGBM a partir de resultados archivados.

Reutiliza la aritmética pura del diagnóstico XGBoost integrado en feature/modeling.
No abre datasets, modelos serializados ni SQLite; no importa bibliotecas de modelado.
"""

import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostico_xgboost import (
    METRICAS, _validar_registro, comprobar_metricas, huella, leer_json,
    metricas_guardadas, preparar_residuos, resumir_residuos,
)

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = "results/lightgbm_diagnostico_validacion"
EXPERIMENTOS = {"primera": "lightgbm", "segunda": "lightgbm_2"}
FIGURAS = ("prediccion_vs_real", "predicciones_temporales", "distribucion_residuos",
           "residuos_vs_predicciones", "residuos_temporales")


def auditar_tracking(raiz=RAIZ):
    """Concilia parámetros, métricas, timestamps y commits sin modificar el log."""
    raiz = Path(raiz)
    log = pd.read_csv(raiz / "results/experiment_logs.csv", keep_default_na=False)
    propios = log.loc[log.modelo.eq("lightgbm")]
    usados, filas, estados = [], [], []
    for carpeta in EXPERIMENTOS.values():
        detalle = leer_json(raiz / "results" / carpeta / "detalle.json")
        resumen = leer_json(raiz / "results" / carpeta / "resumen.json")
        estudio = resumen["estudio"]
        estados.append({"estudio": estudio, **detalle_estados(detalle)})
        for trial in detalle["trials"]:
            if trial["estado"] != "COMPLETE":
                continue
            nota = f"Optuna {estudio}; trial={trial['trial']}; CV temporal 5 pliegues; solo entrenamiento"
            registros = propios.loc[propios.notas.eq(nota) & propios.tramo.eq("entrenamiento")]
            if len(registros) != 1:
                raise ValueError(f"Registro ausente o duplicado: {nota}")
            pred = trial["predicciones"]
            metricas = metricas_guardadas(pred["real"], pred["predicho"])
            comprobar_metricas(metricas, trial["metricas"])
            parametros = {**detalle["configuracion"]["configuracion"]["fijos"], **trial["parametros"]}
            registro = registros.iloc[0]
            _validar_registro(registro, parametros, metricas, len(pred["real"]))
            usados.append(registros.index[0])
            filas.append({"estudio": estudio, "trial": trial["trial"], "tramo": "entrenamiento",
                          "ajustes_agrupados": len(trial["pliegues"]),
                          "fecha_hora_original": registro.fecha_hora, "commit": registro.commit,
                          "arbol_sin_commitear": registro.arbol_sin_commitear,
                          "filas": len(pred["real"]), **metricas})
        evaluacion = leer_json(raiz / "results" / carpeta / "validacion.json")
        registro = evaluacion["registro"]
        registros = propios.loc[propios.notas.eq(registro["notas"]) & propios.tramo.eq("validacion")]
        if len(registros) != 1:
            raise ValueError("Registro de validación ausente o duplicado.")
        fila = registros.iloc[0]
        for clave in ("fecha_hora", "commit", "hiperparametros", "arbol_sin_commitear"):
            if fila[clave] != registro[clave]:
                raise ValueError(f"Difiere la evidencia histórica: {clave}")
        _validar_registro(fila, evaluacion["parametros"], evaluacion["metricas"], evaluacion["filas_validacion"])
        usados.append(registros.index[0])
        filas.append({"estudio": estudio, "trial": evaluacion["trial"], "tramo": "validacion",
                      "ajustes_agrupados": 1, "fecha_hora_original": fila.fecha_hora,
                      "commit": fila.commit, "arbol_sin_commitear": fila.arbol_sin_commitear,
                      "filas": int(fila.filas), **evaluacion["metricas"]})
    if len(set(usados)) != len(usados) or set(usados) != set(propios.index):
        raise ValueError("Hay filas LightGBM sin correspondencia o reutilizadas.")
    # La firma por registros tolera integrar otras familias sin invalidar la evidencia.
    firma = propios.sort_values(["tramo", "notas"]).to_json(orient="records", double_precision=15)
    import hashlib
    return pd.DataFrame(filas), {
        "registros_lightgbm": len(propios), "cv_completos": len(filas) - 2, "validaciones": 2,
        "estudios": estados, "registros_propios_sha256": hashlib.sha256(firma.encode()).hexdigest(),
        "cumplimiento_literal_por_ajuste": "Parcial",
        "limite": "Una fila por COMPLETE agrupa cinco ajustes. Los PRUNED quedan en JSON/Optuna. "
                  "Timestamp de registro sin zona horaria; no equivale al inicio de cada ajuste. "
                  "Las corridas conservan arbol_sin_commitear=True; código y hashes están archivados.",
    }


def detalle_estados(detalle):
    """Cuenta estados tal como fueron exportados por las búsquedas existentes."""
    return {estado.lower(): sum(t["estado"] == estado for t in detalle["trials"])
            for estado in ("COMPLETE", "PRUNED", "FAIL")}


def graficar(tablas, salida):
    """Mantiene nombres, paneles y convención de residuos del diagnóstico XGBoost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    todos = pd.concat(tablas.values())
    extremos = [todos[["real", "predicho"]].min().min(), todos[["real", "predicho"]].max().max()]
    bins = np.histogram_bin_edges(todos.residuo, bins=18)
    limite = todos.residuo.abs().max() * 1.08
    titulos = ("Predicciones frente a valores reales", "Valores reales y predicciones en validación",
               "Distribución de residuos", "Residuos frente a predicciones", "Evolución de los residuos")
    for nombre, titulo in zip(FIGURAS, titulos):
        fig, ejes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
        for eje, (experimento, tabla) in zip(ejes, tablas.items()):
            eje.set_title(f"{experimento.capitalize()} búsqueda · trial {72 if experimento == 'primera' else 99}", loc="left")
            if nombre == "prediccion_vs_real":
                eje.scatter(tabla.real, tabla.predicho, s=20, alpha=.7)
                eje.plot(extremos, extremos, "k--", label="Predicción exacta")
                eje.set(xlabel="Real (kW)", ylabel="Predicho (kW)", xlim=extremos, ylim=extremos)
                eje.legend()
            elif nombre == "predicciones_temporales":
                eje.plot(tabla.DIA, tabla.real, label="Real", linewidth=1.1)
                eje.plot(tabla.DIA, tabla.predicho, label="Predicho", linewidth=1.1)
                eje.set(xlabel="Fecha rotulada de emisión D (objetivo: bloque siguiente)", ylabel="Consumo (kW)")
                eje.legend()
            elif nombre == "distribucion_residuos":
                eje.hist(tabla.residuo, bins=bins, color="#2878a6", edgecolor="white")
                eje.axvline(0, color="black", linestyle="--", label="Cero")
                eje.axvline(tabla.residuo.mean(), color="#bd541a", label="Media")
                eje.set(xlabel="Residuo = real − predicho (kW)", ylabel="Filas")
                eje.legend()
            else:
                eje.scatter(tabla.predicho if nombre == "residuos_vs_predicciones" else tabla.DIA,
                            tabla.residuo, s=20, alpha=.7)
                eje.axhline(0, color="black", linestyle="--")
                eje.set(ylabel="Real − predicho (kW)", ylim=(-limite, limite),
                        xlabel="Predicho (kW)" if nombre == "residuos_vs_predicciones"
                        else "Fecha rotulada de emisión D (objetivo: bloque siguiente)")
            eje.grid(alpha=.2)
        fig.suptitle(f"LightGBM · 171 filas de validación · {titulo}", fontsize=14)
        fig.savefig(salida / f"{nombre}.png", dpi=160, facecolor="white")
        plt.close(fig)


def generar_diagnostico(raiz=RAIZ):
    """Deriva tablas y PNG en carpeta propia, sin modificar las fuentes originales."""
    raiz = Path(raiz).resolve()
    fuentes = [raiz / "results/experiment_logs.csv"]
    for carpeta in EXPERIMENTOS.values():
        fuentes.extend(p for p in (raiz / "results" / carpeta).rglob("*") if p.is_file())
    originales = {p.relative_to(raiz).as_posix(): huella(p) for p in fuentes}
    auditoria, tracking = auditar_tracking(raiz)
    tablas, resumenes, mensuales, mayores, parametros, rangos = {}, [], [], [], [], []
    referencia, contrato_anterior = None, None
    for nombre, carpeta in EXPERIMENTOS.items():
        directorio = raiz / "results" / carpeta
        evaluacion = leer_json(directorio / "validacion.json")
        resumen = leer_json(directorio / "resumen.json")
        detalle = leer_json(directorio / "detalle.json")
        contrato = detalle["configuracion"]["configuracion"]
        if (len(contrato["predictoras"]) != 15 or len(set(contrato["predictoras"])) != 15
                or contrato["pliegues"] != 5 or contrato["gap"] != 1):
            raise ValueError("Contrato de variables o partición temporal incompatible.")
        if contrato_anterior is not None:
            for clave in ("predictoras", "dataset_sha256", "linaje_sha256"):
                if contrato[clave] != contrato_anterior[clave]:
                    raise ValueError("Las búsquedas no comparten el contrato de datos.")
        contrato_anterior = contrato
        ganador = next(t for t in detalle["trials"] if t["trial"] == evaluacion["trial"])
        completos = [t for t in detalle["trials"] if t["estado"] == "COMPLETE"]
        if (ganador["estado"] != "COMPLETE" or ganador["trial"] != resumen["mejor_trial_provisional"]
                or ganador["metricas"]["mae"] != min(t["metricas"]["mae"] for t in completos)
                or evaluacion["parametros"] != {**contrato["fijos"], **ganador["parametros"]}):
            raise ValueError("Ganador o hiperparámetros incompatibles con el estudio archivado.")
        tabla = preparar_residuos(pd.read_csv(directorio / "predicciones_validacion.csv", float_precision="round_trip"))
        if len(tabla) != 171 or evaluacion["registro"]["tramo"] != "validacion":
            raise ValueError("Solo se admite la validación archivada de 171 filas.")
        claves = tabla[["indice_csv", "DIA", "real"]]
        if referencia is not None:
            pd.testing.assert_frame_equal(referencia, claves)
        referencia = claves
        estadisticas = resumir_residuos(tabla)
        comprobar_metricas(estadisticas, evaluacion["metricas"])
        tablas[nombre] = tabla
        resumenes.append({"experimento": nombre, "trial": evaluacion["trial"],
                          "mae_cv": resumen["metricas_provisionales"]["mae"], **estadisticas})
        for mes, parte in tabla.groupby(tabla.DIA.dt.strftime("%Y-%m"), sort=True):
            mensuales.append({"experimento": nombre, "mes_emision": mes, "filas": len(parte),
                              "sesgo": float(parte.residuo.mean()), "mae": float(parte.error_absoluto.mean())})
        mayores.append(tabla.nlargest(10, "error_absoluto").assign(experimento=nombre))
        parametros.extend({"experimento": nombre, "hiperparametro": k, "valor": v}
                          for k, v in evaluacion["parametros"].items())
        rangos.extend({"experimento": nombre, "hiperparametro": k, **v} for k, v in contrato["espacio"].items())
    log = pd.read_csv(raiz / "results/experiment_logs.csv")
    bases = log.loc[log.modelo.isin(["persistencia", "media_movil"]) & log.tramo.eq("validacion")]
    if len(bases) != 2 or not bases.filas.eq(171).all():
        raise ValueError("Baselines ambiguos o evaluados en otra cantidad de filas.")
    comparacion = pd.DataFrame([
        {"experimento": s["experimento"], "filas": s["filas"], **{k: s[k] for k in METRICAS}}
        for s in resumenes
    ] + [{"experimento": r.modelo, "filas": int(r.filas), **{k: float(r[k]) for k in METRICAS}}
         for _, r in bases.iterrows()])
    salida = raiz / SALIDA
    salida.mkdir(parents=True, exist_ok=True)
    productos = {"resumen_residuos": pd.DataFrame(resumenes), "residuos_por_mes": pd.DataFrame(mensuales),
                 "errores_mayores": pd.concat(mayores), "hiperparametros": pd.DataFrame(parametros),
                 "espacios_busqueda": pd.DataFrame(rangos), "comparacion_validacion": comparacion,
                 "auditoria_tracking": auditoria,
                 **{f"residuos_{k}": v for k, v in tablas.items()}}
    for nombre, tabla in productos.items():
        tabla.to_csv(salida / f"{nombre}.csv", index=False, lineterminator="\n")
    graficar(tablas, salida)
    intactos = all(huella(raiz / p) == valor for p, valor in originales.items())
    if not intactos:
        raise ValueError("Se modificó una fuente durante el diagnóstico.")
    manifiesto = {
        "fecha_diagnostico": "2026-11-15", "fuentes_sha256": originales,
        "nota_log": "El hash completo acredita preservación en esta ejecución. La firma de registros propios "
                    "permite integrar otras familias sin invalidar la evidencia LightGBM.",
        "figuras": [f"{f}.png" for f in FIGURAS], "tracking": tracking,
        "controles": {"metricas_coinciden_con_evaluaciones": True, "mismas_171_filas_y_objetivos": True,
                      "contrato_15_variables_igual": True, "ganadores_y_parametros_verificados": True,
                      "tracking_conciliado": True, "originales_intactos": intactos},
        "python": platform.python_version(),
        "versiones": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib")},
        "ayudante_reutilizado": {"ruta": "src/diagnostico_xgboost.py",
                                  "sha256": huella(raiz / "src/diagnostico_xgboost.py")},
        "alcance": "CSV/JSON/PNG archivados; sin modelos, entrenamiento, predicción, Optuna ni dataset/test.",
    }
    (salida / "verificacion.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifiesto


if __name__ == "__main__":
    print(json.dumps(generar_diagnostico()["controles"], ensure_ascii=False))
