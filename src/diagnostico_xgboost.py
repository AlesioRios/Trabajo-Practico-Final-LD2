"""Diagnóstico de predicciones archivadas: no importa ni ejecuta estimadores.

Lee exclusivamente CSV/JSON de resultados y el log. No abre datasets, modelos
serializados ni bases mutables de Optuna. Los originales se verifican por SHA-256.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_SALIDA = "results/xgboost_diagnostico_validacion"
EXPERIMENTOS = {
    "inicial_15": ("xgboost_15", "xgboost_15_validacion"),
    "refinado_15": ("xgboost_15_refinado", "xgboost_15_refinado_validacion"),
}
METRICAS = ("mae", "rmse", "r2")


def leer_json(ruta):
    """Lee evidencia JSON, sin reconstruir objetos de las bibliotecas de modelado."""
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def huella(ruta):
    """Calcula SHA-256 de un archivo de evidencia."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def metricas_guardadas(real, predicho):
    """Recalcula métricas aritméticas de vectores guardados, sin ningún estimador."""
    real, predicho = np.asarray(real, dtype=float), np.asarray(predicho, dtype=float)
    if real.ndim != 1 or real.shape != predicho.shape or len(real) < 2:
        raise ValueError("Se requieren al menos dos pares real/predicho alineados.")
    if not np.isfinite(real).all() or not np.isfinite(predicho).all():
        raise ValueError("Las predicciones guardadas deben ser finitas.")
    residuo = real - predicho
    total = np.square(real - real.mean()).sum()
    if total == 0:
        raise ValueError("R² requiere variación del objetivo para esta auditoría.")
    return {"mae": float(np.abs(residuo).mean()),
            "rmse": float(np.sqrt(np.square(residuo).mean())),
            "r2": float(1 - np.square(residuo).sum() / total)}


def comprobar_metricas(calculadas, registradas):
    """Exige coincidencia numérica con la evaluación histórica."""
    for clave in METRICAS:
        if not np.isclose(calculadas[clave], float(registradas[clave]), rtol=1e-10, atol=1e-8):
            raise ValueError(f"No coincide la métrica guardada: {clave}.")


def preparar_residuos(tabla):
    """Valida identidad por índice CSV y define residuo como real menos predicho."""
    columnas = ["indice_csv", "DIA", "real", "predicho"]
    if not set(columnas).issubset(tabla.columns) or tabla.empty:
        raise ValueError("Faltan filas o columnas de predicciones.")
    nueva = tabla[columnas].copy()
    indices = pd.to_numeric(nueva["indice_csv"], errors="raise")
    if (not np.isfinite(indices).all() or (indices % 1 != 0).any()
            or not indices.is_unique or not indices.is_monotonic_increasing):
        raise ValueError("indice_csv debe ser entero, único y creciente.")
    nueva["indice_csv"] = indices.astype(int)
    nueva["DIA"] = pd.to_datetime(nueva["DIA"], errors="raise")
    if nueva["DIA"].isna().any() or not nueva["DIA"].is_monotonic_increasing:
        raise ValueError("Las fechas deben existir y estar ordenadas.")
    nueva[["real", "predicho"]] = nueva[["real", "predicho"]].astype(float)
    metricas_guardadas(nueva["real"], nueva["predicho"])
    nueva["residuo"] = nueva["real"] - nueva["predicho"]
    nueva["error_absoluto"] = nueva["residuo"].abs()
    return nueva


def resumir_residuos(tabla):
    """Resume magnitud, signo y concentración del error observado."""
    residuo, absoluto = tabla["residuo"], tabla["error_absoluto"]
    return {**metricas_guardadas(tabla["real"], tabla["predicho"]),
            "filas": len(tabla), "sesgo_real_menos_predicho": float(residuo.mean()),
            "mediana_residuo": float(residuo.median()),
            "desvio_residuo": float(residuo.std(ddof=1)),
            "p05_residuo": float(residuo.quantile(.05)),
            "p95_residuo": float(residuo.quantile(.95)),
            "p95_error_absoluto": float(absoluto.quantile(.95)),
            "max_error_absoluto": float(absoluto.max()),
            "subestimaciones": int((residuo > 0).sum()),
            "sobreestimaciones": int((residuo < 0).sum()),
            "fraccion_error_top10": float(absoluto.nlargest(10).sum() / absoluto.sum())
            if absoluto.sum() else 0.0}


def _validar_registro(fila, parametros, metricas, filas):
    """Comprueba campos del log sin modificar timestamps ni registros históricos."""
    datetime.fromisoformat(str(fila["fecha_hora"]))
    if not re.fullmatch(r"[0-9a-f]{40}", str(fila["commit"])):
        raise ValueError("Commit del log inválido.")
    if int(fila["semilla"]) != 42 or int(fila["filas"]) != filas:
        raise ValueError("Semilla o cantidad de filas incompatible.")
    if json.loads(fila["hiperparametros"]) != parametros:
        raise ValueError("Hiperparámetros del log incompatibles.")
    comprobar_metricas(metricas, fila)


def auditar_tracking(raiz):
    """Contrasta todos los COMPLETE archivados y ambas validaciones con el log.

    Los trials podados no se convierten en experimentos completos. Se informa
    explícitamente el límite del tracking por experimento y no por llamada de ajuste.
    """
    raiz = Path(raiz)
    log = pd.read_csv(raiz / "results/experiment_logs.csv", keep_default_na=False)
    propios = log.loc[log["modelo"].eq("xgboost")]
    usados, filas, estados = [], [], []
    for carpeta in ("xgboost", "xgboost_15", "xgboost_15_refinado"):
        detalle = leer_json(raiz / "results" / carpeta / "detalle.json")
        resumen = leer_json(raiz / "results" / carpeta / "resumen.json")
        estudio = resumen["estudio"]
        estados.append({"estudio": estudio, "total": len(detalle["trials"]),
                        "complete": sum(t["estado"] == "COMPLETE" for t in detalle["trials"]),
                        "pruned": sum(t["estado"] == "PRUNED" for t in detalle["trials"])})
        for trial in detalle["trials"]:
            if trial["estado"] != "COMPLETE":
                continue
            marca = f"Optuna {estudio}; trial={trial['trial']}; CV temporal 5 pliegues; solo entrenamiento"
            coincidencias = propios.loc[propios["notas"].eq(marca) & propios["tramo"].eq("entrenamiento")]
            if len(coincidencias) != 1:
                raise ValueError(f"Registro ausente o duplicado: {marca}")
            pred = trial["predicciones"]
            calculadas = metricas_guardadas(pred["real"], pred["predicho"])
            comprobar_metricas(calculadas, trial["metricas"])
            parametros = {**detalle["configuracion"]["configuracion"]["fijos"], **trial["parametros"]}
            registro = coincidencias.iloc[0]
            _validar_registro(registro, parametros, calculadas, len(pred["real"]))
            usados.append(coincidencias.index[0])
            filas.append({"estudio": estudio, "trial": trial["trial"], "tramo": "entrenamiento",
                          "ajustes_agrupados": len(trial["pliegues"]),
                          "fecha_hora_original": registro["fecha_hora"], "commit": registro["commit"],
                          "filas": len(pred["real"]), **calculadas})
    for _, (_, carpeta) in EXPERIMENTOS.items():
        evaluacion = leer_json(raiz / "results" / carpeta / "evaluacion.json")
        registro = evaluacion["registro"]
        coincidencias = propios.loc[propios["notas"].eq(registro["notas"]) & propios["tramo"].eq("validacion")]
        if len(coincidencias) != 1:
            raise ValueError("Registro de validación ausente o duplicado.")
        for clave in ("fecha_hora", "commit", "hiperparametros"):
            if coincidencias.iloc[0][clave] != registro[clave]:
                raise ValueError(f"Difiere el registro histórico: {clave}")
        _validar_registro(coincidencias.iloc[0], json.loads(registro["hiperparametros"]), registro, registro["filas"])
        usados.append(coincidencias.index[0])
        filas.append({"estudio": evaluacion["estudio"], "trial": evaluacion["trial"], "tramo": "validacion",
                      "ajustes_agrupados": 1, "fecha_hora_original": registro["fecha_hora"],
                      "commit": registro["commit"], "filas": registro["filas"],
                      **{k: registro[k] for k in METRICAS}})
    if len(set(usados)) != len(usados) or set(usados) != set(propios.index):
        raise ValueError("Hay registros XGBoost sin correspondencia o reutilizados.")
    resumen = {"registros_xgboost": len(propios), "cv_completos": len(filas) - 2,
               "validaciones": 2, "campos_y_metricas_verificados": True,
               "estudios": estados, "log_original_modificado": False,
               "cumplimiento_literal_por_ajuste": "Parcial",
               "limite": "El CSV agrupa cinco ajustes por COMPLETE; los PRUNED quedan en Optuna/JSON. "
                         "El timestamp es de registro, sin zona horaria, no el inicio del ajuste. "
                         "No se inventan filas ni fechas retrospectivas."}
    return pd.DataFrame(filas), resumen


def graficar(tablas, salida):
    """Genera cinco diagnósticos comparables desde los vectores archivados."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etiquetas = {"inicial_15": "Inicial · trial 56", "refinado_15": "Refinado · trial 49"}
    especificaciones = [
        ("prediccion_vs_real", "Predicciones frente a valores reales"),
        ("predicciones_temporales", "Valores reales y predicciones en validación"),
        ("distribucion_residuos", "Distribución de residuos"),
        ("residuos_vs_predicciones", "Residuos frente a predicciones"),
        ("residuos_temporales", "Evolución de los residuos"),
    ]
    todos = pd.concat(tablas.values())
    extremos = [todos[["real", "predicho"]].min().min(), todos[["real", "predicho"]].max().max()]
    bordes = np.histogram_bin_edges(todos["residuo"], bins=18)
    limite = todos["residuo"].abs().max() * 1.08
    rutas = []
    for nombre, titulo in especificaciones:
        fig, ejes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
        for eje, (clave, tabla) in zip(ejes, tablas.items()):
            eje.set_title(etiquetas[clave], loc="left")
            if nombre == "prediccion_vs_real":
                eje.scatter(tabla["real"], tabla["predicho"], s=20, alpha=.7)
                eje.plot(extremos, extremos, color="black", linestyle="--", label="Predicción exacta")
                eje.set(xlabel="Real (kW)", ylabel="Predicho (kW)", xlim=extremos, ylim=extremos)
                eje.legend()
            elif nombre == "predicciones_temporales":
                eje.plot(tabla["DIA"], tabla["real"], label="Real", linewidth=1.1)
                eje.plot(tabla["DIA"], tabla["predicho"], label="Predicho", linewidth=1.1)
                eje.set(xlabel="Fecha rotulada de emisión D (objetivo: bloque siguiente)", ylabel="Consumo (kW)")
                eje.legend()
            elif nombre == "distribucion_residuos":
                eje.hist(tabla["residuo"], bins=bordes, color="#2878a6", edgecolor="white")
                eje.axvline(0, color="black", linestyle="--", label="Cero")
                eje.axvline(tabla["residuo"].mean(), color="#bd541a", label="Media")
                eje.set(xlabel="Residuo = real − predicho (kW)", ylabel="Filas")
                eje.legend()
            else:
                x = tabla["predicho"] if nombre == "residuos_vs_predicciones" else tabla["DIA"]
                eje.scatter(x, tabla["residuo"], s=20, alpha=.7)
                eje.axhline(0, color="black", linestyle="--")
                eje.set(ylabel="Real − predicho (kW)", ylim=(-limite, limite),
                        xlabel="Predicho (kW)" if nombre == "residuos_vs_predicciones"
                        else "Fecha rotulada de emisión D (objetivo: bloque siguiente)")
            eje.grid(alpha=.2)
        fig.suptitle(f"XGBoost · 171 filas de validación · {titulo}", fontsize=14)
        ruta = salida / f"{nombre}.png"
        fig.savefig(ruta, dpi=160, facecolor="white")
        plt.close(fig)
        rutas.append(ruta.name)
    return rutas


def escribir_informe(salida, resumen, meses, mayores, comparacion):
    """Escribe conclusiones descriptivas con cifras de las tablas derivadas."""
    inicial, refinado = resumen.iloc[0], resumen.iloc[1]
    enero = meses.loc[(meses.experimento == "inicial_15") & (meses.mes_emision == "2023-01")].iloc[0]
    abril = meses.loc[(meses.experimento == "inicial_15") & (meses.mes_emision == "2023-04")].iloc[0]
    mayor = mayores.loc[mayores.experimento == "inicial_15"].iloc[0]
    media = comparacion.loc[comparacion.experimento == "media_movil", "mae"].iloc[0]
    texto = f"""# Diagnóstico archivado de XGBoost — Fase 3

Fecha del análisis: 15/11/2026. Fuentes: predicciones y evaluaciones de
`results/xgboost_15_validacion/` y `results/xgboost_15_refinado_validacion/`,
estudios exportados de `results/xgboost*/` y `results/experiment_logs.csv`.
No se abrió el dataset ni el conjunto de prueba, no se cargó un estimador y no
se ejecutaron entrenamiento, predicción, optimización ni una nueva evaluación.
Los SHA-256 de las fuentes están en `verificacion.json` y permanecen intactos.

## 1. Convención y comparación

Se conservan las mismas 171 filas e índices CSV de validación, con fecha rotulada
de emisión entre 29/10/2022 y 26/04/2023. DIA identifica el bloque de emisión;
el objetivo corresponde al bloque siguiente, no necesariamente a DIA + 1 físico
debido al desfase del calendario documentado. Las fechas pueden repetirse;
la identidad y el orden se comprueban por indice_csv. No se interpolan las filas
ausentes. La línea temporal une solo observaciones disponibles y no implica
que se haya evaluado cada día intermedio.

El residuo es real menos predicho: positivo significa subestimación y negativo,
sobreestimación. Se presentan tanto el experimento inicial (trial 56) como el
refinado (trial 49), sin elegir al ganador global del trabajo.

El MAE registrado inicial es {inicial.mae_registrado:.2f} kW y el refinado,
{refinado.mae_registrado:.2f} kW. El refinamiento reduce el MAE de validación
cruzada en {inicial.mae_cv - refinado.mae_cv:.2f} kW, pero aumenta el MAE de
validación en {refinado.mae_registrado - inicial.mae_registrado:.2f} kW.
La mejora del inicial frente a la media móvil de {media:.2f} kW es
{100 * (media - inicial.mae_registrado) / media:.2f} % en estas filas de validación;
no es un resultado de prueba ni satisface por sí sola la condición de aprobación.
Las tablas completas están en `comparacion_validacion.csv`, `hiperparametros.csv`
y `espacios_busqueda.csv`; incluyen parámetros fijos y optimizados de ambos ganadores.

## 2. Lectura de las predicciones y los residuos

El gráfico `prediccion_vs_real.png` muestra asociación con el consumo observado,
pero también dispersión alrededor de la diagonal. Las predicciones suavizan
algunos cambios bruscos, como se aprecia en `predicciones_temporales.png`.
No hay un ajuste nuevo de una recta ni calibración de las predicciones.

La distribución (`distribucion_residuos.png`) está cerca de cero en promedio:
sesgo inicial {inicial.sesgo_real_menos_predicho:.2f} kW y refinado
{refinado.sesgo_real_menos_predicho:.2f} kW. Eso no implica errores pequeños:
el desvío inicial es {inicial.desvio_residuo:.2f} kW y el percentil 95 del error
absoluto es {inicial.p95_error_absoluto:.2f} kW. El inicial subestima en
{int(inicial.subestimaciones)} filas y sobreestima en {int(inicial.sobreestimaciones)};
el refinado lo hace en {int(refinado.subestimaciones)} y {int(refinado.sobreestimaciones)}.
No se afirma normalidad ni independencia de los residuos.

En `residuos_vs_predicciones.png` los errores no forman una nube completamente
homogénea: visualmente hay predominio de errores negativos en la zona de
predicciones altas. Es un diagnóstico descriptivo, sin una prueba formal de
heterocedasticidad ni una inferencia causal.

El sesgo global oculta diferencias temporales (`residuos_temporales.png`):
en enero de 2023 el inicial subestima {enero.sesgo:.2f} kW en promedio sobre
{int(enero.filas)} filas, mientras que en abril sobreestima {abs(abril.sesgo):.2f} kW
sobre {int(abril.filas)}. Octubre tiene solo dos filas y no permite generalizar
una comparación mensual. `residuos_por_mes.csv` conserva tamaños, MAE y sesgos.

El mayor error inicial es una sobreestimación de {mayor.error_absoluto:.2f} kW
en el índice CSV {int(mayor.indice_csv)}, fecha de emisión {str(mayor.DIA)[:10]}:
real {mayor.real:.2f} y predicho {mayor.predicho:.2f} kW. Los diez mayores errores
concentran {100 * inicial.fraccion_error_top10:.2f} % del error absoluto total.
Se conservan las diez filas de cada experimento en `errores_mayores.csv`, sin
eliminarlas ni atribuirles fallas de planta que las predicciones no pueden probar.

## 3. Optimización, tracking y precisión

La búsqueda inicial vigente conserva 100 trials (47 completos, 53 podados);
el refinado, 60 (45 completos, 15 podados). Ambos usan TPE con semilla 42,
MedianPruner y cinco pliegues temporales con gap=1 sobre entrenamiento.
El refinado se definió con los diez mejores trials anteriores: amplió árboles
a 1750–3000, mantuvo profundidad 3 y tasa 0.0020702351473759497, y acotó peso
de hijos y fracciones de filas/columnas. Se minimizó el MAE conjunto de las
filas comunes, no el promedio no ponderado por pliegue. Ver ADR 0019.

La auditoría contrasta 137 COMPLETE (incluidos 45 históricos de 14 variables)
y dos validaciones con 139 filas propias del log. Compara parámetros completos,
métricas, filas, semilla, commit y timestamps ISO originales; no faltan ni se
duplican esos registros. `auditoria_tracking.csv` es evidencia derivada, no un
reemplazo ni una ampliación retrospectiva de `experiment_logs.csv`.

La sección 3.2 queda parcialmente cubierta bajo su lectura literal: el log guarda
un experimento por COMPLETE, agrupando cinco ajustes; los podados quedan en
Optuna/JSON, sin fila en el CSV. Los timestamps son de registro, no de inicio del
ajuste, y no llevan zona horaria. No hay evidencia para inventar tiempos por
pliegue ni métricas finales de podados. Resolver el contrato futuro corresponde
a la integración, sin modificar las ejecuciones históricas.

Los CSV contienen representaciones decimales de predicciones float32. Las
métricas calculadas sobre esos decimales difieren mínimamente del log; recuperar
float32 reproduce las tres métricas históricas con tolerancia estricta. El cambio
máximo de predicción es {max(inicial.max_delta_prediccion_float32, refinado.max_delta_prediccion_float32):.8f} kW.
Los residuos usan los decimales originales; el resumen separa las métricas del CSV
de las columnas terminadas en `_registrado`. No se corrige ningún resultado previo.

## 4. Reproducción y entrega

Con Python 3.12 y las dependencias congeladas, desde la raíz del repositorio:

```powershell
python -B src/diagnostico_xgboost.py
python -B -m pytest tests/test_diagnostico_xgboost.py -q -p no:cacheprovider
```

El primer comando solo deriva CSV, PNG y verificaciones dentro de esta carpeta.
No importa XGBoost, LightGBM, scikit-learn, Optuna ni joblib. El segundo archivo
de pruebas se limita a aritmética, validaciones de evidencia y controles estáticos.
No se autoriza la suite completa, que contiene ajustes de estimadores.
La sección 13 en adelante de `notebooks/modelado_xgboost.ipynb` es autónoma y
consulta este diagnóstico; las primeras secciones son históricas y no deben
ejecutarse para esta revisión. Las dependencias de estado de las celdas históricas
no se certifican como reproducción integral del notebook desde un kernel vacío.

## 5. Alcance y pendientes

Quedan completos para estos candidatos XGBoost los gráficos de predicción frente
a real y el análisis descriptivo de residuos de 3.3, con salidas ejecutadas.
Los cinco PNG y las tablas quedan preparados para integrar al notebook común,
que no se modifica. Se preservan las 15 variables y todos los resultados previos.

No hay un modelo XGBoost serializado del que extraer importancia de variables.
La f-ANOVA existente trata hiperparámetros y no cubre ese requisito. La consigna
pide importancia de variables del mejor modelo global; queda pendiente de su
elección y de un artefacto real. No se reconstruye un estimador para obtenerla.
También quedan para la integración: comparación de todas las familias, selección
por MAE de validación, entrenamiento final con train+validation, versión semántica,
registry, etiqueta Git y punto de entrada común. Todos los pasos que requieren
ejecutar modelos están fuera del alcance autorizado.

La selección previa de variables no está anidada y validación ya fue consultada
en decisiones anteriores. Estos gráficos añaden inspección del mismo período;
no son una nueva estimación independiente ni justifican ajustar hiperparámetros.
Los cambios de sesgo por mes no demuestran causa operativa ni generalización.
"""
    (salida / "informe.md").write_text(texto, encoding="utf-8")


def generar_diagnostico(raiz=RAIZ):
    """Escribe solo resultados derivados en una carpeta propia, preservando fuentes."""
    raiz = Path(raiz).resolve()
    salida = raiz / CARPETA_SALIDA
    fuentes = [raiz / "results/experiment_logs.csv"]
    for carpeta in ("xgboost", "xgboost_15", "xgboost_15_refinado"):
        fuentes.extend(p for p in (raiz / "results" / carpeta).rglob("*") if p.is_file())
    for _, carpeta in EXPERIMENTOS.values():
        fuentes.extend(p for p in (raiz / "results" / carpeta).rglob("*") if p.is_file())
    originales = {str(p.relative_to(raiz)).replace("\\", "/"): huella(p) for p in fuentes}
    auditoria, tracking = auditar_tracking(raiz)
    tablas, resumenes, meses, mayores, parametros, rangos = {}, [], [], [], [], []
    referencia = None
    for nombre, (carpeta, carpeta_val) in EXPERIMENTOS.items():
        detalle = leer_json(raiz / "results" / carpeta / "detalle.json")
        contrato = detalle["configuracion"]["configuracion"]
        resumen = leer_json(raiz / "results" / carpeta / "resumen.json")
        evaluacion = leer_json(raiz / "results" / carpeta_val / "evaluacion.json")
        if len(contrato["predictoras"]) != 15 or contrato["pliegues"] != 5 or contrato["gap"] != 1:
            raise ValueError("Contrato de 15 variables y cinco pliegues temporales incompatible.")
        tabla = preparar_residuos(pd.read_csv(raiz / "results" / carpeta_val / "predicciones.csv"))
        if len(tabla) != 171 or evaluacion["registro"]["tramo"] != "validacion":
            raise ValueError("Solo se admiten las 171 predicciones archivadas de validación.")
        claves = tabla[["indice_csv", "DIA", "real"]]
        if referencia is not None:
            pd.testing.assert_frame_equal(referencia, claves)
        referencia = claves
        calculadas = resumir_residuos(tabla)
        # El CSV decimal proviene de predicciones float32. Recuperar ese dtype
        # reproduce el log, sin cambiar los decimales usados en los residuos.
        recuperadas = metricas_guardadas(tabla["real"], tabla["predicho"].to_numpy(dtype=np.float32).astype(float))
        comprobar_metricas(recuperadas, evaluacion["registro"])
        calculadas.update({f"{k}_registrado": evaluacion["registro"][k] for k in METRICAS})
        calculadas["max_delta_prediccion_float32"] = float(np.max(np.abs(
            tabla["predicho"].to_numpy() - tabla["predicho"].to_numpy(dtype=np.float32).astype(float))))
        tablas[nombre] = tabla
        resumenes.append({"experimento": nombre, "trial": evaluacion["trial"],
                          "mae_cv": resumen["metricas_provisionales"]["mae"], **calculadas})
        for mes, parte in tabla.groupby(tabla["DIA"].dt.strftime("%Y-%m"), sort=True):
            meses.append({"experimento": nombre, "mes_emision": mes, "filas": len(parte),
                          "sesgo": float(parte["residuo"].mean()),
                          "mae": float(parte["error_absoluto"].mean())})
        mayores.append(tabla.nlargest(10, "error_absoluto").assign(experimento=nombre))
        parametros.extend({"experimento": nombre, "hiperparametro": k, "valor": v}
                          for k, v in json.loads(evaluacion["registro"]["hiperparametros"]).items())
        rangos.extend({"experimento": nombre, "hiperparametro": k, **v}
                      for k, v in contrato["espacio"].items())
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in tablas.items():
        tabla.to_csv(salida / f"residuos_{nombre}.csv", index=False, lineterminator="\n")
    pd.DataFrame(resumenes).to_csv(salida / "resumen_residuos.csv", index=False, lineterminator="\n")
    pd.DataFrame(meses).to_csv(salida / "residuos_por_mes.csv", index=False, lineterminator="\n")
    pd.concat(mayores).to_csv(salida / "errores_mayores.csv", index=False, lineterminator="\n")
    pd.DataFrame(parametros).to_csv(salida / "hiperparametros.csv", index=False, lineterminator="\n")
    pd.DataFrame(rangos).to_csv(salida / "espacios_busqueda.csv", index=False, lineterminator="\n")
    auditoria.to_csv(salida / "auditoria_tracking.csv", index=False, lineterminator="\n")
    log = pd.read_csv(raiz / "results/experiment_logs.csv")
    bases = log.loc[log.modelo.isin(["persistencia", "media_movil"]) & log.tramo.eq("validacion")]
    if len(bases) != 2 or not bases.filas.eq(171).all():
        raise ValueError("Baselines de validación ambiguos.")
    comparacion = pd.DataFrame([
        {"experimento": s["experimento"], "filas": s["filas"],
         **{k: s[f"{k}_registrado"] for k in METRICAS}} for s in resumenes
    ] + [{"experimento": r.modelo, "filas": int(r.filas), **{k: float(r[k]) for k in METRICAS}}
         for _, r in bases.iterrows()])
    comparacion.to_csv(salida / "comparacion_validacion.csv", index=False, lineterminator="\n")
    escribir_informe(salida, pd.DataFrame(resumenes), pd.DataFrame(meses), pd.concat(mayores), comparacion)
    figuras = graficar(tablas, salida)
    controles = {"metricas_float32_coinciden_con_evaluaciones": True, "mismas_171_filas_y_objetivos": True,
                 "tracking_complete_y_validaciones_coincide": True,
                 "originales_intactos": all(huella(raiz / p) == h for p, h in originales.items()),
                 "sin_importar_bibliotecas_de_modelado": True,
                 "sin_lectura_de_dataset_ni_test": True}
    if not all(controles.values()):
        raise ValueError("Se modificó una fuente durante el diagnóstico.")
    manifiesto = {"fuentes_sha256": originales, "figuras": figuras, "controles": controles,
                  "tracking": tracking, "python": platform.python_version(),
                  "versiones": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib")},
                  "fecha_diagnostico": "2026-11-15",
                  "nota_fecha": "Fecha de este análisis; no reemplaza timestamps históricos de entrenamiento.",
                  "alcance": "Solo aritmética y gráficos desde resultados archivados. Sin modelos ni test."}
    (salida / "verificacion.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifiesto


def main():
    """Ejecuta únicamente el diagnóstico offline autorizado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    resultado = generar_diagnostico()
    print(json.dumps({"carpeta": CARPETA_SALIDA, "controles": resultado["controles"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
