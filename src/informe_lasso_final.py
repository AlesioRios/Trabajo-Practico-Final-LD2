"""Informe y figuras desde evidencia archivada: no ajusta ni predice modelos."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]


def generar_informe(raiz=RAIZ):
    """Deriva diagnóstico descriptivo exclusivamente de CSV y JSON guardados."""
    salida = raiz / "results/lasso_final"
    meta = json.loads((salida / "entrenamiento.json").read_text(encoding="utf-8"))
    resultado = json.loads((salida / "evaluacion_test.json").read_text(encoding="utf-8"))
    tabla = pd.read_csv(salida / "predicciones_test.csv", parse_dates=["DIA"])
    coef = pd.read_csv(salida / "coeficientes.csv")
    metricas = pd.DataFrame(resultado["metricas"]).T.rename_axis("modelo")
    metricas.to_csv(salida / "comparacion_test.csv")
    tabla.nlargest(15, "error_absoluto").to_csv(salida / "errores_mayores.csv", index=False)
    grupos = pd.qcut(tabla.real, 4, duplicates="drop")
    sesgos = tabla.groupby(grupos, observed=True).agg(filas=("real", "size"), real_medio=("real", "mean"),
        residuo_medio=("residuo", "mean"), mae=("error_absoluto", "mean"))
    sesgos.to_csv(salida / "residuos_por_cuartil.csv")
    resumen = {"residuo_medio": float(tabla.residuo.mean()), "residuo_mediano": float(tabla.residuo.median()),
        "desvio_residuo": float(tabla.residuo.std()), "error_absoluto_maximo": float(tabla.error_absoluto.max()),
        "p90_error_absoluto": float(tabla.error_absoluto.quantile(.9)),
        "coeficientes_no_nulos": int(coef.coeficiente_estandarizado.ne(0).sum()),
        "residuos_positivos": int(tabla.residuo.gt(0).sum()), "residuos_negativos": int(tabla.residuo.lt(0).sum())}
    (salida / "resumen_residuos.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.scatter(tabla.real, tabla.predicho, alpha=.65, s=22)
    limites = [min(tabla.real.min(), tabla.predicho.min()), max(tabla.real.max(), tabla.predicho.max())]
    ax.plot(limites, limites, "--", color="black", linewidth=1)
    ax.set(xlabel="Frío real (kW)", ylabel="Predicción Lasso (kW)", title="Test local · predicción frente a valor real")
    fig.savefig(salida / "prediccion_vs_real.png", dpi=160); plt.close(fig)
    fig, ejes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained", sharex=True)
    ejes[0].plot(tabla.DIA, tabla.real, label="Real", linewidth=1.2)
    ejes[0].plot(tabla.DIA, tabla.predicho, label="Lasso", linewidth=1.1)
    ejes[0].set(ylabel="Frío (kW)", title="Test local · valores y residuos por fecha"); ejes[0].legend()
    ejes[1].plot(tabla.DIA, tabla.residuo, linewidth=.9); ejes[1].axhline(0, color="black", linestyle="--")
    ejes[1].set(ylabel="Real − predicho (kW)", xlabel="DIA (identidad completa en CSV)")
    fig.savefig(salida / "predicciones_y_residuos_temporales.png", dpi=160); plt.close(fig)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    ejes[0].hist(tabla.residuo, bins=20, edgecolor="white"); ejes[0].axvline(0, color="black", linestyle="--")
    ejes[0].set(xlabel="Real − predicho (kW)", ylabel="Observaciones", title="Distribución de residuos")
    ejes[1].scatter(tabla.predicho, tabla.residuo, alpha=.65, s=20); ejes[1].axhline(0, color="black", linestyle="--")
    ejes[1].set(xlabel="Predicción (kW)", ylabel="Real − predicho (kW)", title="Sesgo y dispersión")
    fig.savefig(salida / "diagnostico_residuos.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 8), layout="constrained")
    colores = ["#177a78" if c >= 0 else "#b24c48" for c in coef.coeficiente_estandarizado]
    ax.barh(coef.variable, coef.coeficiente_estandarizado, color=colores); ax.invert_yaxis(); ax.axvline(0,color="black",linewidth=.8)
    ax.set(xlabel="Coeficiente (kW por una desviación estándar del predictor)", title="Lasso final · coeficientes estandarizados")
    fig.savefig(salida / "coeficientes.png", dpi=160); plt.close(fig)
    mejor = min(["persistencia", "media_movil"], key=lambda b: resultado["metricas"][b]["mae"])
    mejora = resultado["comparaciones"][mejor]["mejora_porcentual"]
    texto_complejidad = ("La mejora es menor al 10 %. No justificaría desplegar este pipeline por precisión únicamente: "
        "se compra una relación lineal interpretable y reutilización de la preparación, pero se mantienen imputación, "
        "escalado y dependencias. El baseline sigue siendo una referencia operativa de bajo costo. "
        "Se conserva Lasso por la selección ya fijada; no se lo reemplaza ni ajusta después de observar test."
        if mejora < 10 else "La mejora supera el 10 % frente al mejor baseline local. La ganancia medida aporta evidencia "
        "a favor del costo del pipeline, aunque no garantiza rendimiento futuro ni en el test oculto.")
    tabla_md = "| Modelo | MAE test | RMSE test | R² test |\n|---|---:|---:|---:|\n" + "\n".join(
        f"| {m} | {v['mae']:.12f} | {v['rmse']:.12f} | {v['r2']:.12f} |" for m,v in resultado["metricas"].items())
    mejores_coef = "\n".join(f"- `{r.variable}`: {r.coeficiente_estandarizado:+.9f} kW por desviación estándar."
        for r in coef.head(5).itertuples())
    peores = tabla.nlargest(5,"error_absoluto")
    peores_md = "| Índice CSV | DIA | Real | Predicho | Residuo |\n|---:|---|---:|---:|---:|\n" + "\n".join(
        f"| {r.indice_csv} | {r.DIA.date()} | {r.real:.3f} | {r.predicho:.3f} | {r.residuo:+.3f} |" for r in peores.itertuples())
    texto = f'''# Cierre de Fase 3 — Lasso v{meta['version']}

## Selección y entrenamiento

Se mantuvo Lasso con las 14 entradas históricas y pipeline A. MAE de validación original: {meta['metricas_validacion_original']['mae']}; RMSE: {meta['metricas_validacion_original']['rmse']}; R²: {meta['metricas_validacion_original']['r2']}. Es evidencia de selección, no una nueva medición del modelo reentrenado. El ranking histórico permanece en [comparacion_modelos_final.md](../comparacion_modelos_final.md).

Se reajustaron conjuntamente mediana, StandardScaler y Lasso con {meta['filas_train_validation']} filas etiquetadas ({meta['filas_por_tramo']['entrenamiento']} train y {meta['filas_por_tramo']['validacion']} validation). Se conservaron las etiquetas de frontera eliminadas por Fase 2; no se reconstruyó ninguna. Parámetros: `{meta['hiperparametros']}`. Commit del código entrenado: `{meta['commit']}`. Python {meta['python']}; semilla 42. Dos ajustes independientes con esos datos dieron iguales coeficientes y predicciones antes de test. La carga joblib reprodujo exactamente las predicciones de ajuste.

Artefacto: `{meta['path']}`; SHA-256 `{meta['modelo_sha256']}`. El artefacto incluye orden de variables, imputador, escalador y estimador. [entrenamiento.json](entrenamiento.json) conserva parámetros, índices y huellas. [coeficientes.csv](coeficientes.csv) enumera las 14 entradas, sus coeficientes y escalas. El registry separa validación original y test local.

## Evaluación local única

No se encontró evaluación final previa en los registros y ramas inspeccionados. Se creó test_consumido.json antes de cargar las observaciones de prueba. Solo hubo una llamada de predicción sobre test del artefacto congelado. Las consultas y el notebook leen evidencia archivada.

Test contiene {resultado['filas_test_etiquetadas']} filas etiquetadas; se evaluaron exactamente **{resultado['filas_evaluadas']}** con objetivo y ambos baselines disponibles. Se excluyeron los índices {resultado['indices_excluidos']}. Modelo, persistencia y media móvil comparten las mismas filas, target y fechas. No se asumió el tamaño de validación. La ventana del baseline es de siete bloques, incluido D, con mínimo cuatro valores válidos, según la política existente.

{tabla_md}

Frente a persistencia: reducción absoluta de MAE {resultado['comparaciones']['persistencia']['diferencia_mae']:.12f} kW y mejora {resultado['comparaciones']['persistencia']['mejora_porcentual']:.9f} %. Frente a media móvil: {resultado['comparaciones']['media_movil']['diferencia_mae']:.12f} kW y {resultado['comparaciones']['media_movil']['mejora_porcentual']:.9f} %. Los signos negativos indican empeoramiento. ¿Supera ambos baselines?: **{resultado['supera_ambos_baselines']}**. ¿MAE local menor a 3000?: **{resultado['mae_local_menor_3000']}**. El umbral eliminatorio corresponde al test oculto de la cátedra; este resultado local no lo acredita.

{texto_complejidad}

## Interpretación de coeficientes

Magnitudes mayores del modelo reajustado:

{mejores_coef}

Hay {resumen['coeficientes_no_nulos']} coeficientes no nulos de 14. Cada coeficiente expresa el cambio de predicción por una desviación estándar del predictor después de imputar, manteniendo los otros fijos. El signo positivo aumenta y el negativo disminuye la predicción. Productos y variables originales comparten información; correlación y regularización redistribuyen sus coeficientes. Magnitud no equivale a contribución total, efecto causal ni importancia estable fuera de muestra.

![Coeficientes](coeficientes.png)

## Residuos y errores grandes

Residuo = real − predicho. Media: {resumen['residuo_medio']:+.9f} kW; mediana: {resumen['residuo_mediano']:+.9f}; desvío muestral: {resumen['desvio_residuo']:.9f}; percentil 90 de error absoluto: {resumen['p90_error_absoluto']:.9f}. Hubo {resumen['residuos_positivos']} subestimaciones y {resumen['residuos_negativos']} sobreestimaciones. El signo de la media muestra la dirección del sesgo agregado, sin probar significación estadística. [residuos_por_cuartil.csv](residuos_por_cuartil.csv) permite contrastar ese sesgo por nivel real de consumo.

{peores_md}

![Predicción frente a real](prediccion_vs_real.png)

![Residuos](diagnostico_residuos.png)

![Evolución temporal](predicciones_y_residuos_temporales.png)

Los errores extremos y la dispersión se reportan como diagnóstico posterior; no se eliminan observaciones ni se cambian parámetros a partir de ellos. [predicciones_test.csv](predicciones_test.csv) conserva valores, baselines, residuos e identidad física para auditar sin una nueva predicción.

## Limitaciones y conclusión técnica

La selección reutilizó validation, que ya había intervenido en la elección de variables; su MAE puede ser optimista. El contrato operativo vigente tiene quince entradas, pero este modelo usa explícitamente las catorce ganadoras: es_fin_de_anio no entra. La etiqueta utiliza consumo del bloque siguiente. DIA no siempre es único y no está demostrada la correspondencia entre bloque y fecha física ni la disponibilidad real de todas las entradas al cierre D. Se comprobó separación por índices y orden temporal y ausencia de test en los ajustes; esto no certifica ausencia absoluta de fuga operativa. Los baselines usan la historia disponible hasta el bloque D y no el valor objetivo futuro.

El resultado de test no se utilizó para elegir otro algoritmo ni ajustar Lasso. La conclusión local es que la mejora frente al mejor baseline ({mejor}) es {mejora:.9f} %. El modelo queda congelado con su resultado real, favorable o desfavorable. La Fase 4 y la evaluación del test oculto no se implementan en esta tarea. Los controles técnicos y el resultado completo de pytest se consignan en verificacion_final.json; un fallo esencial impide afirmar que el cierre técnico es correcto.
'''
    (salida / "informe_final.md").write_text(texto, encoding="utf-8")
    return resumen


if __name__ == "__main__":
    print(json.dumps(generar_informe(), ensure_ascii=True, indent=2))
