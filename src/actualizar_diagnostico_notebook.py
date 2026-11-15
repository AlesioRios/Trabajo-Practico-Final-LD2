"""Ejecuta solo el apéndice de diagnóstico, preservando las celdas históricas.

No ejecuta el notebook completo. Requiere los artefactos ya generados por
diagnostico_xgboost.py y el kernel cervecera_env (Python 3.12).
"""

import copy
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]


def celdas_diagnostico():
    """Construye una sección autónoma que solo lee CSV/JSON/PNG archivados."""
    celdas = []

    def texto(contenido):
        celdas.append(nbformat.v4.new_markdown_cell(contenido, metadata={"diagnostico_offline": True}))

    def codigo(contenido):
        celdas.append(nbformat.v4.new_code_cell(contenido, metadata={"diagnostico_offline": True}))

    texto("""## 13. Diagnóstico de validación desde predicciones archivadas

Esta sección es autónoma: se ejecutó únicamente este apéndice con cervecera_env,
sin ejecutar las celdas históricas anteriores. Lee resultados de
results/xgboost_diagnostico_validacion, sin abrir el dataset ni importar estimadores.
La sección 3.3 de la consigna pide comparar validación y analizar predicciones y
residuos; la importancia de variables se reserva al mejor modelo global.
Se mantienen las quince variables oficiales y los resultados históricos intactos.

Primero se recuperan las métricas registradas y los parámetros completos. La
validación de ambas búsquedas contiene las mismas filas, y los baselines se
comparan sobre ese mismo conjunto. La tabla de residuos usa real menos predicho:
valores positivos son subestimaciones y negativos, sobreestimaciones.""")
    codigo('''from pathlib import Path
import json
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

raiz_diagnostico = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(raiz_diagnostico / "src"))
from diagnostico_xgboost import huella

carpeta_diagnostico = raiz_diagnostico / "results/xgboost_diagnostico_validacion"
verificacion_diagnostico = json.loads((carpeta_diagnostico / "verificacion.json").read_text(encoding="utf-8"))
resumen_diagnostico = pd.read_csv(carpeta_diagnostico / "resumen_residuos.csv")
print("Tabla 22. Métricas registradas de validación y baselines sobre las mismas filas")
display(pd.read_csv(carpeta_diagnostico / "comparacion_validacion.csv").round(4))
print("Tabla 23. Parámetros completos de los ganadores de ambas búsquedas")
display(pd.read_csv(carpeta_diagnostico / "hiperparametros.csv").pivot(index="hiperparametro", columns="experimento", values="valor"))
print("Tabla 24. Espacios de búsqueda conservados en los contratos originales")
display(pd.read_csv(carpeta_diagnostico / "espacios_busqueda.csv"))''')
    texto("""## 14. Predicciones y distribución de residuos

La comparación de predicción frente a real incluye la diagonal de predicción
exacta. La serie temporal conserva las fechas rotuladas de emisión D, cuyo objetivo
es el bloque siguiente. Las líneas unen solo filas disponibles y no interpolan los
días ausentes; las fechas no se utilizan como clave de unión. Se muestran ambos
experimentos con escalas comparables, sin recalibrar ni alterar las predicciones.""")
    for nombre, titulo in [("prediccion_vs_real", "Predicción frente a real"),
                          ("predicciones_temporales", "Real y predicción por fecha de emisión")]:
        codigo(f'print("Figura. {titulo}")\ndisplay(Image(filename=str(carpeta_diagnostico / "{nombre}.png")))')
        texto("La figura siguiente complementa la comparación con el orden temporal y conserva todas las observaciones disponibles."
              if nombre == "prediccion_vs_real" else
              "El histograma siguiente muestra el signo y la magnitud de los errores, con referencias en cero y en su media. La forma visual no constituye una prueba de normalidad.")
    codigo('print("Figura. Distribución de residuos")\ndisplay(Image(filename=str(carpeta_diagnostico / "distribucion_residuos.png")))')
    texto("La tabla siguiente distingue el sesgo promedio de la dispersión y las colas del error. Las métricas terminadas en registrado provienen del registro histórico; las demás se calculan sobre los decimales del CSV. La recuperación de float32 verifica la coincidencia con el log sin cambiar los CSV originales.")
    codigo('print("Tabla 25. Resumen descriptivo de residuos y conciliación de precisión")\ndisplay(resumen_diagnostico.round(5))')
    texto("""## 15. Residuos frente a predicciones, evolución y errores mayores

Los residuos frente a predicciones permiten examinar si el error cambia según el
nivel estimado. La evolución temporal y las tablas mensuales muestran patrones
que pueden cancelarse en el sesgo global. Se informa la cantidad de filas por mes
para no dar el mismo peso interpretativo a meses completos y parciales.""")
    codigo('print("Figura. Residuos frente a predicciones")\ndisplay(Image(filename=str(carpeta_diagnostico / "residuos_vs_predicciones.png")))')
    texto("La figura temporal conserva la misma convención de signo y las mismas escalas para ambas búsquedas. No se atribuye un patrón a una causa operativa sin evidencia adicional.")
    codigo('print("Figura. Evolución temporal de residuos")\ndisplay(Image(filename=str(carpeta_diagnostico / "residuos_temporales.png")))')
    texto("Las tablas muestran sesgos por mes y los mayores errores individuales. Los casos extremos se conservan: no se eliminan filas después de conocer su error y no se modifican hiperparámetros a partir de esta inspección.")
    codigo('''print("Tabla 26. Sesgo y MAE por mes de emisión, con tamaño de muestra")
display(pd.read_csv(carpeta_diagnostico / "residuos_por_mes.csv").round(3))
print("Tabla 27. Diez errores mayores de cada experimento, identificados por índice CSV")
display(pd.read_csv(carpeta_diagnostico / "errores_mayores.csv").round(3))''')
    texto("""## 16. Tracking, preservación y conclusiones

La auditoría compara los trials completos archivados y las validaciones con el
log original, sin agregar registros de entrenamiento. Un registro agrupa los
cinco ajustes de un trial completo; los podados quedan en Optuna y JSON. Esta
granularidad deja parcialmente cubierta la lectura literal de cada entrenamiento
de la sección 3.2. No se inventan fechas por pliegue ni métricas finales de podados.
La tabla de controles verifica también que las fuentes siguen intactas.""")
    codigo('''print("Tabla 28. Auditoría del tracking histórico")
display(pd.DataFrame([{k: v for k, v in verificacion_diagnostico["tracking"].items() if k != "estudios"}]))
controles_diagnostico = dict(verificacion_diagnostico["controles"])
controles_diagnostico["fuentes_siguen_intactas"] = all(
    huella(raiz_diagnostico / ruta) == valor
    for ruta, valor in verificacion_diagnostico["fuentes_sha256"].items())
print("Tabla 29. Controles del diagnóstico offline; todos deben ser verdaderos")
display(pd.Series(controles_diagnostico, name="Cumple").to_frame())
assert all(controles_diagnostico.values()), controles_diagnostico''')
    texto("""### Conclusión

El informe siguiente se genera desde las tablas guardadas e incluye las cifras de
sesgo, dispersión, patrones temporales y errores mayores. Explica las limitaciones
de la comparación y los pendientes de integración. No se eligió un ganador global,
no se reconstruyó un modelo para extraer importancias y no se utilizó prueba.""")
    codigo('display(Markdown((carpeta_diagnostico / "informe.md").read_text(encoding="utf-8")))')
    return celdas


def actualizar_notebook():
    """Ejecuta solo las celdas propias nuevas y conserva las históricas sin ejecución."""
    ruta = RAIZ / "notebooks/modelado_xgboost.ipynb"
    original = json.loads(ruta.read_text(encoding="utf-8"))
    historicas = [c for c in original["cells"] if not c.get("metadata", {}).get("diagnostico_offline")]
    copia = copy.deepcopy(historicas)
    apendice = nbformat.v4.new_notebook(cells=celdas_diagnostico())
    NotebookClient(apendice, timeout=90, kernel_name="cervecera_env",
                   resources={"metadata": {"path": str(RAIZ / "notebooks")}}).execute()
    nuevas = json.loads(nbformat.writes(apendice))["cells"]
    original["cells"] = historicas + nuevas
    if original["cells"][:len(copia)] != copia:
        raise ValueError("Se alteraron celdas históricas.")
    ruta.write_text(json.dumps(original, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Ejecutadas solo {sum(c['cell_type'] == 'code' for c in nuevas)} celdas del apéndice; "
          f"{len(historicas)} celdas históricas preservadas.")


if __name__ == "__main__":
    actualizar_notebook()
