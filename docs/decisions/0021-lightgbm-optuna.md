# ADR 0021: primera búsqueda conservadora LightGBM con Optuna

- **Fecha:** 14/11/2026
- **Estado:** aceptada

## Contexto
Al comenzar la Fase 3 y antes del entrenamiento de los modelos de los experimentos
vigentes, decidimos incorporar la característica `fin_de_año`, identificada como
`es_fin_de_anio` en el código y los datos. La decisión se basó en la hipótesis de
que este indicador de calendario podía aportar información predictiva sobre el
consumo del sistema de refrigeración durante el período de fin de año.

Los experimentos vigentes de XGBoost y LightGBM se realizaron con 15 características,
incluyendo `fin_de_año`. Su incorporación es una decisión de ingeniería de
características motivada por una hipótesis, no una mejora demostrada del MAE:
los resultados disponibles no permiten atribuirle de manera aislada una reducción
del error. El indicador utiliza calendario conocido al emitir la predicción;
vale 1 para DIA + 1 entre el 24 de diciembre y el 6 de enero inclusive, y 0 en el
resto del año.

Se replica la configuración de Optuna de la primera búsqueda XGBoost (commit
b964060 y results/xgboost/detalle.json), sin tomar configuraciones de sus búsquedas
posteriores. dataset_final.csv y data_lineage.json fijan las 15 características;
se mantienen sus valores, el preprocesamiento y la partición temporal.

## Decisión
100 trials totales secuenciales; TPESampler(seed=42), minimize, MedianPruner con
n_startup_trials=5, n_warmup_steps=2, interval_steps=1, n_min_trials=3.
Cada trial ajusta modelos nuevos sobre cinco pliegues TimeSeriesSplit(gap=1),
exclusivamente en entrenamiento. Se minimiza el MAE reunido de las predicciones
sobre filas_comunes, no la media no ponderada de MAE por pliegue. Se reporta el
MAE acumulado para poda, exactamente como en la primera búsqueda XGBoost.
No se imputa, escala, transforma el objetivo ni agrega early stopping.

Rangos: n_estimators 100–500; learning_rate 0.02–0.10 uniforme; max_depth 3–6;
num_leaves 8–min(31, 2**max_depth); min_child_samples 20–60; subsample y
colsample_bytree 0.75–1.0; reg_alpha y reg_lambda 0.001–10 logarítmicos.
Fijos: subsample_freq=1, random_state=42, verbosity=-1, n_jobs=1.

SQLite independiente: results/optuna/lightgbm/estudio.sqlite3, estudio
lightgbm-temporal-v1. Se conservan el contrato con hashes/versiones, los estados,
intermedios, predicciones y métricas por pliegue. Se recupera la secuencia TPE al
reanudar y se rechazan contratos diferentes o trials inconclusos.
Los completos se agregan idempotentemente al log común local; progreso.jsonl
conserva el seguimiento de todos los estados, distinguiendo MAE parcial y final.
Los números de trial siguen la numeración Optuna desde cero.

El ganador se ajusta con las 830 filas de entrenamiento y se evalúa una vez en
las 171 filas comunes de validación independiente. Se guarda el modelo de
validación (joblib y texto LightGBM), sin entrenar con validación ni evaluar test.
La evaluación es idempotente y conserva predicciones y métricas.
Los gráficos replican el uso de optuna.visualization.matplotlib de XGBoost:
historial, importancia f-ANOVA(seed=42) y coordenadas paralelas, PNG a 220 dpi.
La visualización verifica que SQLite no cambió. f-ANOVA estándar excluye las
distribuciones dinámicas del espacio común: num_leaves puede no figurar en las
importancias por depender de max_depth; se conserva en historial y coordenadas.

## Justificación y límites
La validación cruzada selecciona hiperparámetros; el MAE de validación independiente
se informa por separado. La selección previa de variables no es anidada. Esta
primera búsqueda tiene rangos conservadores y no se amplía según sus resultados.
La trazabilidad conserva commit base, estado real del árbol y hashes del código,
datos y entorno, sin afirmar que los cambios sin commit están versionados.
No se modifica ningún archivo de modelado-xgboost ni los módulos comunes.

## Reproducción
Desde modelado-lightgbm con Python 3.12 y requirements.txt instalados:

```powershell
python -m pytest tests -q --basetemp=.pytest_tmp_completa
python src/modelado_lightgbm.py
```

El segundo comando reanuda hasta 100 totales; no inicia una segunda búsqueda.
Resultados en results/lightgbm; gráficos en results/lightgbm/graficos;
modelo en models/lightgbm. La copia SQLite comprimida permite archivar la corrida.

## Resultados ejecutados
100 trials: 38 completos, 62 podados y cero fallidos. Ganador: trial 72
(numeración desde cero), MAE CV 3982.2352900988362, RMSE CV
5264.670670065065 y R² CV 0.5907297211321275, sobre 677 filas comunes.
Los parámetros exactos están en results/lightgbm/resumen.json.
En las 171 filas comunes de validación independiente: MAE 2321.0388311700585,
RMSE 2937.758378144207, R² 0.42863915413104614.
Mejora frente a media móvil: 7.07 %, menor que el 10 %
requerido para ese criterio de la consigna. Esto no mide generalización a test.
Se verificaron 110 archivos de XGBoost sin cambios y el modelo recargado
reprodujo las predicciones guardadas con sus 181 árboles.
Pruebas: 168 aprobadas y una omitida por ausencia de Excel privados, contando
la repetición del control del contrato de datos actualizado.
El código ejecutado, su entorno y el log se archivan en results/lightgbm/.
matplotlibrc alinea los títulos a la izquierda para evitar superponer el título
propio con el título predeterminado de Optuna. No cambia la optimización.
No se ejecutó una segunda búsqueda.

## Aclaraciones de la auditoría previa a PR

Este documento se numeraba 0019-lightgbm-optuna.md. Se renumera a 0021 para
mantener una numeración única en el registro de decisiones.
Las copias históricas de evidencia no se modifican. El dataset y el linaje guardados
bastan para revisar los resultados actuales. Las condiciones técnicas de reproducción
desde los datos originales se documentan en README.md y en el ADR 0015.
El checksum general del dataset se corrigió durante la auditoría, sin cambiar sus bytes.
La corrida se ejecutó con cambios sin commit; el registro conserva ese hecho y sus
hashes. No se atribuye el código nuevo al commit base como si ya estuviera versionado.
