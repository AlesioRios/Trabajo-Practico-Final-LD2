# ADR 0022: segunda búsqueda de LightGBM con Optuna

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

La primera búsqueda obtuvo MAE CV 3982.2353 y MAE de validación independiente
2321.0388. Son evaluaciones de períodos distintos y con diferentes tamaños de
ajuste: la CV reúne 677 filas comunes entre 22/11/2020 y 27/10/2022; el ganador
se ajusta con las 830 filas de entrenamiento (01/07/2020–27/10/2022) y se mide
en 171 filas comunes de validación (29/10/2022–26/04/2023).
Los períodos se refieren a las etiquetas DIA; el repositorio documenta fechas
repetidas y desfasadas, por lo que los índices de fila son la referencia.

Se revisaron la función objetivo, la creación de rezagos y medias móviles, el
descarte de etiquetas de frontera de Fase 2 y las separaciones de cada pliegue.
No se detectó fuga entre ajuste y evaluación en esa implementación. No se altera
el objetivo para hacer coincidir errores de distintas épocas. Persisten la
selección previa de variables no anidada, el uso previo de validación para
comparaciones y la falta de certificación de disponibilidad operacional de
algunas mediciones (ADR 0017). La auditoría detallada se conserva en
results/lightgbm_2/auditoria_previa.json.

## Decisión
Crear lightgbm-temporal-v2 en results/optuna/lightgbm_2/estudio.sqlite3.
Conservar exactamente 100 trials totales secuenciales, TPESampler(seed=42),
minimize y MedianPruner(n_startup_trials=5, n_warmup_steps=2,
interval_steps=1, n_min_trials=3). La función objetivo es idéntica a la primera
búsqueda: cinco pliegues TimeSeriesSplit con gap=1, ajuste solo en entrenamiento,
MAE acumulado sobre filas comunes y un estimador nuevo por pliegue.
Las pruebas comparan los AST de ambas funciones objetivo y la configuración fija.

| Parámetro | Rango | Distribución |
|---|---|---|
| n_estimators | 150–900 | Entera |
| learning_rate | 0.01–0.10 | Uniforme |
| min_child_samples | 10–60 | Entera |
| colsample_bytree | 0.65–1.0 | Uniforme |
| max_depth | 3–6 | Entera |
| num_leaves | 8–min(31, 2**max_depth) | Entera condicionada |
| subsample | 0.75–1.0 | Uniforme |
| reg_alpha | 0.001–10 | Logarítmica |
| reg_lambda | 0.001–10 | Logarítmica |

Se conserva el orden de propuestas de la primera búsqueda para evitar cambios
innecesarios. Fijos: subsample_freq=1, random_state=42, verbosity=-1, n_jobs=1.
No se agregan imputación, escalado, early stopping ni transformaciones del objetivo.
Los datos y las 15 variables se mantienen idénticos por SHA-256.

Se mantienen registro común append-only, persistencia SQLite, huellas y versiones,
reanudación de TPE y progreso por trial. Las notas identifican el estudio v2.
progreso.jsonl, CSV, JSON, copia SQLite, PNG y modelos tienen carpetas v2 propias.
Los archivos de la primera búsqueda se verifican mediante un manifiesto SHA-256.

## Comparación
La selección se basa exclusivamente en el MAE CV de Optuna. Solo después de los
100 trials se ajusta el ganador sobre entrenamiento y se evalúa una vez en la
validación independiente, sobre las mismas filas de la primera búsqueda.
comparar_lightgbm.py recarga ambos modelos y verifica sus predicciones frente a
las guardadas. No ajusta modelos ni propone hiperparámetros.
Los errores de entrenamiento son diagnósticos dentro de muestra sobre las 813
filas comunes, calculados con los modelos ya guardados. No equivalen a CV.
Se guardan brechas train/validación; una brecha positiva puede sugerir sobreajuste,
pero también refleja cambios de distribución entre períodos. No demuestra por sí
sola una causa ni significación estadística.
Se reportan diferencias firmadas segunda menos primera, y porcentajes relativos
a la primera; para errores, un valor negativo indica mejora.
Se considera cerca de un límite al 5% del ancho del rango, en log10 para reg_alpha
y reg_lambda. num_leaves usa el máximo efectivo condicionado por profundidad.

## Gráficos y reproducción
Los mismos tres gráficos matplotlib: historial, importancia f-ANOVA(seed=42) y
coordenadas paralelas, con el estilo de la primera búsqueda y PNG a 220 dpi.
La importancia estándar excluye num_leaves por su distribución dinámica;
coordenadas paralelas sí lo incluye. El guardado verifica que SQLite no cambie.

Desde la raíz de modelado-lightgbm, con Python 3.12 y requirements.txt:

```powershell
python -m pytest tests/test_modelado_lightgbm_2.py -q --basetemp=.pytest_tmp_lgbm2
python src/modelado_lightgbm_2.py
```

El último comando reanuda hasta 100 trials totales, nunca agrega otra búsqueda.
Resultados y comparación: results/lightgbm_2/. Modelos: models/lightgbm_2/.
No se usa test y no se ejecuta una tercera búsqueda.

## Resultados ejecutados

100 trials: 46 completos, 54 podados, cero fallidos. Ganador: trial 99 (desde cero).
Se mantuvieron sampler, semilla, pruner, datos, función objetivo y evaluación.
La selección fue exclusivamente por MAE CV, antes de evaluar validación independiente.

| Métrica | Primera | Segunda | Segunda menos primera |
|---|---:|---:|---:|
| mae_optuna | 3982.235290 | 3948.726060 | -33.509230 |
| mae_validacion | 2321.038831 | 2276.735434 | -44.303398 |
| rmse_validacion | 2937.758378 | 2873.135674 | -64.622705 |
| r2_validacion | 0.428639 | 0.453499 | 0.024860 |
| mae_entrenamiento | 2060.125213 | 2358.134517 | 298.009304 |
| rmse_entrenamiento | 2681.561802 | 3044.962152 | 363.400350 |
| r2_entrenamiento | 0.882142 | 0.848034 | -0.034108 |

| Hiperparámetro | Primera | Segunda |
|---|---:|---:|
| subsample_freq | 1 | 1 |
| random_state | 42 | 42 |
| verbosity | -1 | -1 |
| n_jobs | 1 | 1 |
| n_estimators | 181 | 200 |
| learning_rate | 0.03424176159068181 | 0.036985470389312915 |
| max_depth | 6 | 3 |
| num_leaves | 19 | 8 |
| min_child_samples | 30 | 31 |
| subsample | 0.7707876885996955 | 0.7870286296445328 |
| colsample_bytree | 0.8526564656588659 | 0.8638540491377883 |
| reg_alpha | 0.04936749658169118 | 0.4172446705183493 |
| reg_lambda | 0.2615223634574941 | 2.5151783649755584 |

La reducción del MAE CV es 33.509230 (0.841468 %); en validación independiente,
44.303398 (1.908775 %). Es una mejora pequeña y consistente en las dos evaluaciones,
sin evidencia suficiente para declararla estadísticamente significativa. Estas
diferencias corresponden a las configuraciones de las búsquedas; no demuestran
por sí solas una mejora atribuible al indicador `fin_de_año`.

La primera búsqueda tiene MAE de entrenamiento 2060.13 y validación 2321.04;
la segunda tiene 2358.13 y 2276.74. No se observa una brecha mayor compatible con
más sobreajuste en la segunda. Su menor profundidad limita complejidad.
Estos conjuntos pertenecen a épocas distintas: la ausencia de brecha positiva
no demuestra ausencia de sobreajuste ni garantiza el desempeño en test.
Los errores de entrenamiento usan las mismas 813 filas comunes para ambos modelos;
los de validación las mismas 171, sin reentrenar el primer ganador.

max_depth=3 está en el límite inferior. Con profundidad 3, el rango efectivo de
num_leaves es 8–8, por lo que ese parámetro queda fijo por condición. n_estimators=200
está en la zona baja (150–900, a 6.67 % del ancho desde el mínimo), sin tocar el límite.
Ningún otro parámetro queda a menos del 5 % de un extremo en su escala de muestreo.
No hay señales de que se haya aprovechado el nuevo máximo de 900 árboles.

La diferencia entre el MAE CV cercano a 4000 y la validación cercana a 2300 se explica
por períodos y tamaños de ajuste diferentes. La auditoría no detectó fuga entre
ajuste y evaluación; sigue vigente la selección previa de variables no anidada.
No se modificó la función objetivo para acercar artificialmente las métricas.

Se conservan la primera búsqueda y los datos sin cambios por SHA-256. El log previo
permanece íntegro y se agregaron 46 trials completos y una evaluación de validación.
El código y sus hashes quedan archivados; el log conserva el estado Git real,
con cambios sin commit. No se evaluó test ni se ejecutó una tercera búsqueda.


Ver results/lightgbm_2/informe.md y verificacion.json.

## Aclaraciones de la auditoría previa a PR

Antes: 0020-lightgbm-segunda-busqueda.md; ahora ADR 0022 para no colisionar
con otros identificadores del registro de decisiones. La metodología y la evidencia
histórica no cambian. Los archivos bajo codigo_ejecutado son snapshots para
auditoría: no son puntos de entrada ejecutables desde esa subcarpeta porque
RAIZ se calcula respecto de src/. Los ejecutores son src/modelado_lightgbm*.py.
Para consultar resultados existentes, leer informe.md, comparacion.csv y los PNG;
no ejecutar los comandos de optimización como método de lectura. En un clon,
SQLite está archivado en estudio.sqlite3.zip; el script de búsqueda no lo restaura
automáticamente y, sin base local, inicia un estudio nuevo solo si se lo ejecuta.
La restauración exacta de los contratos exige Python y versiones registradas,
bytes del código y finales de línea originales; véanse las huellas de cada detalle.json.

## Auditoría del tracking exigido en 3.2

Se contrastaron los trials COMPLETE y las evaluaciones guardadas con el log:
86 registros: 84 configuraciones CV completas y 2 ajustes evaluados en validación. Todas las filas incluyen parámetros fijos
y sugeridos en hiperparametros (JSON), mae, rmse, r2 y fecha_hora ISO. El timestamp
lo genera registrar_experimento al registrar; no representa el inicio del fit.
El callback de Optuna escribe automáticamente y usa estudio/trial para evitar
duplicados. Cada experimento CV completo agrupa los cinco pliegues; no se registra
una fila por ajuste de pliegue. Los trials podados conservan sus tiempos, estados
e intermedios originales en SQLite y detalle.json; no se inventan métricas finales
para agregarlos como COMPLETE al CSV. No se detectaron registros completos faltantes.

El atributo merge=union del CSV no valida filas ni deduplica. La integración debe
conservar las filas completas de todas las ramas, normalizando solo el orden JSON
para detectar copias idénticas; no deduplicar por modelo/hiperparámetros/MAE, porque
ejecuciones reales distintas pueden repetir esos valores. Se preservan exactamente
fecha_hora, commit, notas y métricas. La consolidación de registros de todos los modelos corresponde a la integración
posterior del equipo, fuera de los PR individuales. Los logs de estas ramas
conservan solo sus registros propios y los baselines heredados. Los archivos de
unión y auditoría general se mantienen fuera de ambos repositorios y no se
proponen para estos PR. No se repitieron entrenamientos ni se asignaron timestamps
históricos nuevos.
