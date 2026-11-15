# Fase 3 — Comparación final por MAE de validación

Fecha de análisis: 15/11/2026. Estado: selección provisional; sin modelo definitivo.

El ganador de los experimentos de **Fase 3** es **Lasso, 14 variables, pipeline A**, con MAE de validación **2215.796375483132 kW**. Dos ejecuciones históricas (búsqueda inicial y Optuna) registran la misma configuración y las mismas métricas. El segundo candidato distinto es XGBoost inicial con 15 variables. La elección utiliza exclusivamente MAE del tramo `validacion`; RMSE y R² son descriptivos.

## 1. Alcance, sincronización y procedencia

Se revisó `git status`, se ejecutaron `git fetch --all --prune` y `git pull --ff-only`. `feature/modeling` quedó en `e596b79f27d828fec782c61e3b4ee486d89f1e02`, idéntico a `origin/feature/modeling` al consultar el remoto. Se utilizó un worktree separado, `comparacion-final-modelos`, para preservar los cambios locales de la carpeta principal. No se hicieron merges, commits ni push.

| Rama inspeccionada | Commit remoto | Integrada en feature/modeling |
|---|---|---|
| `origin/feature/modeling-rf-lineales` | `9ae7b736f2adf6d1b3ec8981d13ebfe039ba7d8a` | No |
| `origin/feature/modeling-rf-lineales-experimentos` | `9832116044b7f9b82cdd041f7b66efdc7338603c` | No |
| `origin/feature/modeling-gradient-boosting` | `3d91d4ab44f7dba05d6a2814161059cc918ecce2` | No |
| `origin/main` | `14b1a6df13fc5f0db05e79ed836f76210e5790d9` | No |
| `origin/feature/prediction-pipeline` | `75bd9499b24d6724b1a9a505824dde38ad0de698` | No |

Los resultados de Random Forest, Ridge y Lasso provienen de las dos ramas rf-lineales; la rama de experimentos incluye las cuatro búsquedas de cada familia y contiene los resultados de la rama inicial. Gradient Boosting procede de su rama propia. Se inspeccionaron mediante `git show`, sin integrar código ajeno. `origin/main` y prediction-pipeline no aportaron experimentos nuevos a la unión.

El registro exigido por la consigna es [experiment_logs.csv](experiment_logs.csv): se conservaron sus 229 filas previas byte a byte y se agregaron 655 registros históricos ausentes, con timestamp, hiperparámetros, MAE, RMSE, R², commit y notas originales. No son nuevas ejecuciones. Total: **884 registros**, 19 de validación y 865 de entrenamiento/evaluación interna. Las copias del mismo registro en distintas ramas se deduplicaron por identidad; las ejecuciones distintas con iguales métricas se conservaron. [comparacion_modelos_fuentes.csv](comparacion_modelos_fuentes.csv) relaciona cada fila del ranking con rama, commit, fila original y fila consolidada; conserva todos los decimales originales.

## 2. Comparabilidad y prevención de fuga

La validación tiene 176 filas etiquetadas; se comparan las **171 filas comunes**, con objetivo, persistencia y media móvil presentes, fechadas entre **2022-10-29 y 2023-04-26**. Índices del CSV: 830 a 1005, excluyendo 891, 907, 948, 993 y 998. `DIA` no es clave única: la identidad se verificó por posición, fecha y valor objetivo.

Se comprobó igualdad exacta de las 14 columnas originales, objetivo, baselines, orden e índices entre el dataset vigente y los commits de ejecución de las cuatro búsquedas rf-lineales y Gradient Boosting. Las diferencias de 14 frente a 15 entradas no alteran las observaciones evaluadas: las versiones de 15 agregan `es_fin_de_anio`, calculado con calendario conocido para DIA+1. La comparación mide configuraciones completas y debe informar esa diferencia.

Las cinco tablas guardadas de predicciones de XGBoost inicial/refinado, LightGBM inicial/segunda búsqueda y Gradient Boosting coinciden exactamente en las 171 identidades y objetivos. Para rf-lineales, se verificó el código de los notebooks ejecutados: cargan ese dataset, usan `filas_comunes(validacion)`, ajustan con las 830 filas de entrenamiento y registran 171 predicciones. No se encontraron archivos independientes con las predicciones individuales de Lasso/Ridge/Random Forest: su respaldo es el log y el notebook ejecutado, no una reproducción nueva.

El objetivo nominal es `Frio (Kw)` del día siguiente. El código realmente etiqueta con el consumo del **bloque siguiente**, `consumo_dia.shift(-1)` (ADR 0002 y 0015). Se respeta la división cronológica 70/15/15 y se quitan las dos etiquetas de frontera antes de exportar el dataset. Los ajustes internos usan cinco pliegues temporales con una fila de separación (`TimeSeriesSplit(n_splits=5, gap=1)`), sin mezcla aleatoria. Lasso/Ridge aprenden imputación por mediana y `StandardScaler` dentro de cada ajuste; el objetivo queda en sus unidades originales. Rezagos y medias miran hoy y el pasado; calendario futuro no utiliza consumos futuros. En este análisis no se ejecutaron funciones de carga/modelado del proyecto ni se calcularon nuevas predicciones.

**Límite de la verificación:** no se observa incorporación explícita del objetivo futuro a las entradas en los flujos revisados, pero no se certifica ausencia absoluta de fuga operativa: faltan publicación efectiva a las 23:59, latencias y correspondencia bloque-fecha física. La validación ya intervino en la selección de variables y se reutilizó para comparar familias; por ello no es una evaluación final independiente. La separación de un bloque no demuestra por sí sola separación de fechas físicas con fechas duplicadas.

## 3. Ranking completo de experimentos comparables de Fase 3

Se muestran **todas las ejecuciones registradas con `tramo=validacion`**, ordenadas de menor a mayor MAE. Los empates de Lasso y Ridge se conservan: no son nuevos competidores distintos. Las cifras se transcriben del registro, sin inventar métricas faltantes ni sustituir MAE de validación por error interno.

| Modelo | Experimento | MAE validación | RMSE validación | R² validación | Timestamp | Hiperparámetros |
|---|---|---:|---:|---:|---|---|
| Lasso | 14 variables; búsqueda inicial | 2215.796375483132 | 2835.803833786694 | 0.4676090070547668 | 2026-11-14T16:38:40 | `{"alpha": 75.0, "max_iter": 20000, "random_state": 42}` |
| Lasso | 14 variables; Optuna | 2215.796375483132 | 2835.803833786694 | 0.4676090070547668 | 2026-11-14T17:35:09 | `{"alpha": 75.0, "max_iter": 20000, "random_state": 42}` |
| XGBoost | 15 variables; inicial, trial 56 | 2221.760633680556 | 2844.3074191916603 | 0.4644113105021027 | 2026-11-14T18:35:43 | `{"colsample_bytree": 0.8086142929785466, "eval_metric": "mae", "gamma": 0, "learning_rate": 0.0020702351473759497, "max_depth": 3, "min_child_weight": 4.49480789319248, "n_estimators": 1908, "n_jobs": 1, "objective": "reg:squarederror", "random_state": 42, "reg_alpha": 0, "reg_lambda": 1, "subsample": 0.9516935603337213, "tree_method": "hist"}` |
| Ridge | 14 variables; búsqueda inicial | 2225.4431772899006 | 2863.228861288026 | 0.4572617187707618 | 2026-11-14T16:38:35 | `{"alpha": 30.0, "random_state": 42}` |
| Ridge | 14 variables; Optuna | 2225.4431772899006 | 2863.228861288026 | 0.4572617187707618 | 2026-11-14T17:35:06 | `{"alpha": 30.0, "random_state": 42}` |
| XGBoost | 15 variables; refinado, trial 49 | 2229.224629363121 | 2853.130567574821 | 0.4610833242874175 | 2026-11-14T19:26:21 | `{"colsample_bytree": 0.89176387518403, "eval_metric": "mae", "gamma": 0, "learning_rate": 0.0020702351473759497, "max_depth": 3, "min_child_weight": 4.632253473557002, "n_estimators": 1994, "n_jobs": 1, "objective": "reg:squarederror", "random_state": 42, "reg_alpha": 0, "reg_lambda": 1, "subsample": 0.9642293818510409, "tree_method": "hist"}` |
| Lasso | 15 variables; búsqueda inicial | 2265.620734059099 | 2901.5021861981018 | 0.4426549696120088 | 2026-11-14T17:55:36 | `{"alpha": 75.0, "max_iter": 20000, "random_state": 42}` |
| Lasso | 15 variables; Optuna | 2265.620734059099 | 2901.5021861981018 | 0.4426549696120088 | 2026-11-14T17:58:05 | `{"alpha": 75.0, "max_iter": 20000, "random_state": 42}` |
| LightGBM | 15 variables; segunda búsqueda, trial 99 | 2276.735433652959 | 2873.1356736323023 | 0.4534994570180016 | 2026-11-14T20:04:25 | `{"colsample_bytree": 0.8638540491377883, "learning_rate": 0.036985470389312915, "max_depth": 3, "min_child_samples": 31, "n_estimators": 200, "n_jobs": 1, "num_leaves": 8, "random_state": 42, "reg_alpha": 0.4172446705183493, "reg_lambda": 2.5151783649755584, "subsample": 0.7870286296445328, "subsample_freq": 1, "verbosity": -1}` |
| Ridge | 15 variables; búsqueda inicial | 2284.2012581602876 | 2936.328465501733 | 0.42919522250774456 | 2026-11-14T17:55:32 | `{"alpha": 30.0, "random_state": 42}` |
| Ridge | 15 variables; Optuna | 2284.2012581602876 | 2936.328465501733 | 0.42919522250774456 | 2026-11-14T17:58:01 | `{"alpha": 30.0, "random_state": 42}` |
| Random Forest | 15 variables; búsqueda inicial | 2284.521082507486 | 2954.7496658104246 | 0.4220108137021351 | 2026-11-14T17:55:29 | `{"max_depth": 10, "max_features": 0.5, "min_samples_leaf": 12, "n_estimators": 200, "n_jobs": 1, "random_state": 42}` |
| Random Forest | 14 variables; búsqueda inicial | 2293.7032756508256 | 2958.8973721710026 | 0.4203869793257915 | 2026-11-14T16:38:28 | `{"max_depth": 10, "max_features": 0.5, "min_samples_leaf": 12, "n_estimators": 200, "n_jobs": 1, "random_state": 42}` |
| Random Forest | 14 variables; Optuna | 2305.952520283372 | 2972.28716651894 | 0.415129305110031 | 2026-11-14T17:35:03 | `{"max_depth": 4, "max_features": 0.5, "min_samples_leaf": 1, "n_estimators": 200, "n_jobs": 1, "random_state": 42}` |
| Gradient Boosting | 15 variables; trial 76 | 2316.309953168955 | 2948.0389023858443 | 0.42463326532370216 | 2026-11-15T01:16:59 | `{"learning_rate": 0.01592324888670594, "loss": "absolute_error", "max_depth": 2, "max_features": 0.8, "min_samples_leaf": 4, "n_estimators": 550, "n_iter_no_change": null, "random_state": 42, "subsample": 0.8}` |
| Random Forest | 15 variables; Optuna | 2319.0804723343085 | 2987.3735621407086 | 0.409176997622983 | 2026-11-14T17:57:57 | `{"max_depth": 6, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 200, "n_jobs": 1, "random_state": 42}` |
| LightGBM | 15 variables; primera búsqueda, trial 72 | 2321.0388311700585 | 2937.758378144207 | 0.4286391541310461 | 2026-11-14T19:45:28 | `{"colsample_bytree": 0.8526564656588659, "learning_rate": 0.03424176159068181, "max_depth": 6, "min_child_samples": 30, "n_estimators": 181, "n_jobs": 1, "num_leaves": 19, "random_state": 42, "reg_alpha": 0.04936749658169118, "reg_lambda": 0.2615223634574941, "subsample": 0.7707876885996955, "subsample_freq": 1, "verbosity": -1}` |
| Media móvil 7 días | Baseline | 2497.6535783904205 | 3368.0752006558305 | 0.2489966261072022 | 2026-11-14T13:56:33 | `{"minimo_dias": 4, "ventana": 7}` |
| Persistencia | Baseline | 2668.3274853801167 | 3319.4335844594675 | 0.2705319144472131 | 2026-11-14T13:56:33 | `{}` |

## 4. Mejor configuración por familia

En caso de empate exacto se muestra la primera ejecución histórica. Los baselines también aparecen para facilitar la comparación.

| Modelo | Experimento | MAE validación | RMSE validación | R² validación | Timestamp | Hiperparámetros |
|---|---|---:|---:|---:|---|---|
| Lasso | 14 variables; búsqueda inicial | 2215.796375483132 | 2835.803833786694 | 0.4676090070547668 | 2026-11-14T16:38:40 | `{"alpha": 75.0, "max_iter": 20000, "random_state": 42}` |
| XGBoost | 15 variables; inicial, trial 56 | 2221.760633680556 | 2844.3074191916603 | 0.4644113105021027 | 2026-11-14T18:35:43 | `{"colsample_bytree": 0.8086142929785466, "eval_metric": "mae", "gamma": 0, "learning_rate": 0.0020702351473759497, "max_depth": 3, "min_child_weight": 4.49480789319248, "n_estimators": 1908, "n_jobs": 1, "objective": "reg:squarederror", "random_state": 42, "reg_alpha": 0, "reg_lambda": 1, "subsample": 0.9516935603337213, "tree_method": "hist"}` |
| Ridge | 14 variables; búsqueda inicial | 2225.4431772899006 | 2863.228861288026 | 0.4572617187707618 | 2026-11-14T16:38:35 | `{"alpha": 30.0, "random_state": 42}` |
| LightGBM | 15 variables; segunda búsqueda, trial 99 | 2276.735433652959 | 2873.1356736323023 | 0.4534994570180016 | 2026-11-14T20:04:25 | `{"colsample_bytree": 0.8638540491377883, "learning_rate": 0.036985470389312915, "max_depth": 3, "min_child_samples": 31, "n_estimators": 200, "n_jobs": 1, "num_leaves": 8, "random_state": 42, "reg_alpha": 0.4172446705183493, "reg_lambda": 2.5151783649755584, "subsample": 0.7870286296445328, "subsample_freq": 1, "verbosity": -1}` |
| Random Forest | 15 variables; búsqueda inicial | 2284.521082507486 | 2954.7496658104246 | 0.4220108137021351 | 2026-11-14T17:55:29 | `{"max_depth": 10, "max_features": 0.5, "min_samples_leaf": 12, "n_estimators": 200, "n_jobs": 1, "random_state": 42}` |
| Gradient Boosting | 15 variables; trial 76 | 2316.309953168955 | 2948.0389023858443 | 0.42463326532370216 | 2026-11-15T01:16:59 | `{"learning_rate": 0.01592324888670594, "loss": "absolute_error", "max_depth": 2, "max_features": 0.8, "min_samples_leaf": 4, "n_estimators": 550, "n_iter_no_change": null, "random_state": 42, "subsample": 0.8}` |
| Media móvil 7 días | Baseline | 2497.6535783904205 | 3368.0752006558305 | 0.2489966261072022 | 2026-11-14T13:56:33 | `{"minimo_dias": 4, "ventana": 7}` |
| Persistencia | Baseline | 2668.3274853801167 | 3319.4335844594675 | 0.2705319144472131 | 2026-11-14T13:56:33 | `{}` |

## 5. Ganador y contraste con baselines

- Familia y configuración: Lasso, 14 predictoras, pipeline A; configuración 6 en la búsqueda inicial y ensayo 2 de Optuna (mismo resultado).
- MAE: **2215.796375483132 kW**; RMSE: **2835.803833786694 kW**; R²: **0.4676090070547668**.
- Hiperparámetros registrados: `{"alpha": 75.0, "max_iter": 20000, "random_state": 42}`. Preparación: imputación por mediana, sin indicadores adicionales, y escalado estándar ajustados solo con entrenamiento.
- Timestamps originales: 2026-11-14T16:38:40 y 2026-11-14T17:35:09. Los timestamps originales no incluyen zona horaria; no se les asigna una artificialmente.
- Segundo candidato distinto: XGBoost inicial, 15 variables, ensayo 56; MAE **2221.760633680556**. Ventaja absoluta de Lasso: **5.964258197423987 kW**; reducción respecto del segundo: **0.268447379390 %**, calculada como `(MAE_segundo − MAE_Lasso) / MAE_segundo × 100`.

| Baseline | MAE validación | Mejora absoluta de Lasso | Reducción porcentual |
|---|---:|---:|---:|
| Persistencia | 2668.3274853801167 | 452.531109896985 kW | 16.959354216 % |
| Media móvil 7 días | 2497.6535783904205 | 281.857202907288 kW | 11.284879751 % |

Persistencia usa el consumo del bloque actual. Media móvil usa siete bloques, incluido el actual, con mínimo de cuatro observaciones válidas; no siempre son siete fechas físicas consecutivas (ADR 0018). Sus RMSE y R² constan en el ranking y en el log.

Las 14 entradas del ganador, en orden, son:

1. `Frio (Kw)`
2. `Frio (Kw) x frio_hace_2_dias`
3. `frio_media_7_dias`
4. `Sala Maq (Kw)`
5. `Frio (Kw) x Consolidado Agua | Agua Planta (Hl)`
6. `Servicios (Kw)`
7. `Frio (Kw) x Totalizadores Energia | KW Trafo 9`
8. `dia_anio_coseno`
9. `frio_hace_2_dias x Consolidado Agua | Agua Planta (Hl)`
10. `dia_anio_coseno x Consolidado Agua | Agua Planta (Hl)`
11. `KW Gral Planta`
12. `frio_hace_2_dias`
13. `Consolidado Agua | Agua Planta (Hl)`
14. `Totalizadores Energia | KW Trafo 9`

Lasso ganó porque su MAE en esas mismas 171 observaciones es el menor de la Fase 3. Su regularización y preparación forman parte de la configuración evaluada; esta comparación no demuestra causalmente qué componente produjo la ventaja. La diferencia de aproximadamente 0,268 % frente a XGBoost es pequeña y no prueba superioridad estadística. No se eligió por métricas de entrenamiento, residuos, complejidad ni R².

Fuentes del ganador: [log de la rama de experimentos](https://github.com/AlesioRios/Trabajo-Practico-Final-LD2/blob/9832116044b7f9b82cdd041f7b66efdc7338603c/results/experiment_logs.csv), [notebook inicial](https://github.com/AlesioRios/Trabajo-Practico-Final-LD2/blob/9832116044b7f9b82cdd041f7b66efdc7338603c/notebooks/modelado_rf_lineales.ipynb), [notebook Optuna](https://github.com/AlesioRios/Trabajo-Practico-Final-LD2/blob/9832116044b7f9b82cdd041f7b66efdc7338603c/notebooks/modelado_rf_lineales_optuna.ipynb) y [ADR 0020](https://github.com/AlesioRios/Trabajo-Practico-Final-LD2/blob/9832116044b7f9b82cdd041f7b66efdc7338603c/docs/decisions/0020-random-forest-y-modelos-lineales.md). Los commits de ejecución fueron `8d70702524ff12b9b923877752e337ccb995b5b8` y `954dadb2a8a6709be49f832ab8198870f100c0c0`, ambos con árbol limpio según el registro.

Fuentes de las otras familias: [XGBoost inicial](xgboost_15_validacion/evaluacion.json), [XGBoost refinado](xgboost_15_refinado_validacion/evaluacion.json), [LightGBM inicial](lightgbm/validacion.json), [LightGBM segunda búsqueda](lightgbm_2/validacion.json), [Gradient Boosting](https://github.com/AlesioRios/Trabajo-Practico-Final-LD2/blob/3d91d4ab44f7dba05d6a2814161059cc918ecce2/results/gradient_boosting/validacion.json); rf-lineales y baselines se respaldan fila por fila en el CSV de fuentes.

## 6. Experimentos registrados que no se mezclan con validación

Las 865 filas `entrenamiento` del registro consolidado son evaluaciones internas por pliegue, agregaciones o referencias de entrenamiento, según sus notas. **No tienen un MAE de validación independiente**: no se los ordena contra las 19 filas anteriores ni se usa su métrica para decidir. El registro conserva todos sus hiperparámetros, métricas y timestamps para el punto 3.2. Dentro de las búsquedas se entrenaron muchas configuraciones, pero solo las elegidas fueron evaluadas sobre la validación reservada. No se presume que el resto tenga un resultado en ese tramo.

| Estudio | Ensayos registrados | Estados | MAE de validación disponible |
|---|---:|---|---|
| `results/xgboost/trials.csv` | 100 | {'COMPLETE': 45, 'PRUNED': 55} | No: solo evaluación interna |
| `results/xgboost_15/trials.csv` | 100 | {'COMPLETE': 47, 'PRUNED': 53} | Solo la configuración elegida, en el ranking |
| `results/xgboost_15_refinado/trials.csv` | 60 | {'COMPLETE': 45, 'PRUNED': 15} | Solo la configuración elegida, en el ranking |
| `results/lightgbm/trials.csv` | 100 | {'COMPLETE': 38, 'PRUNED': 62} | Solo la configuración elegida, en el ranking |
| `results/lightgbm_2/trials.csv` | 100 | {'COMPLETE': 46, 'PRUNED': 54} | Solo la configuración elegida, en el ranking |
| `results/gradient_boosting/trials.csv` | 100 | {'COMPLETE': 38, 'PRUNED': 62} | Solo la configuración elegida, en el ranking |

Se detectan 62 registros históricos de Gradient Boosting con `estado=PRUNED`, `filas=0` y MAE/RMSE/R² vacíos en el log de origen. Se preservan tal como fueron registrados y no se inventan métricas; constituyen una salvedad del tracking del punto 3.2, no resultados de validación. Las 19 filas del ranking sí tienen timestamp, hiperparámetros y las tres métricas completos.

Los ensayos podados tienen evaluación interna parcial: no equivalen a modelos ajustados y evaluados en la validación completa. La búsqueda XGBoost original con 14 entradas quedó registrada con resultados internos; no se encontró evaluación independiente de esa configuración en validación. Las tablas de curvas por árbol de XGBoost refinado son diagnóstico de un experimento ya listado, no otras 1994 configuraciones independientes.

Los espacios de búsqueda y su justificación se conservan en ADR 0019, 0021, 0022 y, en las ramas pendientes, ADR 0020/0023 y las celdas de configuración de sus notebooks. La búsqueda inicial de rf-lineales prueba 20 configuraciones del bosque y 10 valores de alpha por lineal; la variante Optuna queda identificada en las notas del log. No se ejecutó ninguna búsqueda en esta tarea.

### Antecedentes de Fase 2: advertencia sobre el alcance del ganador

[seleccion/comparacion_validation_congelada/resumen.csv](seleccion/comparacion_validation_congelada/resumen.csv) contiene evaluaciones históricas de **Ridge(alpha=100, solver=svd)** con distintas selecciones y preparación con indicadores de ausencia. Son antecedentes reales de Fase 2, no experimentos registrados como candidatos de Fase 3. No se los atribuye a Lasso: «lasso» allí nombra un selector de variables, mientras que el regresor evaluado era Ridge.

| Selector usado por Ridge histórico | Filas | MAE validación (176 filas) | MAE sobre soporte de baselines (171 filas) |
|---|---:|---:|---:|
| Pearson | 176 | 2239.579956965333 | 2171.966082973769 |
| Spearman | 176 | 2247.817074339489 | 2173.39634430503 |
| Forward | 176 | 2377.2169092207037 | 2283.3563391169687 |
| Lasso (selector) | 176 | 2723.349101862164 | 2532.070837442604 |
| Información mutua | 176 | 2736.273086935318 | 2538.2923480421878 |

Los errores de 176 filas no son comparables con el ranking de 171. Los valores históricos de 171 sí permiten contraste numérico en ese soporte y **Pearson y Spearman son menores que el MAE de Lasso**. Se mantienen fuera del ranking principal por el alcance explícito de esta tarea (candidatos entrenados de Fase 3), no porque otro preprocesamiento haga incomparables por sí solos sus errores. En consecuencia, **no se afirma que Lasso sea el mejor de toda la historia del repositorio**: si se incluyeran aquellos antecedentes como candidatos, habría que revisar el alcance de la selección; el mínimo histórico mostrado sería Ridge con Pearson. No se reentrenó ni se promovió ese antecedente a un modelo global.

## 7. Cumplimiento y limitaciones pendientes

El punto 3.2 se respalda con el registro consolidado, los timestamps históricos, los hiperparámetros y las fuentes de búsqueda. Para 3.3 se referencia el análisis de residuos, predicción frente a valores reales e importancia por permutación del Lasso ya ejecutado en las secciones 5 y 6 de `notebooks/modelado_rf_lineales.ipynb` (celdas de código 10 y 12). El ADR 0020 identifica frío actual y transformador 9 como las entradas con mayor aumento de error al permutarlas. Esa evidencia está en la rama pendiente y no se vuelve a ejecutar. Los diagnósticos de XGBoost y LightGBM ya integrados se conservan en sus carpetas; Gradient Boosting conserva sus diagnósticos en su rama.

El punto 3.4 queda cerrado **solo en la selección provisional por MAE**. Reentrenamiento con entrenamiento más validación, versionado semántico, serialización definitiva, registro del modelo y evaluación final quedan expresamente pendientes por instrucción del usuario. `src/train_model.py` sigue siendo un archivo de declaración; `notebooks/modelado.ipynb` contiene los baselines y el protocolo, y el resto de las corridas está en notebooks por familia. No se declara completa la entrega de Fase 3.

La configuración ganadora usa 14 entradas; el contrato vigente agrega una decimoquinta. Esto debe reconciliarse explícitamente en la etapa posterior, sin atribuir al Lasso de 15 entradas las métricas del de 14. LightGBM y Gradient Boosting tienen registros con árbol modificado; sus hashes de commit por sí solos no reconstruyen toda la ejecución, aunque se conservan copias del código y huellas. La ausencia de predicciones independientes de los lineales impide recalcular sus métricas sin una nueva ejecución, que no estaba autorizada.

No se ejecutó ni se analizó ninguna métrica de prueba, no se entrenó, no se modificaron modelos serializados ni `models/model_registry.json` y no se registró una versión definitiva. Este informe termina en la selección provisional.
