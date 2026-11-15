# Segunda búsqueda de LightGBM: resultados

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
sin evidencia suficiente para declararla estadísticamente significativa.

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

Archivos:
- estudio y datos por trial: resumen.json, trials.csv, detalle.json y estudio.sqlite3.zip;
- seguimiento: progreso.jsonl y experiment_logs.csv;
- comparación: comparacion.json, comparacion.csv, parametros_comparados.csv y limites.csv;
- auditoría: auditoria_previa.json y verificacion.json;
- modelo: ../../models/lightgbm_2/modelo_validacion.joblib y modelo_validacion.txt;
- figuras: graficos/historial_optimizacion.png, graficos/importancia_hiperparametros_fanova.png
  y graficos/coordenadas_paralelas.png.

La f-ANOVA usa 46 trials completos y excluye num_leaves por su distribución dinámica;
las coordenadas paralelas sí muestran num_leaves. No se interpretan importancias
como causalidad. Los PNG fueron revisados visualmente y el guardado no modificó SQLite.
