# Gradient Boosting — reproducción del candidato de Fase 3

Desde la raíz del repositorio, en `cervecera_env` con Python 3.12 y las versiones de `requirements.txt`:

```powershell
conda activate cervecera_env
python -m pytest tests/test_modelado_gradient_boosting.py -q --basetemp=.pytest_tmp_gb
python src/modelado_gradient_boosting.py
```

El segundo comando inicia directamente Optuna: 100 trials totales, secuenciales, sin entrenamiento preliminar. Reutiliza `modelado_comun.py`, `TimeSeriesSplit(5, gap=1)`, TPESampler(seed=42) y MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1, n_min_trials=3), según los ejecutores reales de XGBoost y LightGBM. El objetivo es el MAE conjunto de las predicciones fuera de muestra sobre filas comunes a ambos baselines.

Cada fold crea `armar_pipeline("A", columnas=columnas)` y reemplaza su paso `regressor` por GradientBoostingRegressor. La mediana y StandardScaler se ajustan solo con ese fold. El dataset, el orden de las 15 variables y las particiones se leen del contrato vigente; no se regeneran ni se transforma el objetivo.

| Parámetro | Espacio |
|---|---|
| n_estimators | 100–600, paso 50 |
| learning_rate | 0.01–0.15, logarítmico |
| max_depth | 2–4, entero |
| min_samples_leaf | 3–20, entero |
| subsample | 0.6–1.0, paso 0.1 |
| max_features | 0.6–1.0, paso 0.1 |
| loss | squared_error, absolute_error, huber |
| alpha | 0.75–0.95, paso 0.05, exclusivamente huber |

Fijos: random_state=42 y n_iter_no_change=None. Los demás valores por defecto pertenecen a scikit-learn 1.9.1; las versiones efectivas se conservan en dependencias_ejecutadas.txt y detalle.json.

## Persistencia y reanudación

El estudio propio es `gradient_boosting-temporal-v1`, en `results/optuna/gradient_boosting/estudio.sqlite3`. El comando normal continúa hasta 100 totales, valida el contrato y reconstruye el estado del sampler reproduciendo propuestas sin volver a ajustar trials anteriores. No iniciar dos procesos simultáneos sobre el mismo estudio.

Una interrupción entre trials se reanuda con el mismo comando. Si quedaron RUNNING, después de verificar que el proceso anterior terminó, usar `python src/modelado_gradient_boosting.py --recuperar-interrumpidos`: conserva los números y resultados, marca los interrumpidos FAIL y continúa hasta 100; por ello no garantiza 100 COMPLETE. No elimina ni repite trials. No continuar si cambió el código, los datos o las versiones: el contrato lo rechaza.

En un clon, restaurar primero `results/gradient_boosting/estudio.sqlite3.zip` en `results/optuna/gradient_boosting/` mediante `python -m zipfile -e results/gradient_boosting/estudio.sqlite3.zip results/optuna/gradient_boosting`. La ejecución sobre el estudio terminado no añade trials. Para repetir desde cero sin reutilizar el log ni las evaluaciones, crear un worktree separado en el commit de entrenamiento: `git worktree add --detach ../reproduccion-gradient 2e16156fb04078f83518df9d16b7b0507ae2529b`. Entrar en ese directorio, activar el entorno y ejecutar `python src/modelado_gradient_boosting.py`. Ese commit contiene el código probado y el log previo, sin los resultados de esta corrida. Conservar intacta la rama de entrega. Para continuar un SQLite en otra máquina, además de las versiones deben coincidir los bytes de código: restaurar desde codigo_ejecutado los archivos correspondientes de src/ y requirements.txt si el checkout convirtió finales de línea; los hashes del contrato rechazan diferencias. Los snapshots de codigo_ejecutado son evidencia, no puntos de entrada.

## Evidencia y alcance

`results/gradient_boosting/` contiene trials.csv, detalle.json con métricas y tiempos por fold, progreso.jsonl, resumen.json, validacion.json, predicciones, residuos, comparación, ocho gráficos e informe.md. El log común agrega una fila por fold completo y una por estado final de trial; los estados incompletos tienen métricas finales vacías. Las notas identifican estudio, trial, fold y estado. El timestamp de fold es UTC; Optuna y el registrador común conservan el reloj local de ejecución. La duración usa un reloj monotónico. Un experimento CV completo agrupa además sus predicciones para recuperar sus métricas agregadas.

La evaluación independiente sucede solamente después de los 100 trials. El mejor pipeline se ajusta con entrenamiento y se guarda en `models/gradient_boosting/modelo_validacion.joblib`, como candidato. El notebook propio lee evidencia y puede ejecutarse desde cero sin lanzar otra búsqueda. El diagnóstico recarga el candidato y verifica predicciones sin reentrenar.

La preservación de datos, código compartido y resultados históricos se verifica por SHA-256; el log anterior conserva su prefijo exacto. El commit de entrenamiento y el estado real de Git quedan registrados; no se atribuyen cambios sin commit a una versión histórica.

Se cubren optimización, tracking y diagnóstico del candidato. Gradient Boosting es adicional a los cuatro modelos obligatorios de 3.1. El cumplimiento global exige completar y comparar todos ellos. Se reservan al cierre conjunto la elección global, train+validation, test, versión final y registry (3.4). No se alteran los puntos de entrada compartidos.

## Verificación ejecutada

El estudio real terminó con 38 COMPLETE, 62 PRUNED y cero FAIL. Se verificó la reapertura y reproducción de las cien propuestas del sampler sin añadir trials. Los 37 checks propios y de infraestructura común pasaron. La suite completa ejecutada durante la corrida dio 217 passed, 1 skipped y 1 fallo heredado en `test_tracking_historico_completo_y_fuentes_intactas` de XGBoost. `fallo_preexistente_suite.json` demuestra que el hash esperado del log ya discrepaba en el commit base; además tres CSV históricos presentan diferencias de finales de línea desde el checkout inicial de Windows. Se conservaron todos esos bytes y no se modificaron pruebas ni resultados ajenos para ocultar el fallo. Los reportes XML guardan la ejecución real.

Los ocho PNG se guardaron y el notebook se ejecutó desde cero con el kernel cervecera_env. La evidencia previa comprueba preservación por hash de todos los archivos históricos y el prefijo exacto del log. El candidato contiene el pipeline completo; no es el modelo final global.
