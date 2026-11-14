# ADR 0019: búsqueda temporal de hiperparámetros de XGBoost con Optuna

- **Fecha:** 14/11/2026
- **Estado:** propuesta
- **Autor:** Juan Cruz Brown

## Contexto

La Fase 3 requiere optimización y seguimiento de experimentos. El ADR 0018 ya define las funciones comunes,
las 14 variables, las filas comparables y cinco pliegues cronológicos con una fila de separación.
Esta etapa busca configuraciones prometedoras solo en entrenamiento. No elige el modelo definitivo.

## Decisión

Usar Optuna con TPESampler(seed=42), dirección minimize y máximo TOTAL de 100 trials secuenciales.
Minimizar el MAE de todas las predicciones reunidas sobre las filas comunes de los cinco pliegues;
registrar también RMSE y R² agregados y por pliegue, reutilizando las funciones comunes.
No imputar, escalar ni transformar el objetivo. Crear un XGBRegressor nuevo por pliegue, sin early stopping.

| Parámetro | Mínimo | Máximo | Distribución |
|---|---:|---:|---|
| n_estimators | 100 | 2000 | Entera |
| learning_rate | 0.001 | 0.3 | Logarítmica |
| max_depth | 3 | 10 | Entera |
| min_child_weight | 0.1 | 10.0 | Logarítmica |
| subsample | 0.5 | 1.0 | Uniforme |
| colsample_bytree | 0.5 | 1.0 | Uniforme |

Mantener reg_alpha=0, reg_lambda=1, gamma=0, objective=reg:squarederror, eval_metric=mae,
tree_method=hist, random_state=42. Fijar n_jobs=1 para limitar recursos.
MedianPruner usa n_startup_trials=5, n_warmup_steps=2, interval_steps=1 y n_min_trials=3.
Reportar MAE acumulado en los pasos 0 a 4. La poda puede comenzar en el tercer pliegue,
una vez disponibles las referencias exigidas; Optuna compara el mejor intermedio del trial con la mediana del paso.

Persistir en results/optuna/xgboost/estudio.sqlite3. La carpeta mutable se ignora para que no altere
el control Git de los experimentos siguientes; results/experiment_logs.csv ya está excluido por el módulo común.
Guardar al terminar una copia SQLite comprimida, detalle JSON, tabla CSV y resumen en results/xgboost/.
Registrar solo trials completos en el log común: podados y fallidos conservan estados y motivos en SQLite y JSON.
El código debe estar commiteado antes de ejecutar; no se modifica la infraestructura común.

La reanudación comprueba huellas de datos y código, versiones y configuración. Reconstruye el estado de TPE
reproduciendo propuestas y resultados anteriores en memoria, sin volver a entrenar ni cargar objetos pickle.
Un trial RUNNING/WAITING bloquea la continuación para revisión; no se descarta ni altera automáticamente.
Optuna 5.0 vincula su caché interna de intersección al estudio de origen: al transferir el sampler
reconstruido se reinicia únicamente esa caché, conservando sus generadores aleatorios. Las pruebas comparan
secuencias continuas y reanudadas, también con trials podados y fallidos. La versión queda fijada por esta integración.
La notebook solo consulta el estudio por defecto y puede restaurar la copia versionada si falta el SQLite local.

## Justificación

Optuna integra propuestas, poda y SQLite sin reemplazar los contratos de evaluación existentes.
TPE aprovecha los trials previos y permite explorar tasas y pesos en escala logarítmica.
MedianPruner evita terminar siempre los cinco ajustes de configuraciones poco prometedoras.
Los cinco pliegues respetan el orden cronológico y hacen comparables los modelos (tabla 2 del notebook).
El MAE acumulado pondera observaciones, evitando que un pliegue con menos filas comunes pese lo mismo que otro mayor.
No usar early stopping garantiza que n_estimators represente la cantidad de árboles realmente entrenada.
Las tablas 6 a 9 y results/xgboost/resumen.json contendrán los resultados de la ejecución, sin presentar la mejor
configuración como definitiva. Esta propuesta se cerrará con las cifras efectivamente obtenidas.

## Alternativas descartadas

- Grilla exhaustiva: combina demasiadas posibilidades y no aprovecha la evidencia de las propuestas anteriores.
- KFold o división aleatoria: mezcla información futura y pasada e incumple el contrato temporal.
- Reutilizar la validación independiente para optimizar: excede el alcance de esta búsqueda.
- Early stopping e imputación o escalado: alteran el experimento acordado y son innecesarios en esta búsqueda de árboles.
- Guardar la base mutable como archivo versionado durante la búsqueda: ensucia el árbol entre trials.
- Reiniciar TPE con la semilla sin recuperar su historia aleatoria: puede cambiar la secuencia al reanudar.

## Limitaciones

La selección previa de variables no se repite dentro de cada pliegue: no es una evaluación anidada completa.
Los trials podados no tienen métricas finales comparables con los completos.
La reproducción exacta requiere mismos datos, versiones y orden; la ejecución es de un único proceso.
Una interrupción abrupta necesita revisar el trial activo antes de reanudar.
Quedan pendientes el análisis exhaustivo de resultados, el entrenamiento definitivo y la evaluación independiente.

## Dependencias y reproducción

Se verificó que todas las versiones previamente declaradas coincidían con cervecera_env (Python 3.12.14).
Se agregan únicamente optuna==5.0.0 y sus dependencias nuevas:
alembic==1.20.1 y sqlalchemy==2.1.8 para SQLite, mako==1.4.3 para las migraciones de Alembic,
colorlog==6.12.0 para mensajes y tqdm==4.70.1 para progreso. XGBoost permanece en 3.4.1 y scikit-learn en 1.9.1.
NumPy, pandas, PyYAML y las restantes dependencias ya estaban declaradas y no se actualizaron.
La instalación inicial usó el pip del Python explícito de cervecera_env; el archivo congelado se verifica con uv.
Ver README para instalación en entorno limpio, ejecución y reanudación.

Referencias: [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html),
[MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html),
[XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor).
