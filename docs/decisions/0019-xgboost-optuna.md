# ADR 0019: búsqueda temporal de hiperparámetros de XGBoost con Optuna

- **Fecha:** 14/11/2026
- **Estado:** aceptada
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
Las tablas 6 a 9 y results/xgboost/resumen.json contienen los resultados ejecutados: 100 trials,
45 completos, 55 podados y 0 fallidos. La mejor configuración provisional es el trial 88,
con MAE 3.986,65 kW, RMSE 5.198,78 kW y R² 0,6009 sobre 677 filas comunes reunidas de los cinco pliegues.
Sus parámetros son n_estimators=1258, learning_rate=0.0042724261637713265, max_depth=4,
min_child_weight=4.491708962592647, subsample=0.9670402445597764 y colsample_bytree=0.9650101376104412.
En esas mismas filas, la persistencia tiene MAE 3.247,71 kW y la media móvil 3.377,77 kW.
Esta búsqueda no mejoró esos baselines; identificar el mínimo de Optuna no demuestra que el modelo agregue valor.
La configuración sigue siendo provisional y el análisis exhaustivo queda para la siguiente etapa.

El código se ejecutó desde b964060a9b253939d1c4d4be5e0e9678f6ee3f18, con árbol limpio en los 45 registros.
La corrida se dividió en una tanda de 10 y su reanudación hasta 100 totales. La copia SQLite comprimida
se restauró y verificó por SHA-256: reanudar al límite no agregó trials ni duplicó filas del log.
La notebook se ejecutó desde cero con cervecera_env y todos sus controles pasaron.
La suite final tiene 166 pruebas aprobadas y una omitida por ausencia de los Excel privados en esta copia separada.
El entorno limpio se instaló con uv --link-mode=copy por la restricción de vínculos de OneDrive;
uv pip check pasó en ambos entornos y dos pruebas funcionales pasaron en el entorno limpio.

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
La auditoría de las 135 versiones efectivas queda en results/xgboost/verificacion_dependencias.json.
Las huellas del código y requirements verifican bytes: esta corrida usó checkout con core.autocrlf=true.
Para restaurar exactamente esa configuración de saltos de línea, seguir el comando de clonación del README.
Los datos mantienen sus bytes por el .gitattributes existente.
Ver README para instalación en entorno limpio, ejecución y reanudación.

Referencias: [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html),
[MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html),
[XGBRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor).


## Vigencia tras la ampliación manual

Este ADR y todos sus resultados corresponden a las 14 variables originales. La selección vigente
tiene 15 por la incorporación manual de es_fin_de_anio (ADR 0015). El estudio se conserva sin cambios
y deberá revisarse mediante un estudio nuevo con identidad y almacenamiento distintos cuando se autorice.
No se ejecutaron nuevos trials ni entrenamientos en esta actualización.
