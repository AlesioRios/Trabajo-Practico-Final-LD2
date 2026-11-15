# ADR 0019: búsqueda temporal de hiperparámetros de XGBoost con Optuna

- **Fecha:** 14/11/2026
- **Estado:** aceptada
- **Autor:** Juan Cruz Brown

## Contexto

La Fase 3 requiere optimización y seguimiento de experimentos. Este ADR documenta
los estudios vigentes `xgboost-temporal-15-features-v1` y
`xgboost-temporal-15-features-refinado-v1`, y conserva los antecedentes experimentales
archivados con sus identificadores, métricas y commits, sin reinterpretarlos como
resultados de los estudios vigentes.

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

El ADR 0018 define las funciones comunes, las filas comparables y cinco pliegues
cronológicos con una fila de separación. Se buscan configuraciones prometedoras
solo en entrenamiento; esta etapa no elige un modelo definitivo.

## Decisión

Usar Optuna con TPESampler(seed=42), dirección minimize y máximo TOTAL de 100 trials
secuenciales en la búsqueda inicial vigente; el refinamiento utiliza 60, como se
detalla en la sección de experimentos ejecutados.
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

Persistir los estudios vigentes por separado en
results/optuna/xgboost_15/estudio_15.sqlite3 y
results/optuna/xgboost_15_refinado/estudio_60.sqlite3. Las carpetas mutables se ignoran
para que no alteren el control Git de los experimentos siguientes; el control del
árbol excluye las escrituras de results/experiment_logs.csv. Guardar las copias
SQLite comprimidas, detalle JSON, CSV y resumen en results/xgboost_15/ y
results/xgboost_15_refinado/. La evidencia archivada en results/xgboost/ conserva su
identidad y sus resultados, sin reasignarlos a los estudios vigentes.
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
### Antecedente experimental archivado

Las tablas 6 a 9 y results/xgboost/resumen.json conservan el antecedente identificado
como xgboost-temporal-v1. Sus resultados no se atribuyen a los estudios vigentes: 100 trials,
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
El entrenamiento definitivo y la elección final siguen pendientes. Las evaluaciones
independientes de los estudios vigentes se detallan en la sección de resultados ejecutados.

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


## Experimentos vigentes ejecutados

Los estudios siguientes utilizan 15 características, incluido el indicador de fin
de año incorporado por la hipótesis descrita en Contexto. Los resultados se conservan
con sus identificadores y contratos, sin atribuir las diferencias entre búsquedas
exclusivamente a esa característica.

1. **Búsqueda inicial vigente.** El conjunto de 15 características incluye
   `fin_de_año` (`es_fin_de_anio`), según la decisión de ingeniería de características
   documentada en el ADR 0015. Estudio xgboost-temporal-15-features-v1: 100 trials, 47 completos,
   53 podados, cero fallidos; ganador 56, MAE CV 3957.959999884601. Mismo TPE(42),
   MedianPruner, cinco pliegues con gap=1 y MAE ponderado sobre filas comunes.
   Evidencia en results/xgboost_15/; ejecutor src/modelado_xgboost_15.py.
2. **Refinamiento.** Estudio xgboost-temporal-15-features-refinado-v1: 60 trials,
   45 completos, 15 podados, cero fallidos; ganador 49, MAE CV 3941.2233212356905.
   Se fijaron learning_rate=0.0020702351473759497 y max_depth=3 del ganador anterior.
   Rangos: n_estimators 1750–3000, min_child_weight 4–5 logarítmico, subsample
   0.87–0.97 y colsample_bytree 0.80–1.0. Sin early stopping. Rangos, top 10 y
   justificación en results/xgboost_15_refinado/ y src/modelado_xgboost_15_refinado.py.
3. **Validación independiente.** Cada ganador de 15 variables se ajustó solo con
   las 830 filas de entrenamiento y se midió en las mismas 171 filas comunes de
   validación (29/10/2022–26/04/2023). Primer estudio: MAE 2221.7606336805557,
   RMSE 2844.3074191916603, R² 0.4644113105021027. Refinado: MAE
   2229.2246293631215, RMSE 2853.130567574821, R² 0.4610833242874175.
   El refinado mejora CV pero empeora validación en 7.4639956825658. No se cambian
   hiperparámetros a partir de las curvas de validación. Se preservan ambos resultados.
4. **Curvas.** El refinado registra train y validación en eval_set solo para
   diagnóstico: sin early stopping, callbacks de selección ni truncamiento de árboles;
   se entrenaron los 1994 árboles fijados antes de evaluar. PNG y CSV por árbol en
   results/xgboost_15_refinado_validacion/.

No se guardó un modelo XGBoost serializado como artefacto de producción. Las
predicciones guardadas permiten verificar las métricas sin reentrenar. El registro
común conserva los commits y el estado Git reales de cada ejecución. No se evaluó
prueba ni se modificó model_registry.json. La elección y el reentrenamiento final
siguen pendientes de decisión del equipo.

La auditoría corrigió solo documentación, la base de la plantilla PR y la lectura
idempotente de la notebook propia: contratos de los artefactos archivados
explícitos, PNG y evaluaciones consultados sin generar nuevos ajustes.
Los módulos usados en las búsquedas no se tocaron para conservar sus hashes.
El texto con codificación antigua en esos módulos se preserva por esa trazabilidad;
no hay rutas personales ejecutables. Las salidas de la notebook no se reejecutaron.

## Auditoría del tracking exigido en 3.2

Se contrastaron los trials COMPLETE y las evaluaciones guardadas con el log:
139 registros: 137 configuraciones CV completas y 2 ajustes evaluados en validación. Todas las filas incluyen parámetros fijos
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

## Diagnóstico de predicciones archivadas (15/11/2026)

Se completa el diagnóstico de 3.3 con `src/diagnostico_xgboost.py`, sin importar
estimadores ni abrir el dataset. Se leen únicamente las predicciones y evaluaciones
guardadas de las dos búsquedas de 15 variables, los estudios exportados y el log.
Los originales se conservan por SHA-256; no hay nuevos entrenamientos ni registros
de experimentos. El código previo de búsqueda y evaluación no se modifica.

Se guardan cinco PNG en `results/xgboost_diagnostico_validacion/`: predicción frente
a real, serie temporal real/predicho, histograma de residuos, residuos frente a
predicción y residuos temporales. `resumen_residuos.csv`, `residuos_por_mes.csv` y
`errores_mayores.csv` fundamentan el informe. El residuo es real menos predicho.
Las secciones 13–16 del notebook propio leen esos resultados de forma autónoma.
`src/actualizar_diagnostico_notebook.py` ejecuta solo ese apéndice; las celdas
históricas conservan sus fuentes y salidas sin reejecución.

El sesgo global inicial es −79,93 kW y el refinado −53,41 kW, pero el desvío inicial
es 2851,53 kW. En enero el sesgo inicial es +1599,66 kW (29 filas), y en abril,
−1673,62 kW (22 filas). El mayor error inicial es 7672,62 kW, índice CSV 953,
emisión 02/03/2023. Los diez mayores concentran 16,89 % del error absoluto.
El refinado mejora CV en 16,74 kW pero empeora validación en 7,46 kW; no se
modifican hiperparámetros ni se interpreta esa diferencia como significativa.

Las representaciones decimales del CSV tienen un pequeño error de serialización:
recuperar las predicciones float32 reproduce las métricas registradas. Los
residuos usan los decimales originales y las métricas históricas se muestran
separadas. No se alteran ni corrigen retrospectivamente los resultados.

`auditoria_tracking.csv` contrasta 137 COMPLETE históricos y vigentes y dos
validaciones con sus 139 registros originales. No faltan esos experimentos y
coinciden parámetros, métricas, filas, semilla, commit y fecha. Sigue pendiente
la lectura literal del tracking por cada ajuste de 3.2: el CSV agrupa pliegues,
los podados quedan en Optuna/JSON y no existen timestamps por ajuste recuperables
sin inventarlos. No se fabrican filas para declarar cumplimiento total.

Se descarta entrenar o reconstruir un modelo para extraer importancias: no hay
estimador XGBoost serializado y la consigna exige feature importance del ganador
global, aún no elegido. El diagnóstico no prueba normalidad, independencia,
causas operativas ni generalización a prueba. Las fechas siguen siendo rótulos
de emisión por bloque y los meses parciales no son comparables como años completos.
El informe completo y los controles son la evidencia para la integración posterior.
