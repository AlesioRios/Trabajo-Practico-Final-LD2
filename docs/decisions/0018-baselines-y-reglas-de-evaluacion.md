# ADR 0018: baselines y reglas de evaluación y registro de la Fase 3

- **Fecha:** 14/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna (3.1 a 3.4) exige medir dos baselines (persistencia y media móvil de 7 días) con el mismo MAE, sobre el mismo conjunto y con el mismo preprocesamiento que el modelo final; entrenar al menos cuatro modelos; registrar cada ejecución en `results/experiment_logs.csv`; elegir el modelo por su MAE de validación y reentrenarlo con entrenamiento más validación. Elegir el modelo mirando el tramo de prueba se penaliza (−15). Como varias personas entrenan modelos distintos en ramas distintas, hacía falta fijar reglas comunes antes de empezar: si cada modelo se evalúa en otras filas o con otros pliegues, sus errores no se pueden comparar.

Toda cifra de este ADR sale de las tablas 1.1 a 1.5 de `notebooks/modelado.ipynb`.

## Decisión
1. **Baselines.** La persistencia es el consumo de hoy (`consumo_dia`) y la media móvil es el promedio de las últimas 7 filas, con al menos 4 días válidos (`baseline_media_movil`). Las dos columnas viajan en `dataset_final.csv` como auxiliares (ADR 0017): no son entradas de ningún modelo.
2. **Filas comunes.** Todo modelo se mide solo sobre las filas del tramo que tienen objetivo y las dos predicciones de los baselines (`filas_comunes`). Ahí se comparan modelo y baselines.
3. **Tramos.** Los modelos se ajustan y se comparan con entrenamiento y validación. El tramo de prueba no se carga por defecto (`cargar_dataset_final` exige `incluir_prueba=True`) y no se puede registrar sin `evaluacion_final=True`: se mide una sola vez, al final, con el modelo ya elegido.
4. **Ajuste de hiperparámetros** solo con los pliegues de `validacion_cruzada_temporal` (`TimeSeriesSplit` de 5 pliegues con `gap=1`) sobre el entrenamiento, y **elección del modelo** por su MAE en validación sobre las filas comunes. Luego se reentrena solo el ganador con entrenamiento y validación.
5. **Registro.** Cada ejecución se registra con `registrar_experimento` en `results/experiment_logs.csv`: fecha y hora, modelo, tramo, cantidad de filas, hiperparámetros, MAE, RMSE, R², semilla (`SEMILLA = 42`), hash del commit y si el árbol de trabajo tenía cambios sin commitear. El archivo solo crece: no se pisa ni se reordena.
6. **Preprocesamiento por modelo.** El dataset se guarda sin escalar; cada modelo arma su propio preprocesamiento y lo ajusta solo con entrenamiento. Ridge o Lasso usan imputación y escalado (variante A del ADR 0016); los modelos de árboles aceptan los vacíos y no necesitan escalado.
7. **Variables.** Todos usan las mismas 14 variables del ADR 0015.

## Justificación
- **Los baselines no necesitan validación para ajustarse, pero sí para compararse.** Como no tienen parámetros, no aprenden de ningún tramo. Se miden en entrenamiento y validación porque el modelo se compara contra ellos en esas mismas filas.
- **Filas comunes (tabla 1.2).** En entrenamiento hay 830 filas y 813 comunes (14 sin persistencia por día inválido, 3 sin media móvil). En validación hay 176 y 171 comunes (5 sin persistencia). Las 171 coinciden con las que se usaron en la Fase 2 para comparar los pipelines.
- **Referencias (tabla 1.3).** En entrenamiento, la persistencia da un MAE de 3.185,88 kW y la media móvil 3.293,61 kW. En validación, 2.668,33 y 2.497,65 kW. No se pueden comparar entre tramos: cada uno cubre otra época, con otro nivel y variabilidad del consumo.
- **Los pliegues también difieren (tabla 1.4).** El error de la persistencia va de 2.576,87 a 3.974,88 kW y el de la media móvil de 2.858,93 a 3.805,95 kW según el pliegue. Por eso un modelo se compara pliegue por pliegue con los baselines, no con un número aislado.
- **Qué significa agregar valor.** El mejor baseline en validación es la media móvil (2.497,65 kW); una mejora del 10 % exige quedar por debajo de 2.247,89 kW. Si el modelo final queda entre ese valor y el baseline, la mejora es menor al 10 % y la consigna pide decir con honestidad qué se compra con la complejidad.
- **Prueba reservada.** Obligar a pedir la prueba de forma explícita deja la intención en el código y evita elegir un modelo por descuido mirando ese tramo.

## Alternativas descartadas
- **Calcular los baselines desde los Excel cada vez:** obliga a quien entrena a tener los Excel y a esperar la tabla, y hace posible que cada uno use otras filas. La media móvil necesita los días anteriores que el CSV ya no tiene, así que se calcula una vez, con la tabla completa, y se guarda.
- **Pliegues distintos por modelo:** los errores no serían comparables.
- **Evaluar cada modelo en todas las filas con objetivo:** compararía modelos y baselines sobre días distintos.
- **Un único notebook compartido por todos:** los `.ipynb` son JSON y dos personas que lo editan a la vez generan conflictos; cada modelo va en su notebook y rama, y `modelado.ipynb` se consolida al final.

## Limitaciones
- La validación ya se usó para elegir el método de selección de variables (ADR 0015): el error de un modelo en validación es algo optimista, y elegir el modelo con ese mismo tramo suma otro poco.
- Los baselines de entrenamiento y de validación no son comparables entre sí.
- El test oculto de la cátedra (2024 y 2025) es de otra época que los datos con los que trabajamos: lo que se mida en validación no garantiza el MAE menor a 3.000 que exige la consigna sobre ese test.
- Las filas de baselines del log se registraron con cambios sin commitear en el árbol de trabajo (la columna `arbol_sin_commitear` lo indica); el hash que figura es el del commit anterior.
- La columna `DIA` no es única (ADR 0017): los índices de los tramos son los de las filas del CSV.
