# ADR 0017: conexión de la selección, el Pipeline y el guardado, y contenido del dataset final

- **Fecha:** 14/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna (2.4) pide que el preprocesamiento deje guardado `data/processed/dataset_final.csv` sin escalar, con su huella en `data/checksums.json`, un registro de linaje y las dimensiones de `X_train`, `X_val` y `X_test`, con un `Pipeline` ajustado solo con entrenamiento. Las piezas ya existían por separado: las reglas de limpieza y de variables (ADR 0008 a 0014), la selección de 14 variables (ADR 0015) y el `Pipeline` (ADR 0016). Faltaba conectarlas en el punto de entrada y decidir qué se guarda.

Las tablas que se citan con número `7.x` son las de la sección 7 de `notebooks/preprocesamiento.ipynb`.

## Decisión
1. **`completar_fase2(resultado, carpeta_salida)`** vive en `src/preprocessing_pipeline.py` y la llama `main`. Recibe el diccionario que devuelve `ejecutar_preprocesamiento` y escribe los archivos en `data/processed/`. Se ejecuta con `python src/preprocessing_pipeline.py`; con `--sin-guardar` hace las comprobaciones sin escribir nada. `pipeline_sklearn` y `seleccion_de_variables` se importan dentro de la función para evitar un import circular (`pipeline_sklearn` ya importa este módulo).
2. **Se quitan dos filas de frontera.** El objetivo de una fila es el consumo del día siguiente: en la última fila de entrenamiento ese día ya es de validación, y en la última de validación es de prueba. Esas dos etiquetas se dejan vacías y las filas no entran al dataset final (tabla 7.6).
3. **El `Pipeline` se ajusta solo con `X_train`** (variante A, ADR 0016). A `X_val` y `X_test` se les aplica la transformación aprendida con entrenamiento solo para comprobar que corre y registrar sus dimensiones (tabla 7.7: 830 × 14, 176 × 14 y 178 × 14, iguales antes y después porque la variante A no agrega columnas). Nada de lo transformado se guarda.
4. **Contenido de `dataset_final.csv`:** `DIA`, las 14 variables seleccionadas, `consumo_dia`, `objetivo` y `tramo`, en las unidades originales y con los vacíos que tengan. Son 1.184 filas: 830 de entrenamiento, 176 de validación y 178 de prueba.
5. **`consumo_dia` es una columna auxiliar, no una entrada del modelo.** Vale lo mismo que `Frio (Kw)` fila por fila (tabla 7.8), porque el consumo de hoy es una de las 14 variables. Se conserva aparte porque la Fase 3 la necesita para recalcular los baselines (persistencia y media móvil) sobre el mismo conjunto que el modelo final; no se agrega por la selección de variables, y no depende de que `Frio (Kw)` siga entre las elegidas. Como no forma parte de `X`, no introduce colinealidad en el modelo; el linaje la registra como auxiliar para que la Fase 3 la excluya de las entradas.
6. **`data_lineage.json`** guarda las reglas de limpieza, de variables y de grupos (todas serializables a JSON), la lista de columnas y de auxiliares, la variante y el `alpha` de referencia, las dimensiones antes y después del `Pipeline` y la cantidad de filas de frontera quitadas. Con eso `predict.py` (Fase 4) puede reaplicar las mismas reglas sobre un Excel nuevo. La huella SHA-256 del CSV se suma a `data/checksums.json` sin tocar las de los Excel.

## Justificación
- **La frontera.** Dejar la etiqueta de la última fila de entrenamiento haría que el modelo aprenda con un valor que pertenece a validación, y lo mismo entre validación y prueba. Quitar dos filas de más de mil (0,2 %) no afecta lo que el modelo aprende, y cumple a rajatabla la separación temporal (tabla 7.6: la etiqueta de la fila 847 era 17.342 kW, el consumo del primer día de validación).
- **Aplicar la transformación a `X_test`.** El `Pipeline` aprendido con entrenamiento se aplica igual a validación, prueba y datos nuevos, que es lo que hará `predict.py`; no hay fuga porque ninguna estadística sale de esos tramos. La regla de la consigna ("el conjunto de prueba se toca una sola vez") se refiere a elegir o medir el modelo con sus etiquetas, y acá no se usan ni etiquetas ni errores de prueba.
- **Guardar sin escalar.** El escalado ocurre dentro del `Pipeline`, cada vez que se entrena o se predice; el CSV no crea ninguna diferencia entre entrenamiento y prueba. Además, la consigna lo exige para no dejar información de otros tramos dentro del artefacto.
- **Controles (tabla 7.8).** Los ocho controles dan verdadero: el CSV está en unidades originales, tiene exactamente dos filas menos que las filas con objetivo de la tabla, los tres tramos no se solapan, `consumo_dia` coincide con `Frio (Kw)`, la huella coincide con `checksums.json`, las dimensiones del linaje coinciden con las de la tabla 7.7, el imputador usa la mediana de entrenamiento y la variante y el `alpha` registrados son los elegidos.

## Alternativas descartadas
- **Conservar las dos filas de frontera:** se pierde una fila por tramo, pero queda una etiqueta que mira al tramo siguiente.
- **Dejar solo las 14 variables, sin `consumo_dia`:** obligaría a la Fase 3 a depender de que `Frio (Kw)` siga siendo parte de la selección.
- **Poner `completar_fase2` en un módulo nuevo:** evitaba el import circular, pero separaba el cierre del punto de entrada que ya estaba definido.
- **No transformar `X_test` ni siquiera para comprobar:** no habría forma de saber, antes de la Fase 3, que el `Pipeline` corre sobre el tramo de prueba.

## Limitaciones
- La columna `DIA` **no es única** en la tabla (se comprobó al cruzar por fecha: una fecha puede coincidir con más de una fila; ver ADR 0002 sobre el calendario de los bloques), así que cruzar por fecha da coincidencias ambiguas. Cualquier comprobación o unión posterior debe usar la posición de la fila, no `DIA` como clave.
- El CSV conserva los vacíos (hasta el 4,25 % en una variable). Los modelos que no los aceptan (Ridge, Lasso) necesitan imputar; los de árboles pueden usar los vacíos tal cual.
- Sigue pendiente lo que dejó el ADR 0015: no se certificó que todas las mediciones estén disponibles a las 23:59 al emitir la predicción, ni la correspondencia entre las fechas etiquetadas y el calendario físico.
- El `Pipeline` que se arma en esta fase solo verifica la transformación: no se guarda. Se ajusta y se guarda junto con el modelo en la Fase 3.
