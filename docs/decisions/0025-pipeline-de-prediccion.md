# ADR 0025: pipeline de predicción sobre un Excel nuevo

- **Fecha:** 14/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna (Fase 4) pide `src/predict.py`, que corre sobre un Excel que no conocemos. Ese archivo tiene la misma estructura general que los de entrenamiento, pero puede traer hojas y columnas de más o de menos, y el script tiene que acceder por nombre y fallar con un mensaje que diga qué falta (un `KeyError` sin explicación no cumple la condición de aprobación). Hasta ahora el preprocesamiento exigía los tres Excel con todas sus hojas (`construir_tabla_completa`), las 262 columnas conservadas en la limpieza y las 10 variables base de las interacciones, aunque el modelo use pocas.

## Decisión
1. **Se guarda el `Pipeline` de sklearn completo con joblib** (imputación, escalado si corresponde, y modelo). El preprocesamiento anterior al `Pipeline` (leer el Excel, limpiar y crear variables) no va en el archivo del modelo: lo hace `preprocessing_pipeline.py` con las reglas de `data/processed/data_lineage.json`, el mismo código del entrenamiento.
2. **Se lee solo lo que el modelo usa.** Las columnas de entrada se toman del propio modelo (`feature_names_in_`), no de una lista fija: `resolver_dependencias` deduce de ahí qué hojas y mediciones hacen falta. Con las 14 variables del ADR 0015 son tres hojas (`Consolidado EE`, `Consolidado Agua` y `Totalizadores Energia`). Todo lo demás se ignora, esté o no.
3. **Mismo resultado que en entrenamiento.** `limpiar` acepta `columnas_requeridas`: cada columna se limpia por su cuenta, así que limpiar solo las necesarias da lo mismo para ellas. Un producto `A x B` depende solo de `A` y de `B`, por lo que calcular solo las interacciones pedidas da los mismos valores. El `dataset_final.csv` no cambió (misma huella SHA-256).
4. **Se predicen todos los días.** Los que tienen alguna variable vacía (los primeros días del archivo, sin historia para el rezago de 2 días y la media de 7; los días inválidos del contador) se predicen igual con la imputación del `Pipeline` y se listan por consola. El resultado no agrega columnas: el CSV tiene `fecha` y `prediccion_frio_kw`. El último día del archivo también se predice, porque para predecir D+1 solo hacen falta las variables de D.
5. **Errores.** `ErrorDeEntrada` (archivo, hoja o columna que falta) y `ErrorDeModelo` (registro o modelo) llevan un mensaje que nombra lo que falta. `main` los imprime en `stderr`, devuelve 1 y no escribe el CSV. Cualquier otra excepción también termina con un mensaje y código 1.
6. **Registro.** `models/model_registry.json` tiene una clave por versión (`"1.0.0"`) con al menos `path`. `latest` es la mayor versión comparada como números (1.10.0 es mayor que 1.9.0); también se acepta una versión exacta. Las rutas son relativas a la raíz del repositorio.
7. **Las funciones no imprimen ni preguntan.** Los avisos están solo en `main`, así otro programa puede llamar a `predict_consumption` y recibir la tabla.

## Justificación
- **Leer por nombre y solo lo necesario** reduce lo que puede fallar con un archivo distinto del de entrenamiento y hace más corto el mensaje cuando algo falta. En el Excel real de 482 días el preprocesamiento tardó 44 segundos, contra unos 10 minutos de la primera lectura de los tres archivos.
- **Tomar las columnas del modelo** evita que `predict.py` quede desactualizado si la Fase 3 elige un modelo con más variables (hay experimentos con el indicador de fin de año) o con otro conjunto.
- **Predecir todos los días** mantiene una fila por día del archivo, que es lo que pide la consigna. Descartar los días con historia incompleta cambiaría la cantidad de filas sin que quien corrige lo espere.
- **Comprobación con datos reales.** Sobre `Totalizadores Planta 2022-2023.xlsx`, las 14 columnas que arma `predict.py` coinciden con las de `dataset_final.csv` en las 470 filas comparables, salvo las 6 primeras del archivo, que no tienen la historia de los días anteriores (prueba `test_con_el_excel_real_reproduce_las_variables_de_dataset_final`).

## Alternativas descartadas
- **Reutilizar `construir_tabla_completa` y `transform_variables` tal cual:** exigen todas las hojas y columnas, y fallarían con un Excel al que le falte algo que el modelo no necesita.
- **Omitir los días con datos incompletos:** cambia la cantidad de filas y oculta el problema; se prefiere predecir y avisar.
- **Agregar una columna de aviso al CSV:** un script estricto de la cátedra puede esperar solo las dos columnas pedidas.
- **Guardar dentro del modelo también la limpieza y las variables:** obligaría a serializar reglas y funciones en el `.pkl`; el linaje ya las guarda en JSON.

## Limitaciones
- Las primeras predicciones de un archivo sin historia previa son menos confiables: usan la mediana de entrenamiento donde falta el dato.
- `predict.py` depende de que `data/processed/data_lineage.json` sea de la misma corrida que el modelo. Si se reentrenan las reglas, hay que regenerar los dos.
- Las fechas del Excel pueden repetirse (ADR 0002), así que `predicciones.csv` puede tener la misma `fecha` en dos filas.
- Se asume que el archivo nuevo es una sola planilla con las hojas `Consolidado EE`, `Consolidado Agua` y `Totalizadores Energia` (si el modelo final usa otras mediciones, se piden también sus hojas).
- Las pruebas con datos reales se saltean si los Excel no están en `data/`.
