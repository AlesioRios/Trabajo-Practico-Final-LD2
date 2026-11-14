# ADR 0016: elección del pipeline de Ridge y guardado sin escalar

- **Fecha:** 13/11/2026 (actualizada el 14/11/2026 con las 14 variables definitivas)
- **Estado:** aceptada
- **Autor:** Miligonzalzz

## Contexto
La consigna (2.4) pide que la normalización y el escalado vivan dentro de un `Pipeline` de scikit-learn ajustado solo con entrenamiento, y que el archivo guardado en `data/processed/` quede sin escalar. El pipeline de la primera versión aplicaba imputación por vecinos y Yeo-Johnson a un conjunto amplio de variables, y su error era alto. Había que decidir qué tratamiento llevaba el `Pipeline`.

La comparación se hizo en dos momentos: primero con un conjunto provisional de 10 variables (`notebooks/pipeline_y_guardado.ipynb`) y, cuando la selección dejó las 14 variables definitivas (ADR 0015), se repitió con ellas. Las tablas que se citan con número `7.x` son las de la sección 7 de `notebooks/preprocesamiento.ipynb`. Ridge se usa como juez para comparar tratamientos; no es el modelo final, que se elige en la Fase 3.

## Decisión
1. **Se usa la variante A: imputación por mediana y `StandardScaler`**, ajustados solo con `X_train`. Es la que `armar_pipeline` devuelve por defecto (`VARIANTE_POR_DEFECTO = "A"`).
2. **`alpha = 10` queda como valor de referencia** (el de menor error promedio de A con las 14 variables). El valor definitivo lo vuelve a decidir la Fase 3 con el modelo final.
3. **El dataset final se guarda en sus unidades originales**, sin escalar, junto con su tramo, el objetivo, su huella y su linaje. La conexión y los detalles están en el ADR 0017.

| Variante | Tratamiento evaluado |
|---|---|
| A | Mediana y escalado estándar |
| B | Mediana y Yeo-Johnson estandarizado |
| C | Mediana con indicador de vacío y Yeo-Johnson estandarizado |
| D | Mediana, Yeo-Johnson y escalado robusto |
| E | Tratamiento por grupos (binarias, cíclicas, razones y numéricas) |
| F | Escalado estándar, KNN, Yeo-Johnson y escalado estándar |

## Justificación
- **La regla de elección.** Con las 14 variables, la variante de menor error promedio en los cinco pliegues cronológicos es C, con 3.334,15 kW y un desvío entre pliegues de 496,08 kW. El límite de un desvío es 3.830,23 kW. Las seis variantes quedan dentro (tabla 7.3: A 3.381,04; B 3.339,25; D 3.341,48; E 3.335,45; F 3.381,31). La regla no discrimina entre ellas: las diferencias de promedio (menos de 50 kW) son mucho menores que el desvío entre pliegues (de 482 a 756 kW). Se elige A por ser la que tiene menos pasos. Con las 10 variables provisionales el resultado fue el mismo: E tuvo el menor promedio (3.319,33 kW, desvío 242,96, límite 3.562,29) y A (3.407,63 kW) quedó dentro.
- **Se aparta del ADR 0010**, que preveía Yeo-Johnson antes de escalar. Las variantes con Yeo-Johnson (B, C, D, E) no mejoran de manera apreciable el error de A, y agregan pasos y parámetros que hay que guardar para `predict.py`.
- **KNN (variante F) no aporta.** Con las 14 variables da 3.381,31 kW, igual que A (3.381,04), con más pasos y más tiempo de cálculo.
- **El error varía según el tramo (tabla 7.4).** Cada pliegue cubre otra época, con otro consumo medio (de 18.322 a 32.349 kW) y otra variabilidad, y entrena con más historia que el anterior. El pliegue 1 es el peor para A (4.637,29 kW contra 2.576,87 de persistencia y 2.858,93 de media móvil): entrena con 139 filas y hay 14 variables. Por eso los errores solo se comparan dentro de las mismas filas.
- **Frente a los baselines (tabla 7.5).** En los cinco pliegues, el promedio de A (3.381,04 kW) no supera a los baselines (persistencia 3.249,53; media móvil 3.376,83): es 4 % peor que el mejor. En validación, que solo informa y no se usó para elegir, A obtiene 2.292,52 kW contra 2.668,33 de persistencia y 2.497,65 de media móvil, en 171 fechas comunes. La mejora sobre el mejor baseline es del 8,2 %, menor al 10 % que la consigna pide interpretar. Con honestidad: con un modelo lineal y estas variables la mejora es moderada, y lo que se compra con el `Pipeline` es un tratamiento de vacíos y escalas que cualquier modelo posterior puede reutilizar. Si el modelo final de la Fase 3 también queda cerca de los baselines, habrá que decirlo.

## Alternativas descartadas
- **B, C, D y E:** dentro del límite, pero con más pasos y sin una mejora que supere el desvío entre pliegues.
- **F (KNN con dos escalados):** mismo error que A con más cálculo.
- **Interpolar los vacíos, incluso en las cíclicas:** las cíclicas (`dia_anio_seno` y `dia_anio_coseno`) se calculan desde la fecha y no tienen vacíos; en las demás columnas, interpolar usa el valor del día siguiente, que es información del futuro.
- **Guardar el dataset ya escalado:** la consigna lo prohíbe para no dejar información del entrenamiento (ni del test) dentro de un artefacto versionado.

## Limitaciones
- Elegir el `alpha` con los mismos pliegues que se usan para comparar deja una estimación algo optimista en todas las variantes.
- El MAE de validación de A (2.292,52 kW) no se puede comparar con el de la selección (2.247,82 kW): aquella usó `alpha = 100` y agregaba indicadores de vacío, y esta no (ADR 0015).
- Ridge es solo el juez: los modelos de árboles no necesitan escalado y aceptan vacíos, así que la Fase 3 decide el preprocesamiento de cada modelo.
- La validación ya se usó para elegir el método de selección, de modo que no es una estimación independiente del rendimiento final.
