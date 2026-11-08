# ADR 0011: no se incorporan datos de temperatura (Open-Meteo)

- **Fecha:** 08/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna, en sus recomendaciones de ingeniería de características, dice que el consumo de frío depende de la temperatura ambiente ("implícita en estacionalidad") y que, para obtenerla, "se recomienda Open-Meteo" (la planta está en Tecate, Baja California). Es una recomendación, no una exigencia. Había que decidir si agregar una columna de temperatura diaria a la tabla.

La evidencia sale de una **prueba exploratoria reproducible en la sección 6 de `notebooks/preprocesamiento.ipynb`**, hecha solo con `X_train` (848 días) y la temperatura media diaria de Tecate (coordenadas aproximadas 32,5734 N y 116,6254 O) de la API de archivo de Open-Meteo:
- La correlación de Spearman entre `Frio (Kw)` y la temperatura media es 0,53.
- Si a cada serie se le resta el promedio de su mes, la correlación entre los desvíos baja a **−0,01**: dentro de un mismo mes, los días más calientes no consumen más frío. La temperatura se relaciona con el frío casi solo a través de la estación.
- Al predecir el frío del día siguiente con dos modelos simples (rezagos, promedio de 7 días y estacionalidad), validados con 5 tramos de `TimeSeriesSplit` dentro de entrenamiento y sobre 798 días, agregar la temperatura de hoy baja el MAE de 3.215 a 3.179 kW con Ridge (−1,1 %) y de 4.382 a 4.216 kW con gradient boosting (−3,8 %). Agregar además la temperatura real de mañana lo baja a 3.138 con Ridge y no mejora con gradient boosting (4.225). Para comparar, repetir el frío de hoy (persistencia) da 3.189 kW sobre los mismos días.
- La prueba de desfases (temperatura de −3 a +3 días) no tiene un pico en 0 (la correlación sube de 0,52 a 0,55), porque la estación domina las dos series; no sirve para confirmar ni descartar que las fechas de los Excel estén alineadas.

## Decisión
**No se incorporan datos de Open-Meteo ni de ninguna otra fuente externa.** La estacionalidad se captura con las variables de calendario de la ingeniería de características (mes y seno y coseno del día del año).

## Justificación
- **El aporte es pequeño.** Dentro de cada mes la temperatura no explica el frío (−0,01), y en las pruebas rápidas la mejora del error es de 1 a 4 %, del orden de la diferencia entre un modelo y otro. No es lo bastante grande para justificar los costos de abajo.
- **Dependencia de internet.** `predict.py` tendría que descargar la temperatura de las fechas del archivo de test (2024-2025). Si no hay conexión, o la API cambia o limita las consultas, falla, y la consigna exige que corra sin errores en otra máquina (condición eliminatoria). Un sistema que no depende de la red es más robusto y reproducible.
- **Versionado de los datos.** Habría que descargar, guardar y versionar la temperatura de cada período (con su checksum y su registro en el linaje de datos) y actualizarla cada vez que llegue un archivo con fechas nuevas.
- **Riesgo de desfase de días.** Las fechas de los Excel pueden estar corridas (ADR 0002) y nuestras decisiones de armado de la tabla (un día por bloque) podrían introducir un desfase respecto del calendario real; unir una variable externa por fecha podría quedar desalineada sin que nos demos cuenta, y la prueba de desfases no permite descartarlo.
- **Es consistente con la consigna**, que presenta la temperatura como "implícita en estacionalidad" y recomienda Open-Meteo sin exigirlo.

## Alternativas descartadas
- **Descargar la temperatura histórica una vez y versionarla en `data/`:** resuelve la dependencia de internet para el entrenamiento, pero `predict.py` igual necesita temperatura de fechas nuevas.
- **Usar el pronóstico de mañana en la predicción:** en el entrenamiento habría que usar la temperatura real como sustituto del pronóstico, y esa diferencia no se puede medir con los datos disponibles.
- **Pedir una variable de temperatura a la cátedra:** la consigna no la provee y no se podría asumir en `predict.py`.

## Limitaciones
- La prueba es exploratoria: modelos simples, pocos días por tramo de validación y errores por encima de la persistencia en gradient boosting; sirve para comparar con y sin temperatura, no como resultado del modelo final.
- Descargarla requiere internet una vez (el notebook guarda una copia en `data/interim/`, que no se versiona) y la celda avisa si no hay conexión.
- Con otros modelos y más datos podría aparecer un efecto distinto. Si más adelante el error del modelo final se concentrara en días de calor extremo, convendría revisar esta decisión.
