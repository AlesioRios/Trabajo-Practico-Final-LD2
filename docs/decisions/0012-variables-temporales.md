# ADR 0012: variables temporales que se crean y cuáles se descartan

- **Fecha:** 09/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna (2.2) pide crear variables temporales (rezagos, promedios móviles, calendario con codificación cíclica) y justificar con el número las que se descartan. Había que decidir cuáles se construyen, con qué parámetros y sobre qué fecha, sabiendo que la columna `DIA` tiene fechas repetidas y faltantes (ADR 0002).

Toda la evidencia sale de `notebooks/variables_temporales.ipynb`, medida solo con `X_train` (848 días). Las secciones que se citan abajo son las de ese notebook.

## Decisión
Se crean diez variables con `crear_variables_temporales` (`src/variables_temporales.py`), sin transformar ni escalar (ADR 0010):

- **Del frío (hoy y anteriores):** frío de ayer, de hace 2 y de hace 3 días; promedio y desvío estándar de los últimos 7 días (hoy y los 6 anteriores), con un mínimo de 3 días con dato, es decir, tolera hasta 4 faltantes; y la diferencia entre hoy y ayer.
- **Del calendario del día que se predice** (la fecha de hoy más un día): seno y coseno del día del año, indicador de fin de año (del 24 de diciembre al 6 de enero, ambos incluidos) e indicador de fin de semana.

Los rezagos y las ventanas se calculan **por fila**, no por fecha. El indicador de fin de semana se construye solo para probarlo y se decide en la selección.

Se descartan, sin construirlas: los rezagos de 7, 30 y 365 días, la variable de año y el mes como número.

## Justificación
- **Fechas (sección 2).** La tabla tiene 1.212 filas y 1.190 fechas distintas (16 fechas se repiten, 22 filas de más) y faltan 23 fechas del calendario, siempre las mismas de fin de mes (31/03, 31/05, 31/10, 31/12 y del 29 al 31 de agosto). Un año completo tiene tantas filas como días (365 en 2021 y en 2022). La fecha rotulada se desvía de "una fila = un día consecutivo" entre 0 y 3 días, con un ciclo mensual idéntico en 2021 y 2022. Por eso los rezagos van por fila, y el calendario podría tener un error de pocos días, que en 2021 y 2022 se repite igual cada año.
- **Rezagos de 1, 2 y 3 días (sección 3).** La correlación de Spearman con el frío de mañana es 0,847 para el frío de hoy y 0,752, 0,731 y 0,728 para los rezagos de 1, 2 y 3 días. Traen información parecida y la selección decide con cuáles se queda.
- **Rezagos de 7, 30 y 365 días descartados.** El de 7 días (0,732) se parece al frío de la semana en general, que resume mejor el promedio móvil de 7 días. El de 30 días (0,586) es una fracción del ciclo estacional. El de 365 días (0,646) repite la estación, pero deja 462 de 817 días utilizables y exigiría un año de historia previa en `predict.py`.
- **Promedio y desvío móviles, y diferencia con ayer.** El promedio de 7 días suaviza el ruido y sigue el nivel reciente (Spearman 0,834 con el frío de mañana); el desvío mide qué tan inestable viene el consumo y la diferencia con ayer capta subidas y bajadas bruscas (0,118 y 0,177). La tolerancia de 4 faltantes evita perder filas por los 25 días inválidos.
- **Seno y coseno del día del año (sección 4).** El mes explica el 68,0 % de la variación del frío en entrenamiento. Con el seno y el coseno, diciembre y enero quedan cerca (el 31/12 y el 01/01 tienen valores casi iguales). Se usa el día del año y no el mes porque el mes salta el día 1, y se calcula sobre el día que se predice porque ese dato se conoce hoy y `predict.py` lo puede obtener aunque la fila de mañana no exista.
- **Fin de año (sección 5).** La mediana del frío dentro de la ventana es 9.751 kW (25 días) contra 13.089 kW en el resto de diciembre y enero (96 días) y 27.019 kW en el resto del año. El seno y el coseno son una curva suave y no pueden representar una caída tan concentrada.
- **Variable de año descartada (sección 6).** El año tal cual explica 0,031 % de la variación y descontada la mediana de su mes, 5,1 %. Ese 5 % es un cambio de nivel en 2022 (casi todos los meses quedan por debajo de los de 2021), no una tendencia en un solo sentido: entre 2020 y 2021 algunos meses suben y otros bajan. Con un solo año completo en entrenamiento y un año de prueba (2023) que el modelo nunca vio, un modelo de árboles no puede extrapolar y uno lineal extrapolaría una tendencia indistinguible de una coincidencia. El nivel lo siguen el promedio móvil y los rezagos.
- **Fin de semana (sección 6).** El día de la semana explica el 0,82 % de la variación con la fecha rotulada y el 1,34 % con la fecha que daría la posición de la fila; la conclusión no depende de cuál fecha sea la correcta. Se construye el indicador para probarlo y, si la selección lo descarta, queda descartado con ese número.

## Alternativas descartadas
- **Rezagos semanal, mensual y anual:** ver arriba; no aportan sobre el promedio móvil y el calendario, y los largos pierden filas.
- **Mes como número (1 a 12):** diciembre y enero quedarían lejos, y salta el día 1.
- **Variable de año:** no se puede extrapolar al año de la prueba y no hay una tendencia sostenida (ver arriba).
- **Calendario del día de hoy en lugar del de mañana:** el objetivo es el frío de mañana, y su calendario ya se conoce.
- **Reasignar fechas consecutivas según la posición de la fila:** obliga a inventar fechas sin poder comprobarlas (ADR 0002).
- **Ventana de fin de año más estrecha o más ancha:** hay solo dos fines de año en entrenamiento y los bordes tienen poca evidencia; se prefirió una ventana simple y fácil de explicar.

## Limitaciones
- La ventana de fin de año sale de dos años de datos, y hay otra caída previa (alrededor del 19 al 22 de diciembre) que queda fuera de ella y solo la capta el seno y el coseno.
- Las fechas rotuladas podrían estar corridas entre 0 y 3 días respecto de la posición de la fila (ADR 0002). Se debería aclarar este inconveniente en la defensa del tp. El desfase parece sistemático: en 2021 y 2022 es idéntico mes por mes, pero en septiembre y octubre de 2023 (el año de prueba) difiere en un día, así que no se puede asegurar que se mantenga.
- Los vacíos de las variables del frío (por los días inválidos y los primeros días de la tabla) no se tratan acá: se imputan dentro del `Pipeline`, ajustado solo con entrenamiento.
- Las correlaciones de esta decisión son descriptivas y no reemplazan la selección de variables, que usa dos métodos con `TimeSeriesSplit`.
