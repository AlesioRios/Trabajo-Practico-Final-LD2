# ADR 0020: Random Forest, Ridge y Lasso para el consumo de frío

- **Fecha:** 14/11/2026
- **Estado:** propuesta
- **Autor:** Milagros González

## Contexto

La Fase 3 exige comparar modelos para el consumo de frío del día siguiente con la persistencia y la media móvil. Usamos las variables del ADR 0015 y el protocolo del ADR 0018. La tabla 1 de `notebooks/modelado_rf_lineales.ipynb` muestra las 14 entradas y las filas comunes de cada tramo. Las referencias a tablas de este ADR corresponden a ese notebook.

## Decisión

Aportar Lasso con `alpha=75` y el pipeline A como candidato de menor MAE en validación de esta rama. Conservar Ridge y Random Forest en el registro para la comparación del equipo. El pipeline de los lineales aprende medianas y escalado estándar dentro de cada ajuste. Random Forest utiliza las entradas originales, incluidos sus faltantes. El objetivo queda en kW. La tabla 4 muestra los parámetros elegidos y la tabla 6 compara el error de sus predicciones.

## Justificación

El espacio de búsqueda está en la tabla 2. Se registraron 20 configuraciones de Random Forest, 10 de Ridge y 10 de Lasso (tablas 3 y 9). Para el bosque se incluyó una referencia y se muestreó con semilla fija sobre la cantidad de árboles, la profundidad máxima, el mínimo de filas por hoja y la fracción de variables candidatas por división. En los lineales se recorrieron distintos valores de alfa. Se eligió una configuración por modelo con el menor MAE agregado de los cinco pliegues cronológicos y se midió cada elegida en validación (tablas 4, 5 y 6).

Sobre las 171 filas comunes de validación, Lasso obtuvo 2.215,80 kW, Ridge 2.225,44 kW y Random Forest 2.293,70 kW. Los tres quedaron por debajo de la persistencia (2.668,33 kW) y de la media móvil (2.497,65 kW). Lasso mejoró un 11,28 % frente a la media móvil y Ridge un 10,90 %. La ventaja de Lasso sobre Ridge fue de 9,64 kW (tabla 6).

En los pliegues, Lasso tuvo el menor MAE agregado entre los modelos entrenados, 3.290,07 kW. Superó a los dos baselines en los pliegues 2 a 5, pero en el primero obtuvo 4.289,27 kW frente a los 2.576,87 kW de persistencia. En el agregado de los pliegues la persistencia quedó primera con 3.247,71 kW (tabla 5).

Al permutar las entradas del Lasso ajustado, los mayores aumentos medios de MAE fueron los del consumo de frío actual y la energía del transformador 9. En Random Forest el mayor aumento fue el del coseno del día del año (tabla 8). Esta medición se hizo para interpretar las predicciones, con las entradas y parámetros ya fijados.

## Alternativas descartadas

- **Random Forest como candidato principal de esta rama.** Su mejora de 8,17 % frente a la media móvil quedó por debajo del 10 % en validación (tabla 6). Su primer pliegue tuvo un MAE de 6.916,02 kW (tabla 5).
- **Ridge como candidato principal de esta rama.** Quedó cerca de Lasso en validación y su MAE cruzado fue mayor (tablas 5 y 6). Se mantiene disponible para la comparación general.
- **Transformación Yeo-Johnson.** Se mantuvo la variante A del ADR 0016 para concentrar esta búsqueda en la regularización. El efecto de esa transformación en estos modelos queda sin medir.

## Limitaciones

La validación ya se utilizó para seleccionar variables y el tramo oculto corresponde a otra época. El mejor MAE de validación no mide el rendimiento en el test de la cátedra. Los dos modelos analizados sobreestimaron en promedio en el cuartil de menor consumo y subestimaron en el de mayor consumo. En Lasso los residuos medios fueron −2.204,08 kW y +1.447,81 kW en esos cuartiles (tabla 7). El error del primer pliegue sugiere que conviene tener en cuenta qué tan representativo es el historial de ajuste para cada período (tabla 5).
