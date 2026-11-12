# ADR 0014: variables base para las interacciones y `PolynomialFeatures` solo con productos

- **Fecha:** 12/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna (2.2) sugiere crear interacciones entre variables principales. `PolynomialFeatures` multiplica de a pares todas las variables que recibe, así que aplicarlo a las 278 candidatas crearía decenas de miles de columnas con solo 848 días de entrenamiento. Había que decidir sobre qué variables se aplica y qué combinaciones se generan.

Toda la evidencia sale de `notebooks/variables_interacciones.ipynb`, medida solo con `X_train`. Las tablas que se citan son las de ese notebook.

## Decisión
1. **Se aplican las interacciones sobre 10 variables base**, elegidas con dos criterios a la vez: que se parezcan al frío de mañana y que no se parezcan demasiado entre sí. Ambos se miden con la correlación de Spearman y solo con entrenamiento. Dos variables con |Spearman| de 0,8 o más entre sí se consideran redundantes.
2. **El método elegido es el voraz** (`elegir_voraz`, `src/variables_interacciones.py`): se recorren las variables de mayor a menor parecido con el frío de mañana y cada una se agrega solo si su |Spearman| con todas las ya elegidas es menor a 0,8. La lista resultante (tabla 6) es:

| # | Variable base | Spearman con el frío de mañana |
|---|---|---|
| 1 | `Frio (Kw)` | 0,847 |
| 2 | `dia_anio_coseno` | −0,801 |
| 3 | `frio_hace_2_dias` | 0,731 |
| 4 | `Consolidado KPI \| ET Bodega/Hl` | −0,722 |
| 5 | `Consolidado Agua \| Agua Planta (Hl)` | 0,701 |
| 6 | `Totalizadores Energia \| KW Cond 5. 6 y 9` | 0,685 |
| 7 | `Totalizadores Energia \| KW Mycom 7` | 0,634 |
| 8 | `Totalizadores Energia \| KW Trafo 9` | 0,600 |
| 9 | `Consolidado GasVapor \| ET Servicios (Mj)` | −0,600 |
| 10 | `Totalizadores Energia \| KW Mycom 2` | 0,595 |

3. **`PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)`** sobre esas 10 variables: agrega 45 columnas, una por cada par. Con interaction only en True ignoramos los cuadrados.
4. La lista de variables base se aprende solo con entrenamiento y queda guardada como una regla aprendida, igual que las reglas de limpieza y de planta. Se aplica igual a validación, prueba y datos nuevos; no se recalcula con datos nuevos.

## Justificación
- **Candidatas (tabla 2).** 278 variables: 262 columnas originales limpias, 10 temporales y 6 razones de planta. Se excluyen las columnas de control, el texto y las constantes.
- **Por qué no alcanza el ranking solo (tabla 3).** Entre las más parecidas al frío de mañana están el frío de hoy, su promedio de siete días, su rezago de un día y varios consumos eléctricos de la misma planta, que suben y bajan juntos. Elegir solo por el ranking dejaría casi la misma información repetida y las interacciones entre ellas no aportarían nada nuevo.
- **Estructura de redundancia (tabla 4).** El agrupamiento jerárquico con enlace completo sobre 1 − |Spearman| (todas las 278 variables, corte en 0,8) da 190 grupos: 133 con una sola variable y el mayor con 6.
- **Comparación de los dos métodos (tablas 7 y 8).** El método 1 (la mejor variable de cada grupo, tabla 5) y el método 2 (el voraz) coinciden en 7 de 10 variables. En la lista del método 1, el mayor |Spearman| entre dos de sus variables es 0,898 (el promedio de siete días con el frío de ayer); en la del método 2 es 0,797. El enlace completo garantiza redundancia dentro de un grupo, pero no entre variables de grupos distintos; por eso el método 1 deja pares sobre el umbral y el método 2, no.
- **Por qué solo productos.** La consigna pide interacciones entre variables, y un cuadrado es la curvatura de una sola variable. Con 848 filas, 55 columnas nuevas (45 productos y 10 cuadrados) aumentan la chance de que alguna parezca buena por casualidad en la selección.

## Alternativas descartadas
- **Las 10 de mayor Spearman sin más:** variables casi repetidas; sus productos no agregan información.
- **Método 1 (mejor de cada grupo):** deja pares con |Spearman| mayor a 0,8 (tabla 8). Se conserva en el notebook para mostrar los grupos y respaldar la comparación.
- **`interaction_only=False`:** agrega los 10 cuadrados. Si más adelante el modelo lineal sugiere curvatura, se puede probar como un experimento aparte y comparar con y sin ellos.
- **Interacciones sobre todas las candidatas:** del orden de decenas de miles de columnas con 848 días.
- **Reducción de dimensionalidad (PCA, autoencoders):** la consigna no los pide, y mezclan las variables en componentes que no se pueden leer, lo que choca con documentar la importancia de variables del modelo final (consigna, Recomendaciones, punto 3). Quedan como técnica evaluada y no usada.

## Limitaciones
- El umbral de 0,8 es una elección simple y fácil de explicar, no un valor óptimo. No se probaron otros.
- Spearman mide relaciones que van en un solo sentido (suben juntas o una sube cuando la otra baja). Relaciones más complejas, como una en forma de U, no se vería.
- Las correlaciones se calculan con los días en que ambas variables tienen valor, y un par sin días en común queda con correlación 0.
- Solo se multiplican las 10 variables base: una interacción útil entre otras variables no se prueba (trade-off para evitar la "curse of dimensionality").
- Las interacciones se calculan con valores reales (kW, Hl); la transformación y el escalado van después, dentro del `Pipeline` (ADR 0010). Los vacíos de las variables base pasan a sus interacciones y se imputan también en el `Pipeline`.
- Los gráficos del notebook muestran solo las 40 variables más parecidas al objetivo para que se lean; el agrupamiento y la elección usan las 278.
