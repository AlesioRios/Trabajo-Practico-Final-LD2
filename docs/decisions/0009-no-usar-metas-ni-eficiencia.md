# ADR 0009: no se calcula la eficiencia contra la meta (columnas Meta)

- **Fecha:** 08/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna 2.2 lista entre sus **sugerencias** de variables derivadas la "eficiencia energética (consumo real contra meta)", y aclara que las metas están en las columnas que empiezan con `Meta` de la hoja Consolidado KPI. Son 51 columnas. El filtro del ADR 0006 (más de 40 % de faltantes, medido en entrenamiento) las descartó a todas: tienen 43,3 % de vacíos en `X_train` (1,6 % en `X_val` y 0,5 % en `X_test`). Había que decidir si convenía hacerles una excepción al filtro para poder calcular la eficiencia.

Cifras de `X_train` (848 días, sección 5 de `notebooks/preprocesamiento.ipynb`):
- Las metas existen desde el 01/07/2021: 481 de los 848 días tienen metas, y los primeros 367 están 100 % vacíos.
- Dentro de un mismo mes hay un único valor (máximo 1 valor distinto por mes, en 16 meses).
- Cada columna toma como máximo 12 valores distintos en esos 481 días, con una mediana de 11 cambios de valor y un coeficiente de variación mediano de 0,11.
- Una sola columna es constante (`Meta Produccion Agua`).

## Decisión
**No se calcula la eficiencia contra la meta y las columnas `Meta` no se exceptúan del filtro del ADR 0006.** Las razones entre áreas y las interacciones, que también sugiere la consigna 2.2, no dependen de las metas y se hacen igual.

## Justificación
- **Los vacíos son estructurales, no un defecto de calidad.** La meta no existe antes de julio de 2021. Para usarla habría que rellenar hacia atrás (inventar un año de metas) o perder 367 de los 848 días de entrenamiento (43 %).
- **Casi no aportan variación diaria.** Valen lo mismo todo el mes y cada columna cambia pocas veces. La eficiencia (consumo sobre meta) sería, en la práctica, el consumo dividido por un número casi constante dentro de cada mes. Lo que cambia de mes a mes ya lo capta la variable mes del calendario.
- **La consigna lo plantea como sugerencia**, entre ocho opciones, y no como obligación.

## Alternativas descartadas
- **Exceptuar las columnas `Meta` del filtro del 40 %:** los modelos de árboles aceptan vacíos, pero la variable resultante quedaría vacía durante todo el primer año y aportaría poco el resto.
- **Usar solo los días desde julio de 2021:** se pierde el 43 % del entrenamiento.
- **Rellenar hacia atrás con la primera meta de 2021:** fabrica datos que no existieron (mismo criterio del ADR 0004).
- **No justificar y omitir:** la consigna menciona expresamente las columnas `Meta`, así que conviene dejar registrado por qué no se usan.

## Limitaciones
- Las metas mensuales podrían tener alguna relación con la estacionalidad que la variable mes no capte por completo; no se midió.
- Si la cátedra exigiera esta variable, habría que revisar esta decisión y exceptuar las columnas `Meta` en `src/limpieza.py`.
