# ADR 0016: elección del pipeline de Ridge y guardado sin escalar

- **Fecha:** 13/11/2026
- **Estado:** propuesta
- **Autor:** Miligonzalzz

## Contexto

El pipeline anterior aplicaba imputación por vecinos y Yeo-Johnson al conjunto amplio de variables. Se compararon los tratamientos con Ridge usando pliegues cronológicos del tramo de entrenamiento. La validación externa se reservó para informar el rendimiento de la variante elegida.

## Decisión

Usar la variante A como opción por defecto con el conjunto chico provisional. Guardar las variables elegidas en sus unidades originales, junto con su tramo, objetivo, huella y linaje. Generar esos archivos cuando esté lista la selección definitiva.

| Variante | Tratamiento evaluado (tabla 1) |
|---|---|
| A | Mediana y escalado estándar |
| B | Mediana y Yeo-Johnson estandarizado |
| C | Mediana con indicador y Yeo-Johnson estandarizado |
| D | Mediana, Yeo-Johnson y escalado robusto |
| E | Tratamiento por grupos |
| F | Escalado estándar, KNN, Yeo-Johnson y escalado estándar |

## Justificación

Con el conjunto chico, E obtuvo el menor MAE promedio de los pliegues, 3319,33 kW. Su desvío fue 242,96 kW, por lo que el límite para elegir una variante más simple fue 3562,29 kW. A obtuvo 3407,63 kW con alpha 100 y fue la variante más simple dentro de ese límite (tabla 1).

En validación externa, A obtuvo 2272,91 kW frente a 2653,83 kW de persistencia y 2489,91 kW del promedio móvil, en 172 fechas comunes (tabla 5).

Todas las variantes empeoraron al pasar al conjunto grande. Para F, el MAE de los pliegues pasó de 3521,86 a 4107,66 kW (tabla 1). Esto apoya la hipótesis de que el conjunto amplio contribuyó al error histórico, pero el experimento con datos corregidos y regularización ajustada no reproduce la cifra anterior ni aísla su causa completa.

En los pliegues, persistencia obtuvo 3249,53 kW y la media móvil 3376,83 kW. A quedó por encima de ambas referencias. E quedó por debajo de la media móvil, pero por encima de persistencia (tabla 3). Ridge se usa como juez de los tratamientos. La Fase 3 elige el modelo final.

A prescinde del Yeo-Johnson previsto en el ADR 0010. Su error quedó dentro del margen acordado, con menos pasos que E (tabla 1). B y D usan Yeo-Johnson con escalados distintos. Sus errores por pliegue difieren y sus promedios coinciden al redondear (tabla 4).

En validación, la reducción frente a la media móvil fue del 8,72 %, por debajo del 10 %. Ridge aprovecha señales de calendario y de planta y redujo el error en ese bloque, pero la mejora observada es moderada y no se sostiene frente a persistencia en el promedio de los pliegues (tablas 3 y 5).

## Alternativas descartadas

B, C, D, E y F quedan disponibles para repetir la comparación cuando se integre la selección definitiva. E tuvo el promedio más bajo, pero la regla de simplicidad favoreció a A en el conjunto chico (tabla 1).

## Limitaciones

La selección de variables usa imputación por mediana, que puede diferir de otras variantes del pipeline. Elegir alpha con los mismos pliegues que se usan para comparar deja una estimación optimista en todas las variantes. La decisión usa un conjunto chico provisional. El nivel y la variabilidad del consumo cambian por período, igual que la cantidad de historia disponible (tabla 2). La prueba externa queda reservada y todavía no se genera el dataset final real.
