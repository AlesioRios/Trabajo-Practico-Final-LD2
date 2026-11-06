# ADR 0004: los días inválidos se marcan y no se imputan

- **Fecha:** 29/10/2026
- **Estado:** aceptada (el corte de plausibilidad es provisorio)
- **Autor:** Alesio Ríos

## Contexto
Hay días cuyo consumo no sirve como dato: no tienen lectura de las 23:59 (el acumulado queda incompleto) o traen valores del orden de millones. La consigna pide detectar y documentar los registros inválidos. Además, el objetivo del modelo es el consumo del día siguiente, así que un día inválido afecta también a la fila anterior.

## Decisión
Un día es **inválido** si:
1. no tiene lectura de las 23:59, o
2. su consumo de cierre supera `CONSUMO_MAXIMO_PLAUSIBLE` (100.000 kW, corte provisorio).

Para esos días `valido` es falso y `consumo_dia` queda vacío: **no se imputa**. El objetivo de la fila anterior también queda vacío. Las filas se conservan en la tabla; descartarlas al entrenar se decide en el preprocesamiento.

## Justificación
Imputar el objetivo equivaldría a medir el error del modelo contra un número fabricado por nosotros. En cambio, las lecturas que faltan a mitad del día no invalidan nada, porque el valor es un acumulado: solo importa que exista la lectura de cierre.

Resultado sobre los 1.212 días únicos: 1.205 válidos, 7 inválidos, 1.204 filas con objetivo y 8 sin él (los 7 días previos a un inválido y el último día de la serie).

| Día inválido | Motivo |
|---|---|
| 30/04/2021 (bloque 1) | 5.572.801 kW; solo está en el 1.º Excel, sin otra versión que lo corrija |
| 13/07/2021 | 23 lecturas, termina a las 23:00 |
| 02/03/2022 | 24 lecturas, termina a las 23:00 |
| 28/02/2023 | 24 lecturas, termina a las 23:00 |
| 13/04/2023 | 20 lecturas, termina a las 19:00 |
| 19/04/2023 | 17 lecturas, termina a las 16:00 |
| 26/10/2023 | 8 lecturas, termina a las 07:00; es el último día del dataset |

## Alternativas descartadas
- **Imputar el consumo faltante** (interpolar, usar el día anterior o el de las 23:00, que subestima unos 1.000 kW): contamina la métrica de evaluación.
- **Borrar las filas inválidas de la tabla:** correría los días y rompería las variables construidas con valores de días anteriores.

## Pendiente
El corte de 100.000 solo separa los errores de millones. Entre los días válidos el consumo va de 2.310 a 56.128 kW, con 11.051 y 35.980 como percentiles 5 y 95. Una propuesta inicial de rango plausible (0 a 35.000) no tiene respaldo en la consigna y dejaría fuera días con consumos entre 35.000 y 56.128 que quizá sean reales. El rango definitivo se fija en el análisis de atípicos de la Fase 1, con un criterio estadístico, y este registro se actualiza entonces.

**Actualización del 02/11/2026:** el análisis de atípicos mostró que el problema no era el rango sino fallas del contador dentro del día. Se resolvió con el [ADR 0005](0005-dias-con-saltos-en-el-contador.md): no se usa ningún rango fijo, el corte de 100.000 se mantiene y se suman criterios por la evolución del contador. Las cifras de resultado de este registro (1.205 válidos, 7 inválidos) corresponden a la regla anterior.
