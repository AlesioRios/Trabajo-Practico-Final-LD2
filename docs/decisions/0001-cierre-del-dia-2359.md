# ADR 0001: el consumo del día se toma de la lectura de las 23:59

- **Fecha:** 29/10/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
Cada día de la hoja `Consolidado EE` trae lecturas horarias de valores **acumulados** (de 00:00 a 23:00) más una lectura de cierre a las 23:59. Los acumulados arrancan en 0 a las 00:00 y se reinician a medianoche. La consigna exige elegir entre la lectura de las 23:00 y la de las 23:59, aplicar la elección en todo el trabajo y justificarla.

## Decisión
El consumo de frío de un día (`Frio (Kw)`) es el acumulado de la **última lectura del día, a las 23:59**. Algunas lecturas de cierre llevan unos segundos de diferencia (23:59:01 o 23:59:59) y se tratan como el mismo cierre: la condición es "hora mayor o igual a 23:59:00".

## Justificación
El objetivo es el consumo de un **día entero**, y la lectura de las 23:00 deja afuera la última hora. Comparando ambas lecturas en 606 días del primer Excel (sin contar errores del orden de millones), cortar a las 23:00 pierde:

| Medida | kW perdidos | % del total del día |
|---|---|---|
| Promedio | 1.011 | 4,2 |
| Mediana | 961 | 4,0 |
| Percentil 10 | 471 | 2,9 |
| Percentil 90 | 1.496 | 5,5 |

La diferencia va siempre en el mismo sentido: la lectura de las 23:00 subestima el día. Los acumulados reinician a medianoche, así que el valor de las 23:59 ya es el total del día y no hace falta sumar horas.

## Alternativas descartadas
- **Lectura de las 23:00:** pierde una hora completa de consumo de forma sistemática.
- **Sumar los incrementos hora por hora:** innecesario, porque el valor ya es un acumulado, y más sensible a lecturas intermedias faltantes.

## Consecuencias
Un día sin lectura de las 23:59 no tiene su consumo completo y se considera inválido (ver ADR 0004). Las lecturas que faltan a mitad del día no afectan el resultado.
