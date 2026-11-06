# ADR 0005: días con saltos o reinicios tardíos del contador se marcan inválidos

- **Fecha:** 02/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna pide detectar valores atípicos y registros inválidos, entre ellos los días en que el totalizador no se reinició a medianoche. El ADR 0004 solo cubría los días sin lectura de las 23:59 y los errores de millones. Al mirar la evolución del contador dentro de cada día (sección 16 del notebook) aparecieron fallas que ese criterio no detecta y que inflan el consumo de cierre.

Hechos medidos sobre los 1.212 días:
- Más del 99 % de los días arranca en 0 a las 00:00; solo 4 arrancan distinto (29.113, 21.200, 787 y 7).
- En un día normal el contador sube unos 1.300 a 1.500 kW por hora; el 95 % de los días no tiene ninguna hora con un aumento mayor a 2.212.
- Hay 34 días con un aumento de más de 5.000 en una hora o con una caída de más de 5.000.

## Decisión
Además de los criterios del ADR 0004, un día es **inválido** si:
3. tiene un **salto no corregido**: algún aumento de más de `UMBRAL_SALTO` (5.000 kW) entre dos lecturas seguidas y el contador no vuelve a bajar. Se detecta porque la suma del mayor y el menor incremento del día es de 5.000 o más, o
4. tiene un **reinicio tardío**: sin salto grande, pero con una caída de más de 5.000 (el totalizador se reinició horas después de la medianoche), o
5. su **lectura inicial** supera `UMBRAL_LECTURA_INICIAL` (100 kW): el contador no arrancó en 0.

Un **salto corregido** (la lectura errónea sube y la siguiente vuelve a su valor) **no** invalida el día: el cierre de las 23:59 es confiable. Sigue valiendo el ADR 0004: para los días inválidos `consumo_dia` queda vacío y **no se imputa**.

## Justificación
- Un aumento de más de 5.000 en una hora es más del doble de lo que sube cualquier hora de un día normal. No es creíble como consumo de este sistema.
- Ejemplo de salto no corregido: el 01/08/2021 (56.128 kW, el mayor consumo de la serie). El contador sube normal hasta las 21:00 (24.494), a las 22:00 marca 52.457 (un aumento de 27.963) y sigue normal desde ahí. Sin el escalón, el día cerraría cerca de 29.500 (estimación).
- Ejemplo de salto corregido: el 30/04/2021 (primer bloque). A las 05:00 el incremento es +11.518 y a las 06:00 es −8.446; el resto del día es normal y el cierre (35.657) es correcto.
- Ejemplo de reinicio tardío: el 11/07/2020. Arranca en 29.113 y a las 01:00 baja a 0; el cierre (29.046) cubre 23 de las 24 horas.
- La suma del mayor y el menor incremento distingue un pico (se compensa: 11.518 − 8.446 = 3.072) de un escalón (27.963 + 525 = 28.488).
- Los consumos altos sin saltos grandes (48.046, 42.919, 42.005, 41.901…) siguen siendo válidos: el máximo válido queda en unos 48.000 kW. **No se aplica el rango "0 a 35.000"** de la documentación del dataset: habría descartado días reales.

Resultado sobre los 1.212 días únicos (ejecutado el 03/11/2026): **1.187 válidos y 25 inválidos**, con 1.186 filas con objetivo (antes 1.205, 7 y 1.204). Los 25 inválidos son: 6 sin lectura de las 23:59 y 1 de millones (los 7 del ADR 0004), más 16 saltos no corregidos (uno de ellos, el 30/04/2021 bloque 1, ya era inválido), 1 reinicio tardío (11/07/2020) y 3 que arrancan distinto de cero (11/07/2020, 01/07/2021 y 01/07/2023); algunos cumplen más de un criterio. Otros 17 días tienen un salto corregido y siguen siendo válidos.

**Lecturas por día.** La consigna pide verificar los días con menos de 24 o con más de 25 lecturas. Entre los 1.187 días válidos hay un día con 23 lecturas (01/11/2021): se mantiene válido porque llega a las 23:59, arranca en 0 y falta solo una lectura intermedia, que no afecta a un acumulado (ADR 0004). Y hay 3 días con 26 lecturas, todos el segundo bloque del 30/06 de cada año (2021, 2022 y 2023), cuya última lectura es a las 23:59:59: cumplen todos los criterios y se mantienen. Dos de los tres días siguientes (01/07/2021 y 01/07/2023, con 23 lecturas) son inválidos por arrancar distinto de cero; es una sospecha de un artefacto en el cambio de año de los Excel, no confirmada.

Los saltos no corregidos se concentran en dos períodos: julio-agosto de 2021 (5 días) y enero-marzo de 2022 (6 días). La causa no está identificada.

Con la regla nueva, el consumo de cierre de los días válidos va de 2.310 a 48.046 kW (percentiles 5 y 95: 11.026 y 35.700). Las conclusiones del EDA sobre el día de la semana, el mes y el año no cambiaron (el efecto del día de la semana explica 0,60 % de la variación, el mes 58,8 % y el año 11,2 %).

## Alternativas descartadas
- **Marcar solo los saltos de más de 10.000:** dejaría unos 11 días con una falla probable.
- **No agregar nada:** incumple lo que pide la consigna sobre registros inválidos y deja entre los datos de entrenamiento cierres inflados.
- **Corregir el escalón restándolo:** equivale a imputar un valor que fabricamos nosotros (mismo motivo del ADR 0004).
- **Un rango fijo de consumo plausible (0 a 35.000):** no sale de la consigna y descarta días reales.

## Limitaciones
- El umbral de 5.000 es un criterio, no una verdad. Un escalón menor pasaría desapercibido.
- La regla es conservadora: al menos el 11/07/2020 tenía el cierre casi correcto, y dos saltos no corregidos (14/03/2022 y 17/01/2023) cierran en valores coherentes con sus vecinos.
- Solo se mira la columna `Frio (Kw)`; una falla en otra columna no se detecta.
- Cada día inválido también deja sin objetivo al día anterior: se pierden 18 filas más (de 1.204 a 1.186 con objetivo).
