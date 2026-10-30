# ADR 0003: qué versión se usa cuando un día está en dos Excel

- **Fecha:** 29/10/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
Los tres Excel cubren períodos que se solapan: el 1.º y el 2.º comparten 259 bloques diarios, el 2.º y el 3.º comparten 249, y el 1.º y el 3.º no se tocan. Casi siempre ambas versiones coinciden: de los bloques completos en ambos archivos (256 y 247), el consumo de cierre es idéntico salvo en **5 casos**, todos entre el 1.º y el 2.º Excel:

| Fecha | 1.º Excel | 2.º Excel | Diferencia |
|---|---|---|---|
| 28/07/2021 | 2.900.770 | 30.764 | 2.870.006 |
| 25/08/2021 | 2.907.122 | 37.122 | 2.870.000 |
| 28/11/2021 | 6.155.928 | 25.719 | 6.130.209 |
| 02/02/2022 | 52.655 | 19.887 | 32.768 |
| 18/02/2022 | 8.861.470 | 21.470 | 8.840.000 |

Además, algunos bloques están cortados al final de un archivo y completos en el siguiente (16/03/2022 y 06/03/2023).

## Decisión
Ante varias versiones de un mismo bloque se elige la **completa** (llega a las 23:59) y **plausible** (consumo menor o igual al corte de `CONSUMO_MAXIMO_PLAUSIBLE`). Si siguen empatadas, se usa la del **archivo más reciente**.

## Justificación
- En los 5 casos el 2.º Excel trae un valor dentro del rango habitual (el 90 % de los días válidos va de unos 11.000 a 36.000 kW) y el 1.º trae un error.
- Cuatro de los errores son del orden de millones, con diferencias casi redondas. El del 02/02/2022 (52.655 contra 19.887) pasa el corte de 100.000, pero la diferencia es exactamente 32.768, que es 2 elevado a la 15 y no parece un consumo. Sin el desempate por archivo se hubiera quedado el valor erróneo.
- Entre el 2.º y el 3.º Excel no hay ninguna discrepancia.

## Alternativas descartadas
- **Preferir el archivo más antiguo:** lo contradice la evidencia de los 5 casos.
- **Promediar las versiones:** fabrica un valor que no existe en ningún archivo.
- **Endurecer el corte de plausibilidad (por ejemplo a 40.000):** se arriesga a descartar días reales de consumo alto; hay días válidos de hasta 56.128 kW que todavía no se revisaron.

## Limitaciones
Es una regla empírica apoyada en 5 casos. Si apareciera una discrepancia en la que el archivo antiguo tiene razón, la regla elegiría mal. Si la cátedra provee el archivo de test con otra estructura, hay que revisarla.
