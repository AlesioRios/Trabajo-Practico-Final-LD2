# ADR 0002: cada bloque de lecturas es un día distinto aunque repita la fecha

- **Fecha:** 29/10/2026
- **Estado:** aceptada (con una consulta pendiente a la cátedra)
- **Autor:** Alesio Ríos

## Contexto
La consigna pide verificar que ninguna fecha tenga más de 25 registros. En el primer Excel hay 8 fechas que sí los superan (cinco con 50 filas, una con 51 y dos con 100). Todas son **fin de mes de un mes corto**: 28/02, 30/04, 30/06, 30/09 y 30/11. Cada una contiene 2 o 4 **bloques** completos de lecturas (de 00:00 a 23:59), con su propio reinicio a 0 y sus propios valores. Un bloque nuevo empieza cuando la hora retrocede respecto de la fila anterior.

## Decisión
- Cada bloque es **un día distinto**, aunque comparta fecha con otro.
- "Día siguiente" significa el **bloque siguiente** en el orden del archivo, no la fecha siguiente del calendario.
- Un bloque se identifica con la pareja **(fecha rotulada, número de bloque dentro de esa fecha)**.
- La fecha rotulada se conserva como dato informativo, no como una verdad sobre el calendario.

## Justificación
- **No son copias:** los bloques de una misma fecha tienen valores propios (el 30/09/2020 cierra en 29.623 y en 25.558), y los que están en dos Excel coinciden entre sí (30/09/2021, 30/11/2021 y 28/02/2022).
- **El calendario no cierra, pero sí las cuentas:** faltan fechas siempre iguales (31/03, 31/05, 31/10, 31/12 y del 29 al 31 de agosto) y sobran bloques en los meses cortos. La suma de bloques menos días reales da **0** y cada año completo (2021 y 2022) tiene 365 bloques, igual que sus 365 días.
- **La hoja `Seguimiento Dia`** del segundo Excel marca 614 "últimos datos del día", igual a sus 614 bloques y no a sus 602 fechas distintas. Es una pista, no una prueba independiente: tiene las mismas filas que `Consolidado EE`.

## Alternativas descartadas
- **Quedarse con el primer bloque de cada fecha:** descarta unos 7 días reales por año y no corrige las fechas.
- **Quedarse con el último bloque:** mismo problema.
- **Reasignar fechas consecutivas según el orden del archivo:** obliga a inventar fechas sin poder comprobarlas.

## Limitaciones y pendientes
No se sabe qué fecha real le corresponde a cada bloque: las etiquetas pueden estar desfasadas unos días en algunos meses. Hasta resolverlo, las variables de calendario (día de la semana, mes) se usan con cuidado, y la medición del efecto del día de la semana debe tener esto en cuenta. Queda pendiente consultar a la cátedra cómo se asignan las fechas.
