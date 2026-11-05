# ADR 0006: criterios de calidad de columnas y limpieza de textos numéricos

- **Fecha:** 05/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna pide analizar cada hoja de los Excel, su variabilidad y su correlación con el objetivo. Los tres Excel tienen 46 hojas en total, y solo Consolidado EE contiene `Frio (Kw)`. Antes de usar las demás hojas como variables hay que decidir cuáles son confiables (sección 18 del notebook).

Hechos medidos sobre los cierres diarios de las doce hojas perfiladas:
- Las hojas no se pueden pegar por posición: tipeos en la hora, una fecha mal rotulada, un cambio de formato y un hueco de doce horas en Glicol 2020-2022 las desalinean de EE.
- Con la regla "bloque nuevo cuando la hora cae doce horas o más", las 46 hojas coinciden con EE en cantidad de bloques (624, 614 y 482) y en la fecha de cada uno.
- Las columnas con más de 40 % de faltantes se concentran en 2020-2022, el tramo más antiguo.
- Ninguna columna con valores no repetidos tiene un coeficiente de variación menor a 0,05; las únicas por debajo de 0,1 son las columnas `Id`.
- `Id` es un contador de filas (30000 a 42008 en 2022-2023, 10000 a 25559 en Glicol 2020-2022), idéntico en las hojas que lo tienen.
- Hay 68 celdas con texto: 63 son números con coma decimal o punto de miles, 2 son espacios y 3 son anotaciones de la hoja Auxiliar.

## Decisión
1. **Unión entre hojas por número de bloque diario**, nunca por posición de fila.
2. **Descartar en el preprocesamiento** las columnas con más de 40 % de faltantes (`UMBRAL_MUCHOS_FALTANTES`) y las que repiten el mismo valor en 95 % o más de los días (`UMBRAL_CASI_CONSTANTE`). Ambas se miden **solo sobre el conjunto de entrenamiento** (corte por fecha). Además se comprueba que cada variable conservada esté disponible en los datos más recientes: eso es un control de disponibilidad, no un criterio de selección.
3. **Excluir por nombre la columna `Id`**: es un contador de filas, no una medición.
4. **Textos numéricos** (`convertir_texto_a_numero`): vacío o solo espacios, faltante; con coma, los puntos son de miles y la coma es el decimal; sin coma y con dos o más puntos, el último punto es el decimal; lo que no se pueda leer como número, faltante.
5. El coeficiente de variación queda como dato informativo del perfil, no como criterio de descarte.

## Justificación
- Con tres Excel de ventanas solapadas, unir por posición produce desfases silenciosos; el número de bloque es la única clave validada.
- Medir en entrenamiento evita que el futuro influya en qué variables se eligen; casi todos los faltantes están en el inicio de la serie, por lo que pesan más en el entrenamiento que en toda la serie.
- La repetición no depende de la escala; `VarianceThreshold` y el coeficiente de variación sí, y el coeficiente es inestable con media cercana a 0 y sensible a valores extremos.
- La regla de los puntos se verificó contra los vecinos numéricos: en `KW Trafo 8` del 05/02/2022 los valores crecen 338,825 por hora, y `'1.016.475'` equivale a 3 × 338,825 = 1016,475.

## Alternativas descartadas
- **Un corte por coeficiente de variación:** no marcó nada que la repetición no marcara (salvo `Id`).
- **Medir los faltantes sobre toda la serie:** deja que los datos de prueba influyan en la selección.
- **Redondear los textos con tres decimales a uno:** los vecinos tienen hasta tres decimales; se rompería la progresión exacta.
- **Convertir los textos dentro de `cierres_de_hoja`:** el perfil dejaría de ver las celdas con texto.

## Limitaciones
- Los umbrales de 40 % y 95 % son criterios, no verdades. Se deciden durante el desarrollo, no de alguna pauta de las consignas del proyecto.
- Un texto con un solo punto y sin coma (`'1.234'`) se leería como decimal; no aparece en los datos.
- No se perfilaron KPI, Consolidado Producción, Auxiliar ni Seguimiento Dia.
- Las columnas casi constantes de marcas de Producción (`HL Mosto <marca>`) podrían tener relación con el objetivo; se decide después de las correlaciones.
- El tramo interpolado de `KW Trafo 8` (05/02/2022, 00:00 a 10:00) está confirmado en esa columna; falta buscarlo en `Frio`.
