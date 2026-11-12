# ADR 0013: razones entre áreas de la planta y regla para los denominadores chicos

- **Fecha:** 12/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna (2.2) pide crear variables guiadas por el dominio. Había que decidir qué razones entre áreas se construyen y qué hacer cuando el denominador es cero o casi cero (días con la producción parada), sin calcular la eficiencia contra las metas (ADR 0009) y sin transformar ni escalar (ADR 0010).

Toda la evidencia sale de `notebooks/variables_planta.ipynb`, medida solo con `X_train` (848 días). Las tablas que se citan son las de ese notebook.

## Decisión
Se crean seis razones con `crear_variables_planta` (`src/variables_planta.py`), todas con mediciones del mismo día y en sus unidades reales:

| Variable | Numerador | Denominador |
|---|---|---|
| `frio_sobre_planta` | Frío (kW) | Planta (kW) |
| `frio_por_hl_envasado` | Frío (kW) | Hl de cerveza envasada |
| `planta_por_hl_envasado` | Planta (kW) | Hl de cerveza envasada |
| `aire_por_hl_envasado` | Aire producido (m3) | Hl de cerveza envasada |
| `gas_por_hl_envasado` | Gas de planta (MJ) | Hl de cerveza envasada |
| `servicios_sobre_planta` | Servicios (kW) | Planta (kW) |

**Regla del denominador.** Si el denominador es vacío, cero o menor al 10 % de su mediana de entrenamiento, la razón queda vacía (NaN), nunca cero ni infinito. El umbral se aprende solo con entrenamiento (`aprender_reglas_de_planta`) y se aplica igual a validación, prueba y datos nuevos. La imputación de los vacíos se hace después, dentro del `Pipeline`.

No se incluyen: la razón de los compresores Mycom sobre el frío y las razones por línea de envasado.

## Justificación
- **Denominadores chicos (tabla 3).** En entrenamiento los hectolitros envasados tienen mediana 9.798 y umbral 980. Hay 27 días en cero, 2 vacíos y 41 por debajo del umbral (incluidos los ceros); el menor valor mayor que cero es 26. La planta en kW tiene mediana 70.059 y umbral 7.006, sin ceros ni días bajo el umbral (solo 18 días vacíos, los inválidos de la limpieza).
- **Efecto de la regla (tabla 4).** Para el frío por hectolitro envasado, la mediana es 3 kW por Hl con y sin regla, pero sin ella el percentil 99 es 97 y el máximo 871, y con ella 6 y 13. La regla cuesta 14 días con valor (805 contra 791).
- **Controles (tabla 7).** Las seis razones se agregan al final, sin tocar las columnas ni las filas originales, ninguna vale infinito, las reglas recalculadas solo con entrenamiento dan lo mismo y una fila revisada a mano coincide.
- **Parecido con el objetivo (tabla 9).** El Spearman con el frío de mañana es 0,847 para el frío de hoy, y 0,553 (`frio_sobre_planta`), 0,317 (`frio_por_hl_envasado`), 0,044 (`planta_por_hl_envasado`), −0,223 (`aire_por_hl_envasado`), −0,409 (`gas_por_hl_envasado`) y 0,128 (`servicios_sobre_planta`). Ninguna es copia del frío de hoy. Es una medida descriptiva: no decide qué variables se quedan, eso lo hace la selección.
- **Mycom sobre el frío no se incluye (tabla 10).** La suma de los siete compresores Mycom tiene Spearman 0,991 con el frío del mismo día; la razón entre ambos tiene mediana 0,732 y variación relativa 0,102. La suma dice casi lo mismo que el frío de hoy y no agrega información distinta. Queda pendiente evaluar compresores individuales en la selección.
- **Razones por línea no se incluyen (tabla 11).** En entrenamiento, la producción de la línea 2 tiene 134 días en cero y 194 bajo el umbral, la línea 3 tiene 182 y 188, la línea 4 tiene 91 y 101, y la línea 5 tiene mediana cero (492 días en cero) y no tiene columna de consumo eléctrico propio. Las razones quedarían vacías en una parte grande de los días.

## Alternativas descartadas
- **Sin regla, solo evitando la división por cero:** deja valores extremos (máximo 871 contra una mediana de 3).
- **Reemplazar por cero o por un valor grande los días con denominador chico:** inventa un dato que no existe y mezcla días parados con días normales.
- **Umbral fijo (por ejemplo, 100 Hl) en lugar de una fracción de la mediana:** no se adapta a la escala de cada denominador (kW contra hectolitros).
- **Eficiencia contra la meta:** ver ADR 0009.

## Limitaciones
- **Los días de producción casi nula son mucho más frecuentes fuera de entrenamiento (tabla 8):** 4,8 % de los días en entrenamiento, 20,9 % en validación y 19,8 % en prueba. Por eso las cuatro razones por hectolitro quedan vacías en una parte mayor de validación y prueba (tabla 5). La regla se aplica igual en los tres tramos y la causa no se conoce (paradas, temporada baja o un problema de registro); obliga a decidir con cuidado la imputación en el `Pipeline`, porque un día de planta casi parada no debería recibir el valor de un día normal.
- El 10 % es una elección simple y fácil de explicar, no un valor óptimo. No se probó con otros porcentajes.
- Las correlaciones de esta decisión son descriptivas y no reemplazan la selección de variables.
- Las interacciones y `PolynomialFeatures` (grado 2, sobre unas diez variables) se hacen después, junto con la selección.
