# ADR 0007: coeficiente de correlación para comparar columnas con el objetivo

- **Fecha:** 05/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna (1.3) pide identificar las 15 variables más correlacionadas con el objetivo, analizar la multicolinealidad y mostrar heatmaps. Hay que elegir con qué coeficiente se mide la correlación entre las 262 columnas que pasan el ADR 0006 y el objetivo (el consumo de `Frio (Kw)` del día siguiente), sobre los 817 días válidos del tramo de entrenamiento que tienen objetivo.

Hechos medidos (secciones 21 y 22 del notebook):
- Con Pearson, `Frio (Kw)` correlacionado con el objetivo da 0,055 si se incluye el día inválido del 30/04/2021 (5.572.801 kW) y 0,859 si se excluyen los días inválidos.
- Aun con solo días válidos, 11 de las 39 columnas con Spearman mayor a 0,5 tienen un Pearson al menos 0,3 más bajo. Ejemplos: `Sala Maq (Kw)` (Spearman 0,821, Pearson 0,005), `KW Gral Planta` (0,780 y 0,049) y `Planta (Kw)` (0,774 y 0,047).
- La causa son 1 a 4 días por columna con valores de millones frente a medianas de decenas de miles (`Sala Maq (Kw)`: 4.208.132 el 24/12/2021 contra mediana 29.184; `KW Gral Planta`: 2.413.859 el 29/12/2021 contra 72.452).
- Ninguna columna supera 0,98 de correlación (Spearman) con el objetivo; 10 superan 0,7 y 39 superan 0,5.

## Decisión
1. **Spearman es el coeficiente principal** para ordenar las columnas por su relación con el objetivo y para medir la multicolinealidad entre columnas.
2. **Pearson se calcula al lado, solo como contraste**: una diferencia grande entre los dos coeficientes señala valores extremos en la columna.
3. Las correlaciones se calculan **solo con el tramo de entrenamiento** (primeros 70 % de los días) y **solo con días válidos con objetivo** (ADR 0004 y 0005), para que el test no influya en qué variables parecen importantes.
4. Dos columnas se consideran muy parecidas si su Spearman llega a 0,9 en valor absoluto (umbral de lectura, no de descarte). El descarte de redundantes se decide en la Fase 2.

## Justificación
- Spearman usa el orden de los valores, no su tamaño: un valor de 4 millones entre valores de 30 mil pesa lo mismo que cualquier otro valor alto. Pearson lo trata como un punto de enorme influencia.
- Con Pearson solo se habrían subestimado tres de las variables más informativas (`Sala Maq`, `KW Gral Planta`, `Planta`), que tienen Spearman cercano a 0,8.
- Medir solo en entrenamiento mantiene la misma lógica del ADR 0006.

## Alternativas descartadas
- **Pearson como principal:** lo distorsionan los valores absurdos de columnas que todavía no se limpiaron.
- **Limpiar primero los valores absurdos y usar Pearson:** requiere una regla de limpieza que se define en la Fase 2.
- **Usar todos los días, incluidos los inválidos:** los valores erróneos de `Frio` (por ejemplo el de 5,5 millones de kW) dominan el resultado.

## Limitaciones
- Spearman mide relaciones que suben o bajan juntas (monótonas) de a una columna por vez; no detecta combinaciones de columnas ni efectos con retraso.
- Los valores absurdos de las columnas distintas de `Frio` siguen en la tabla; se tratan en la Fase 2 (ADR 0006, limitaciones).
- La multicolinealidad se midió solo de a pares. El VIF (factor de inflación de la varianza) queda para la Fase 2, después de quedarse con una sola columna de cada grupo repetido; con copias exactas la matriz no se puede invertir.
- El corte del 70 % se aplicó sobre los 1.212 días y se redefinirá con el conjunto final.
