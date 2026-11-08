# ADR 0008: política de limpieza del dataset

- **Fecha:** 08/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
La consigna 2.1 pide tratar los registros inválidos de la Fase 1, los valores faltantes y los valores atípicos extremos, y documentar cada decisión. Los ADR 0004, 0005 y 0006 ya decidieron qué días son inválidos y qué columnas se descartan, pero dejaron abiertas tres cosas: qué pasa con las otras columnas en un día inválido (el ADR 0005 admite que "solo se mira la columna `Frio (Kw)`"), qué se hace con los valores imposibles de las demás columnas y qué se hace con los vacíos que quedan.

La limpieza se aplica después de la división temporal (848 / 182 / 182 días) y todo lo que se mide se mide **solo con `X_train`**. Las cifras de abajo son de `X_train`, salvo que se indique otra cosa. Implementación: `src/limpieza.py` (`aprender_reglas_de_limpieza` y `limpiar`). Detalle y tablas: `notebooks/preprocesamiento.ipynb`, sección 3.

## Decisión
1. **Columnas.** Se aplican los criterios del ADR 0006 sobre `X_train`: de 439 columnas candidatas se conservan 262 (más 14 de control, 276 en total). La misma lista se usa en `X_val` y `X_test`.
2. **Días inválidos.** En los días con `valido = False` (17 en `X_train`, 5 en `X_val`, 3 en `X_test`) se anulan (`NaN`) 37 columnas: las 19 de Consolidado EE y las 18 razones "EE ... / Hl" de Consolidado KPI, que se calculan con esos mismos kW. Las demás columnas y el `objetivo` no se tocan. No se imputa nada (ADR 0004).
3. **Valores absurdos.** Se anula (`NaN`, solo esa celda) toda celda que sea **negativa** (salvo en columnas de temperatura) o que supere **5 veces el percentil 99** de su columna, medido en `X_train` después del punto 2. Si el percentil 99 fuera 0, la columna no tiene límite. Resultado: 389 celdas en `X_train`, 46 en `X_val` y 30 en `X_test`; 96 de las 262 columnas tienen alguna celda anulada (91 en `X_train`).
4. **Valores faltantes.** No se rellena ningún vacío en la limpieza. Quedan 5.259 de 222.176 celdas de medición vacías en `X_train` (2,4 %). Su tratamiento (imputación) se define una sola vez, con la misma regla para todos los modelos, dentro de un `Pipeline` ajustado solo con entrenamiento, en el cierre de la Fase 2.
5. **Filas.** La limpieza no borra filas. Las que no sirvan para entrenar (sin objetivo, o con vacíos en las variables que use el modelo) se quitan una sola vez, al final de la Fase 2, antes de guardar `dataset_final.csv`.
6. **Valores altos pero creíbles** (entre el percentil 99 y 5 veces ese valor) se dejan como están. Si hace falta winsorizar, se decide al elegir el modelo (Fase 3) y se compara con y sin.

## Justificación
- **Por qué solo las columnas de energía en un día inválido.** Los días inválidos se detectaron con el contador de `Frio (Kw)` de Consolidado EE. Comparando con los percentiles 1 a 99 de los días válidos, Consolidado EE tiene 10,5 % de celdas fuera de rango en días inválidos contra 2,1 % en días válidos; el resto de las hojas se parece a lo normal (entre 0 % y 5,9 %, con solo 17 días de base). Anular todo habría descartado información sin evidencia de que estuviera dañada. Las 54 columnas de Consolidado KPI son razones por hectolitro, no contadores acumulados, y solo las "EE" se derivan del contador que falló.
- **Por qué 5 veces el percentil 99.** El consumo de `Frio` varía 2,2 veces entre enero y julio, así que ninguna variación estacional real se acerca a ese factor. Entre 5 y 10 veces el percentil 99 hay 75 celdas en `X_train`; de las 66 que se pudieron comparar con los 3 días anteriores y los 3 posteriores, todas son picos aislados (de 7,6 a 1.890 veces la mediana de sus vecinos). Por encima de 10 veces ocurre lo mismo (132 de 138). Ejemplos: `Sala Maq (Kw)` llega a 4.208.132 con un percentil 99 de 46.001; `Restos Planta (Kw)` llega a −4.492.792.
- **Por qué anular y no recortar.** Un valor recortado al percentil parecería un dato verdadero y no lo es: mismo criterio del ADR 0004 (no fabricar datos).
- **Por qué anular la celda y no la fila.** La fila tiene otras 261 columnas de medición que pueden estar bien, y borrar días rompería los rezagos y promedios móviles de las variables nuevas.
- **Por qué no rellenar ahora.** Rellenar con la mediana o el valor vecino antes de dividir usaría información de todo el período. Un relleno previo inventaría datos para los rezagos. Y para comparar modelos hace falta que todos reciban el mismo tratamiento sobre las mismas filas de evaluación.
- **Qué muestran los faltantes.** 234 de las 262 columnas tienen 5 % o menos de vacíos; 204 nunca superan 3 días seguidos sin dato; solo 5 tienen tramos de más de 30 días (los silos de bagazo, 308 días; los cocimientos diarios, 257). Las razones de la Línea 5 del KPI faltan en 34,6 % de `X_train` pero en 61,0 % de `X_val` y 51,6 % de `X_test`: se dejan de medir.
- **Duplicados y tipos de dato** no son parte de esta limpieza: los resuelven el ADR 0003 (una sola versión por día) y la conversión de textos a número del ADR 0006.

## Alternativas descartadas
- **Anular todas las columnas en un día inválido:** descarta información sin evidencia de daño fuera de Consolidado EE.
- **Winsorizar o `np.clip` en p1 y p99 para todo:** modifica el 2 % de los datos buenos de cada columna para corregir un puñado de errores, y fabrica valores que parecen reales.
- **Un criterio de rango intercuartil:** marca miles de celdas válidas en columnas intermitentes (por ejemplo `HL Mosto IPA`, que vale 0 casi siempre) donde la mediana y el rango intercuartil son 0.
- **Un umbral de 10 veces el percentil 99:** deja pasar 75 picos aislados que se comprobaron como errores.
- **Eliminar las filas con alguna celda absurda:** se pierde información de las demás columnas y se corren los días.
- **Rellenar los vacíos durante la limpieza** (media, mediana, último valor): fuga de información desde validación y prueba, y relleno distinto para cada modelo.

## Limitaciones
- El factor 5 es un criterio, no una verdad; su respaldo es la comparación con los días vecinos hecha en `X_train`.
- La comparación del punto 2 se apoya en solo 17 días inválidos en entrenamiento.
- Si en el futuro una hoja distinta de Consolidado EE falla en un día que no es inválido para `Frio`, la limpieza solo lo detecta si el valor supera el límite o es negativo.
- La limpieza se mide con `X_train`; si el comportamiento de una columna cambia en `X_val` o `X_test` (como las razones de la Línea 5), no se detecta aquí, sino en la selección de variables.
- Los vacíos que quedan, y las columnas que dejan de medirse, se resuelven en la selección de variables y en el cierre de la Fase 2.
