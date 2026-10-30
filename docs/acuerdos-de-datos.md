# Acuerdos de datos y de trabajo del equipo

Un solo lugar con lo que las tres personas del equipo tienen que respetar para que sus partes encajen. Las decisiones y su justificación están en [`docs/decisions/`](decisions/); acá va lo operativo. Si algo de esto cambia, se actualiza este archivo **en el mismo commit** y se avisa al equipo.

## 1. Flujo de datos

```
Excel de la planta (data/, no versionados)
   │  notebooks/eda.ipynb  ·  src/auxiliar_functions.py
   ▼
tabla diaria `diario`  (una fila por día, con su objetivo)
   │  src/preprocessing_pipeline.py            (Fase 2)
   ▼
data/processed/dataset_final.csv
   │  src/train_model.py                       (Fase 3)
   ▼
models/modelo_v1.0.0.pkl  ·  models/model_registry.json
   │  src/predict.py                           (Fase 4)
   ▼
results/predicciones.csv
```

`predict.py` tiene que procesar el Excel de la cátedra con **las mismas funciones** que el entrenamiento. Por eso la lectura y la unificación viven en `src/auxiliar_functions.py` y no dentro del notebook.

## 2. La tabla `diario`

Una fila por **día** (bloque de lecturas de 00:00 a 23:59), en el orden original de los datos.

| Columna | Qué es |
|---|---|
| `DIA`, `bloque` | Clave del día: fecha rotulada y número de bloque dentro de esa fecha (0 si la fecha no se repite). **La fecha no alcanza para identificar un día** (ver ADR 0002). |
| `HORA` | Hora de la última lectura del día (normalmente `23:59:00`). Es texto. |
| 19 variables `... (Kw)` y `KW Gral Planta` | Acumulados del día a la hora de cierre, por área de la planta. `Frio (Kw)` es la variable a predecir. |
| `filas` | Cantidad de lecturas del bloque (25 en un día normal). |
| `archivo` | Excel de donde se tomó el día (`2020-2022`, `2021-2023`, `2022-2023`). |
| `completo`, `plausible`, `valido` | Controles de calidad (ADR 0004). `valido` es el que se usa. |
| `consumo_dia` | Consumo de cierre del día; **vacío** si el día es inválido. |
| `objetivo` | `consumo_dia` del **día siguiente**, o sea de la fila que sigue. Vacío si ese día es inválido o no existe. |

Reglas:
- **Día siguiente = fila siguiente.** No reordenar la tabla ni ordenar por fecha rotulada: el orden de las filas es el orden de los datos.
- Las variables de calendario (día de la semana, mes) salen de una fecha que puede estar desfasada; usarlas con cuidado hasta que se resuelva (ADR 0002).
- Las filas sin `objetivo` se descartan **al entrenar**, no antes: sirven como historia para variables de días anteriores.
- Toda estadística que se calcule (medias, escalas, selección de variables) se ajusta solo con el conjunto de entrenamiento.

## 3. Funciones compartidas (`src/auxiliar_functions.py`)

| Función | Qué hace |
|---|---|
| `leer_consolidado_ee(fuente)` | Lee la hoja `Consolidado EE` por nombre, saca las filas vacías del final y convierte `DIA` a fecha. Devuelve `(tabla, filas_sin_fecha)`. |
| `resumir_por_fecha(tabla)` | Cantidad de filas y última hora de cada fecha. |
| `numerar_bloques(tabla)` | Número de bloque de cada fila: empieza uno nuevo cuando la hora retrocede. |
| `cierres_por_bloque(tabla, archivo)` | Una fila por bloque, con su última lectura y los controles de calidad. |
| `unir_versiones(cierres)` | Una sola versión por bloque cuando un día está en dos Excel (ADR 0003). |
| `construir_diario(mejores)` | Agrega `valido`, `consumo_dia` y `objetivo`. |
| `estilizar`, `tabla_simple`, `mostrar_lado_a_lado`, `mostrar_lado_a_lado_simple`, `formatear_fechas` | Presentación de tablas en los notebooks. |
| `SEMILLA`, `HORA_CIERRE`, `CONSUMO_MAXIMO_PLAUSIBLE`, `COLUMNA_CONSUMO` | Constantes compartidas. |

Reglas:
- Se **agregan** funciones nuevas; no se cambia el nombre, los parámetros ni el resultado de una existente sin avisar antes.
- Toda función nueva lleva su docstring (qué hace, parámetros, qué devuelve, supuestos) y, si procesa datos, un test con una tabla chica inventada en `tests/`.
- Los `.xlsx` no se leen en los tests: no están en el repositorio.

## 4. Formato de los archivos intermedios (propuesta, a confirmar antes de la Fase 2)

- CSV en UTF-8, separador coma, punto decimal, **sin columna de índice**.
- Fechas en formato `AAAA-MM-DD`.
- Se conservan los nombres de columna de los Excel y las claves `DIA` y `bloque`; el objetivo se llama `objetivo`.
- Los modelos se guardan con versión semántica (`modelo_v1.0.0.pkl`) y se registran en `model_registry.json`.

## 5. Cómo trabajamos

- Antes de empezar: `git switch <rama de la fase>` y `git pull`.
- Un `.ipynb` lo edita **una persona por vez** (los conflictos de formato son difíciles de resolver). El código reutilizable va a `src/`.
- Antes de subir: `pytest` tiene que pasar.
- Commits chicos y con el usuario de cada uno. El mensaje dice si se cambia una columna, una función o un formato de archivo.
- `SEMILLA = 42` en toda partición y en todo estimador con `random_state`.
- Los Excel no se versionan. La huella de cada uno está en `data/checksums.json`.
- Cada fase se integra a `main` por Pull Request con "Create a merge commit". El PR de `feature/eda` se integra **antes** que el de `feature/preprocessing`.
