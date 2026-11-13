# Guía de la Fase 2: cómo seguir desde acá

Para quien retome el trabajo del equipo (personas o una IA que ayude). **Leer también** `consigna.md` (sección Fase 2), `docs/decisions/` (sobre todo los ADR 0004 a 0008) y `README.md`. Fecha de la guía: 08/11/2026.

## 1. Dónde estamos
Proyecto: predecir el consumo de frío del **día siguiente** (`Frio (Kw)`) de una cervecera. Defensa: **lunes 16/11/2026**.

La Fase 1 (EDA) está en `main`. La Fase 2 se trabaja en la rama `feature/preprocessing`, que ya tiene:

| Etapa | Estado | Dónde |
|---|---|---|
| 1. Tabla completa (46 hojas unidas, un día por fila) | Hecha | `construir_tabla_completa`, `cargar_tabla_completa` en `src/preprocessing_pipeline.py` |
| 2. División temporal 70 / 15 / 15 | Hecha | `dividir_por_fecha` en `src/preprocessing_pipeline.py` |
| 3. Limpieza | Hecha | `aprender_reglas_de_limpieza`, `limpiar` en `src/limpieza.py`; decisiones en el ADR 0008 |
| 4. Variables nuevas (consigna 2.2) | **Por hacer** | `src/variables_temporales.py` y `src/variables_planta.py` (ver secciones 4 y 5) |
| 5. Selección de variables (2.3) | **Por hacer** | `src/seleccion_de_variables.py` (ver secciones 4 y 5) |
| 6. Pipeline y guardado (2.4) | Comparación y funciones probadas. Generación final pendiente de selección. | `src/pipeline_sklearn.py`, `notebooks/pipeline_y_guardado.ipynb`, ADR 0016 |

## 2. Cómo arrancar
1. `git pull` y `git switch feature/preprocessing`.
2. Entorno: `conda activate cervecera_env` (si no existe, ver `README.md`, "Puesta en marcha").
3. Los tres Excel van en `data/` (ver `README.md`, "Datos"). **No se suben al repo.**
4. Cargar los datos limpios en una notebook (`notebooks/`, sin tocar `preprocesamiento.ipynb`):

```python
import sys
from pathlib import Path
import pandas as pd

sys.path.append("../src")
from preprocessing_pipeline import cargar_tabla_completa, dividir_por_fecha
from limpieza import aprender_reglas_de_limpieza, limpiar

tabla = cargar_tabla_completa(Path("../data"))         # la primera vez tarda ~6 minutos; después, segundos
X_train, X_val, X_test = dividir_por_fecha(tabla)       # 848 / 182 / 182 días, en orden cronológico
reglas = aprender_reglas_de_limpieza(X_train)           # se aprende SOLO con entrenamiento
X_train, X_val, X_test = (limpiar(parte, reglas) for parte in (X_train, X_val, X_test))
```

Resultado: tres tablas de 276 columnas (262 mediciones y 14 de control: `DIA`, `valido`, `objetivo`, etc.). `objetivo` es el consumo del día siguiente. Hay `NaN` (2,4 % de las celdas): **la limpieza no rellena nada ni borra filas.**

## 3. Reglas que no se rompen
1. **Decidir solo con `X_train`.** Lo que se aprende (límites, columnas elegidas, medias) sale de entrenamiento y se aplica igual a validación y prueba. `X_test` no se mira hasta el final.
2. **Solo información de hoy o anterior.** La fila de un día *t* no puede usar nada del día *t+1* en adelante. Es fuga de datos y invalida el trabajo.
3. **Variables con memoria (rezagos, promedios móviles):** se calculan sobre la tabla limpia **completa y en orden** (`pd.concat([X_train, X_val, X_test])`), porque la primera fila de validación necesita los últimos días de entrenamiento. Con esa tabla completa nunca se mira hacia adelante (`shift` con valores positivos, ventanas que terminan en el día actual) y después se vuelve a dividir con `dividir_por_fecha`.
4. **Cada función recibe una tabla y devuelve la misma tabla con columnas nuevas**, sin reordenar ni borrar filas.
5. **No tocar `objetivo`** ni las columnas de control.
6. **Una notebook, un módulo de `src/` y una rama por tarea.** Los `.ipynb` en conflicto casi no se resuelven. Solo se integran en `src/preprocessing_pipeline.py` al final.
7. **Cada decisión se justifica en la notebook** y, si afecta a los demás, en un ADR nuevo (`docs/decisions/0000-plantilla.md`).
8. **Pruebas:** cada función lleva pruebas en `tests/` con tablas chicas calculables a mano. Correr `python -m pytest tests -q`; hoy pasan 88.
9. **No transformar ni escalar** dentro de las funciones de variables: se trabaja con los valores reales (kW, Hl); la transformación y el escalado van después, en el `Pipeline` (ADR 0010).
10. **Commits** con prefijo (`feat:`, `fix:`, `docs:`, `chore:`), sin líneas de coautoría, y a la rama propia.

## 4. Las 9 tareas que faltan: qué se puede hacer en paralelo y qué no
Las tareas 1, 2 y 3 están hechas: la tabla con todas las variables (1.212 días × 337 columnas) se obtiene con `transform_variables`. Desde ahí, la selección (4) y el `Pipeline` con el guardado (5 a 7) se pueden hacer **a la vez**, cada una en su rama; el punto de entrada (8) y el cierre (9) se conectan al final.

| # | Tarea | Consigna | ¿Paralela? | Necesita |
|---|---|---|---|---|
| 1 | Variables temporales | 2.2 | **Hecha** (ADR 0012) | La tabla limpia (ya está) |
| 2 | Variables de planta (razones, interacciones) | 2.2 | Razones **hechas** (ADR 0013); faltan las interacciones | La tabla limpia (ya está) |
| 3 | Integrar las variables nuevas: `fitear_reglas_de_variables` y `transform_variables` | 2.2 | **Hecha** | Tareas 1 y 2 (hechas) |
| 4 | Selección con al menos 2 métodos y `TimeSeriesSplit` | 2.3 | **Se puede hacer ya**, en paralelo con la 5 | La tabla con todas las variables (tarea 3, hecha) |

| 6 | Quitar las filas que no sirven para entrenar, una sola vez y con la misma regla para todos los modelos | 2.4 | **Secuencial** | Tareas 4 y 5 |
| 7 | Guardar `dataset_final.csv` sin escalar, su checksum en `data/checksums.json`, `data_lineage.json` y el tamaño de `X_train`, `X_val`, `X_test` | 2.4 | **Secuencial** | Tarea 6 |
| 8 | Que `python src/preprocessing_pipeline.py` corra todo de punta a punta (el README lo promete; hoy el archivo solo arma la tabla y la divide) | Entregable | Esqueleto **ya**, en su propia rama; se conecta con las tareas 4 a 7 al final | Tareas 3 y 7 |
| 9 | Cierre: notebook ejecutado de cero, pruebas, README, `estado.md` y Pull Request a `main` | Entregable | **Secuencial**, al final | Todas |

Decisiones ya tomadas que afectan a las tareas: los valores faltantes **no** se rellenan en la limpieza (ADR 0008); la eficiencia contra la meta **no** se calcula (ADR 0009); la transformación por asimetría y el escalado van **después** de crear las variables, en el `Pipeline` (ADR 0010); las variables temporales son diez y se descartan los rezagos de 7, 30 y 365 días y la variable de año (ADR 0012); las razones de planta son seis, con una regla única para los denominadores chicos (ADR 0013).

Cada tarea va en su rama, que sale de `feature/preprocessing`, y se integra de nuevo a esa rama con Pull Request (merge commit). **No** se hace Pull Request a `main` hasta el cierre. Quién hace cada una se acuerda dentro del equipo.

## 5. Detalle de cada tarea

### Tarea 1: variables temporales (hecha)
- Rama `feature/preprocessing-temporales`, módulo `src/variables_temporales.py`, notebook `notebooks/variables_temporales.ipynb`, decisión en el ADR 0012.
- Función entregada: `crear_variables_temporales(tabla)`. Agrega 10 columnas: frío de ayer, de hace 2 y de hace 3 días; promedio y desvío de los últimos 7 días (mínimo 3 con dato); diferencia entre hoy y ayer; seno y coseno del día del año del día que se predice; indicadores de fin de año (24/12 al 6/01) y de fin de semana.
- Cuidados que sirven de modelo para las demás tareas: los rezagos van **por fila** y no por fecha (la columna `DIA` tiene fechas repetidas y faltantes, ADR 0002); el calendario es el del día que se predice (la fecha de hoy más un día); toda cifra citada en un ADR sale de una celda del notebook.
- Pruebas: `tests/test_variables_temporales.py`, incluida una que verifica que cambiar el futuro no altera las variables de las filas anteriores.

### Tarea 2: variables de planta (razones hechas; faltan las interacciones)
- Rama `feature/preprocessing-planta`, módulo `src/variables_planta.py`, notebook `notebooks/variables_planta.ipynb`, decisión en el ADR 0013.
- Funciones entregadas: `aprender_reglas_de_planta(entrenamiento_limpio)`, que mide el umbral de cada denominador solo con entrenamiento, y `crear_variables_planta(tabla, reglas)`. Agregan 6 razones: frío sobre planta, frío / Hl envasados, planta / Hl envasados, aire / Hl envasados, gas / Hl envasados y servicios sobre planta.
- Regla de los denominadores: vacío, cero o menor al 10 % de la mediana de entrenamiento da razón vacía (nunca cero ni infinito). **No** se calcula la eficiencia contra la meta (ADR 0009). Se usan los valores reales (kW, Hl), sin transformar (ADR 0010).
- Cuidado a tener en cuenta: los días de producción casi nula son mucho más frecuentes en validación y prueba (alrededor del 20 %) que en entrenamiento (5 %), así que las razones por hectolitro tienen más vacíos ahí; la imputación se decide en el `Pipeline`.
- Pruebas: `tests/test_variables_planta.py` (razones calculadas a mano, denominadores en cero, vacíos y chicos, reglas solo de entrenamiento).
- Interacciones hechas (ADR 0014): `aprender_reglas_de_interacciones` y `crear_interacciones` en `src/variables_interacciones.py` (10 variables base, 45 productos, notebook `notebooks/variables_interacciones.ipynb`). La integración quedó en `fitear_reglas_de_variables` y `transform_variables`; sigue la selección.

### Tarea 4: selección de variables
- Rama `feature/preprocessing-seleccion`, módulo `src/seleccion_de_variables.py`, notebook `notebooks/seleccion_de_variables.ipynb`, decisión en el ADR 0015. Función a entregar: `seleccionar_variables(X_train, y_train)`.
- Al menos 2 métodos de selección de los vistos en la materia (por ejemplo, filtro por correlación o información mutua, RFE, Lasso, importancia de Random Forest), evaluados con `TimeSeriesSplit` sobre `X_train` (nunca `KFold`: respeta el orden del tiempo) y `SEMILLA = 42`. Los valores vacíos se imputan dentro de cada pliegue. Si se conserva una interacción, se conservan sus dos variables componentes (principio de jerarquía, ADR 0014).
- Se decide además, con número: si algún compresor Mycom individual aporta (ADR 0013) y si el indicador de fin de semana aporta (ADR 0012).

### Tareas 5 a 7: `Pipeline` y guardado
- La comparación temporal y el guardado están implementados en `src/pipeline_sklearn.py` y documentados en `notebooks/pipeline_y_guardado.ipynb` y el ADR 0016. Falta generar los archivos finales con la selección definitiva.
- La variante provisional usa mediana y escalado estándar, ajustados solo con `X_train` (ADR 0016). Quitar las filas que no sirven para entrenar, una sola vez (sin objetivo, o con vacíos en las variables elegidas si el modelo lo exige). Separar `X_*` e `y_*`, guardar `data/processed/dataset_final.csv` **sin escalar**, agregar su checksum a `data/checksums.json`, escribir `data/processed/data_lineage.json` y documentar el tamaño final de `X_train`, `X_val` y `X_test`.
- Decisiones que quedan para esta tarea: cómo imputar los vacíos (ADR 0008, punto 4), en particular las razones por hectolitro, con muchos más vacíos en validación y prueba (ADR 0013); qué hacer con las columnas que dejan de medirse en validación y prueba (por ejemplo, las razones de la Línea 5 del KPI) y si conviene transformar cada modelo (ADR 0010).

### Tarea 8: punto de entrada
- Rama `feature/preprocessing-entrada`. Es la única que edita `src/preprocessing_pipeline.py` y `notebooks/preprocesamiento.ipynb`: agrega el bloque `if __name__ == "__main__":` que corre todo de punta a punta, y conecta las tareas 4 a 7 cuando están integradas. Las demás tareas no editan esos dos archivos, para evitar conflictos.

## 6. Integración y cierre (en orden)
1. **Integrar las tareas 1 y 2:** merge de las dos ramas en `feature/preprocessing`. En `src/preprocessing_pipeline.py` quedan `fitear_reglas_de_variables(entrenamiento_limpio)` (aprende solo con entrenamiento y devuelve un diccionario de reglas) y `transform_variables(tabla_limpia, reglas)` (agrega temporales, razones e interacciones a la tabla limpia completa y en orden); el notebook es `notebooks/construir_variables.ipynb`. Verificar mismas filas que la tabla limpia y sin columnas repetidas.
2. **Selección definitiva** y revisión del conjunto elegido entre todos.
3. **Cierre:** ejecutar `notebooks/preprocesamiento.ipynb` de cero (reiniciar kernel y correr todo), `python -m pytest tests -q`, y abrir el Pull Request de `feature/preprocessing` a `main` con "Create a merge commit". Antes del merge, otra persona lo revisa.

Si algo se atrasa, la prioridad es que existan `dataset_final.csv`, `data_lineage.json` y la selección con 2 métodos. Las interacciones más elaboradas se pueden recortar.

## 7. Si ayuda una IA
Pasarle este archivo y `docs/decisions/`, decir qué tarea se tiene, y pedirle: una decisión por vez, el código explicado línea por línea, pruebas junto con cada función y la justificación para la notebook. Verificar que respete las reglas de la sección 3, sobre todo la 1 y la 2.
