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
| 6. Guardado y cierre (2.4) | **Por hacer** | `data/processed/`, `data_lineage.json` (ver secciones 4 y 5) |

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
8. **Pruebas:** cada función lleva pruebas en `tests/` con tablas chicas calculables a mano. Correr `python -m pytest tests -q`; hoy pasan 42.
9. **No transformar ni escalar** dentro de las funciones de variables: se trabaja con los valores reales (kW, Hl); la transformación y el escalado van después, en el `Pipeline` (ADR 0010).
10. **Commits** con prefijo (`feat:`, `fix:`, `docs:`, `chore:`), sin líneas de coautoría, y a la rama propia.

## 4. Las 9 tareas que faltan: qué se puede hacer en paralelo y qué no
Las tareas 1 y 2 son independientes entre sí y se pueden hacer a la vez desde ya. La tarea 3 necesita las dos. De la selección y el cierre se puede **preparar el código en paralelo**, pero el resultado definitivo necesita que antes estén todas las variables creadas.

| # | Tarea | Consigna | ¿Paralela? | Necesita |
|---|---|---|---|---|
| 1 | Variables temporales | 2.2 | **Paralela** con la 2 | La tabla limpia (ya está) |
| 2 | Variables de planta (razones, interacciones) | 2.2 | **Paralela** con la 1 | La tabla limpia (ya está) |
| 3 | Integrar 1 y 2 en `construir_variables(tabla)` | 2.2 | **Secuencial** | Tareas 1 y 2 terminadas |
| 4 | Selección con al menos 2 métodos y `TimeSeriesSplit` | 2.3 | Código **preparable en paralelo** (con las 10 variables provisionales de la sección 30 del EDA); resultado definitivo **secuencial** | Tarea 3 |
| 5 | `Pipeline` de imputación, transformación y escalado, ajustado solo con entrenamiento | 2.4 | Código **preparable en paralelo**; se ajusta con el conjunto final | Tarea 4 |
| 6 | Quitar las filas que no sirven para entrenar, una sola vez y con la misma regla para todos los modelos | 2.4 | **Secuencial** | Tareas 4 y 5 |
| 7 | Guardar `dataset_final.csv` sin escalar, su checksum en `data/checksums.json`, `data_lineage.json` y el tamaño de `X_train`, `X_val`, `X_test` | 2.4 | **Secuencial** | Tarea 6 |
| 8 | Que `python src/preprocessing_pipeline.py` corra todo de punta a punta (el README lo promete; hoy el archivo solo arma la tabla y la divide) | Entregable | Esqueleto en **paralelo**; se completa al final | Tareas 3 y 7 |
| 9 | Cierre: notebook ejecutado de cero, pruebas, README, `estado.md` y Pull Request a `main` | Entregable | **Secuencial**, al final | Todas |

Decisiones ya tomadas que afectan a las tareas: los valores faltantes **no** se rellenan en la limpieza (ADR 0008); la eficiencia contra la meta **no** se calcula (ADR 0009); la transformación por asimetría y el escalado van **después** de crear las variables, en el `Pipeline` (ADR 0010).

Cada tarea va en su rama, que sale de `feature/preprocessing`, y se integra de nuevo a esa rama con Pull Request (merge commit). **No** se hace Pull Request a `main` hasta el cierre. Quién hace cada una se acuerda dentro del equipo.

## 5. Detalle de cada tarea

### Tarea 1: variables temporales
- Rama `feature/preprocessing-temporales`, módulo `src/variables_temporales.py`, notebook `notebooks/variables_temporales.ipynb`.
- Función a entregar: `crear_variables_temporales(tabla)`.
- Qué: rezagos del consumo (ayer, hace 7 días), promedios móviles, mes, día de la semana, fin de semana, estacionalidad (seno y coseno del día del año).
- Cuidados: comprobar que los días estén en orden y sin saltos antes de calcular rezagos. Las fechas pueden estar corridas (ADR 0002), así que el día de la semana debe tomarse con cautela; la estacionalidad con seno y coseno es la opción más segura. `calcular_baselines` en `src/auxiliar_functions.py` muestra cómo pedir un mínimo de días en un promedio móvil.
- Prueba: el rezago de 1 día de la fila *t* debe valer lo que tenía la fila *t-1*.

### Tarea 2: variables de planta
- Rama `feature/preprocessing-planta`, módulo `src/variables_planta.py`, notebook `notebooks/variables_planta.ipynb`.
- Función a entregar: `crear_variables_de_planta(tabla)`.
- Qué: razones entre áreas (por ejemplo, frío sobre servicios) e interacciones entre las variables principales. **No** incluye la eficiencia contra la meta: las columnas `Meta` no se usan (ADR 0009: existen solo desde julio de 2021 y valen lo mismo todo el mes).
- Cuidados: una razón con denominador cero o muy chico produce valores enormes; decidir y justificar cómo se trata. Usar los valores reales (kW, Hl), sin transformar (ADR 0010).
- Prueba: una razón 10/2 da 5, y otra con denominador cero.

### Tareas 4 a 8: selección, `Pipeline` y guardado
- Rama `feature/preprocessing-seleccion`, módulo `src/seleccion_de_variables.py`, notebook `notebooks/seleccion_de_variables.ipynb`. Función a entregar: `seleccionar_variables(X_train, y_train)`.
- Para preparar en paralelo mientras se hacen las tareas 1 y 2: el código de **al menos 2 métodos de selección** (por ejemplo, importancia de Random Forest y LASSO o RFE), evaluados con `TimeSeriesSplit` sobre `X_train` (nunca `KFold`: respeta el orden del tiempo); el `Pipeline` de imputación, transformación (Yeo-Johnson) y escalado; la separación en `X_*` e `y_*`; `data_lineage.json`; y el esqueleto del punto de entrada de `src/preprocessing_pipeline.py`.
- Después de integrar las tareas 1 y 2: correr la selección definitiva y justificar el conjunto final. Quitar las filas que no sirven para entrenar (sin objetivo, o con vacíos en las variables elegidas si el modelo lo exige). Guardar `data/processed/dataset_final.csv` **sin escalar**, agregar su checksum a `data/checksums.json` y documentar el tamaño final de `X_train`, `X_val` y `X_test`.
- Decisiones que quedan para esta tarea: cómo imputar los vacíos (ADR 0008, punto 4), qué hacer con las columnas que dejan de medirse en validación y prueba (por ejemplo, las razones de la Línea 5 del KPI) y si conviene transformar cada modelo (ADR 0010).

## 6. Integración y cierre (en orden)
1. **Integrar las tareas 1 y 2:** merge de las dos ramas en `feature/preprocessing`. En `src/preprocessing_pipeline.py` queda `construir_variables(tabla)`, que llama a las dos funciones. Verificar mismas filas que la tabla limpia y sin columnas repetidas.
2. **Selección definitiva** y revisión del conjunto elegido entre todos.
3. **Cierre:** ejecutar `notebooks/preprocesamiento.ipynb` de cero (reiniciar kernel y correr todo), `python -m pytest tests -q`, y abrir el Pull Request de `feature/preprocessing` a `main` con "Create a merge commit". Antes del merge, otra persona lo revisa.

Si algo se atrasa, la prioridad es que existan `dataset_final.csv`, `data_lineage.json` y la selección con 2 métodos. Las interacciones más elaboradas se pueden recortar.

## 7. Si ayuda una IA
Pasarle este archivo y `docs/decisions/`, decir qué tarea se tiene, y pedirle: una decisión por vez, el código explicado línea por línea, pruebas junto con cada función y la justificación para la notebook. Verificar que respete las reglas de la sección 3, sobre todo la 1 y la 2.
