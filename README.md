# Trabajo Práctico Final - Predicción de Consumo Energético Industrial

Laboratorio de Datos II - Ingeniería en Inteligencia Artificial - FICH (UNL)

Pipeline de machine learning completo y reproducible para predecir el consumo eléctrico
diario del sistema de refrigeración (`Frio (kW)`) del **día siguiente** en una planta
cervecera mexicana, con buenas prácticas de MLOps: entorno reproducible, control de
versiones, versionado de datos y modelos, y tracking de experimentos.

La consigna completa está en [`consigna.md`](consigna.md).

## Integrantes

| Nombre | Usuario de GitHub |
|---|---|
| Alesio Ríos | AlesioRios |
| Juan Cruz Brown | jcbrown910 |
| Milagros González | miligonzalzz |

## Estructura del proyecto

```
.
├── data/
│   ├── Totalizadores Planta de Cerveza*.xlsx   (NO versionados, ver "Datos")
│   ├── checksums.json                          (huellas SHA-256 de los Excel y del CSV inicial)
│   └── processed/
│       ├── dataset_unificado_inicial.csv   (Fase 1: solo Consolidado EE, una fila por día)
│       ├── dataset_final.csv               (Fase 2: 15 variables sin escalar, objetivo y tramo)
│       ├── dataset_final.csv               (15 variables oficiales sin escalar, objetivo y tramo)
│       └── data_lineage.json               (Fase 2: reglas aprendidas, columnas y dimensiones)
├── models/
│   ├── modelo_v1.0.0.pkl
│   └── model_registry.json
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocesamiento.ipynb              (Fase 2: de los Excel al dataset final; sección 7 = cierre)
│   ├── (un notebook por tarea de la Fase 2: variables, selección y Pipeline)
│   └── modelado.ipynb
├── src/
│   ├── preprocessing_pipeline.py           (punto de entrada de la Fase 2)
│   ├── limpieza.py, variables_*.py         (limpieza y variables nuevas)
│   ├── seleccion_*.py, pipeline_sklearn.py (selección de variables y Pipeline)
│   ├── modelado_comun.py                   (Fase 3: carga del dataset, baselines y registro de experimentos)
│   ├── train_model.py
│   ├── predict.py
│   └── auxiliar_functions.py
├── results/
│   ├── seleccion/                          (Fase 2: resultados congelados de la selección)
│   ├── predicciones.csv
│   └── experiment_logs.csv
├── tests/                                      (pruebas con datos sintéticos)
├── docs/
│   └── decisions/                              (decisiones de diseño, una por archivo)
├── .gitignore
├── README.md
├── requirements.txt
└── consigna.md
```

Los archivos se van agregando a medida que avanzan las fases (ver "Estado por fase").

## Requisitos previos

- [Git](https://git-scm.com/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (o Anaconda)
- Python 3.12 (lo instala conda al crear el entorno, no hace falta tenerlo en el sistema)

## Puesta en marcha

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/AlesioRios/Trabajo-Practico-Final-LD2.git
   cd Trabajo-Practico-Final-LD2
   ```

2. Crear y activar el entorno aislado con conda:

   ```bash
   conda create -n cervecera_env python=3.12 -y
   conda activate cervecera_env
   ```

3. Instalar las dependencias con uv, a partir del `requirements.txt` congelado:

   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

4. Conseguir los datos y ubicarlos en `data/` (siguiente sección).

## Datos

Los archivos Excel con los totalizadores horarios de la planta **no están en el
repositorio** (están excluidos en `.gitignore`). Se obtienen del aula virtual de la
cátedra (E-FICH, Laboratorio de Datos II, sección del Trabajo Práctico Final). Hay que
copiarlos tal cual, sin renombrarlos ni abrirlos y guardarlos, dentro de la carpeta `data/`.

Para comprobar que se tienen exactamente los mismos archivos que el equipo, el repositorio
versiona `data/checksums.json`, con el hash SHA-256 de cada archivo. Si alguna huella no
coincide, el archivo es distinto y los resultados no serán comparables.

> `data/checksums.json` ya se generó en la Fase 1 (sección 26 de `notebooks/eda.ipynb`): guarda las huellas de los tres Excel y de `data/processed/dataset_unificado_inicial.csv`, que contiene solo la hoja Consolidado EE (los tres archivos unidos, una fila por día); las demás hojas se unen en la Fase 2. Se verifica con `verificar_huellas` de `src/auxiliar_functions.py`.

## Cómo ejecutar el proyecto

Siempre con el entorno activado (`conda activate cervecera_env`) y desde la raíz del
repositorio.

| Qué | Comando | Disponible desde |
|---|---|---|
| Análisis exploratorio | `jupyter lab notebooks/eda.ipynb` | Fase 1 |
| Preprocesamiento | `jupyter lab notebooks/preprocesamiento.ipynb` y `python src/preprocessing_pipeline.py` | Fase 2 |
| Modelado y entrenamiento | `jupyter lab notebooks/modelado.ipynb` y `python src/train_model.py` | Fase 3 |
| Predicción sobre un Excel nuevo | `python src/predict.py "<ruta al archivo .xlsx>"` | Fase 4 |

`python src/preprocessing_pipeline.py` recorre toda la Fase 2: arma la tabla de 337 columnas, aplica las 15
variables oficiales (incluida `es_fin_de_anio`), ajusta el `Pipeline` solo con entrenamiento y guarda `data/processed/dataset_final.csv` (sin escalar),
su huella en `data/checksums.json` y `data/processed/data_lineage.json`. Con `--sin-guardar` hace las mismas
comprobaciones sin escribir archivos. La primera vez reconstruye la tabla desde los Excel (unos 6 minutos) y
después reutiliza la copia de trabajo de `data/interim/`.

`predict.py` carga el modelo desde `models/model_registry.json` (por defecto la versión más alta, hoy la
1.0.0: Lasso con 14 variables) y escribe las
predicciones diarias (una fila por día, correspondientes al consumo del día D+1) en
`results/predicciones.csv`. Lee del Excel solo las hojas y columnas que usan las variables del modelo,
siempre por nombre, e ignora las demás. Si falta alguna, termina con un mensaje que dice cuál (código de salida 1,
sin escribir el CSV). Predice todos los días y avisa por consola cuáles tenían datos incompletos, por
ejemplo los primeros del archivo, que no tienen historia para los rezagos (ADR 0025). No usa internet.

## Reproducibilidad

- Semilla fija: `SEMILLA = 42`, usada en toda partición y en todo estimador que acepte
  `random_state`. Dos ejecuciones sucesivas deben dar los mismos números.
- Las dependencias están congeladas con `uv pip freeze` en `requirements.txt`.
- El modelo serializado lleva versión semántica (`models/modelo_v1.0.0.pkl`) y el commit
  correspondiente se etiqueta con `git tag -a v1.0.0`, de modo que la versión del registro
  y la del repositorio sean la misma.

## Estrategia de ramas (Git Flow)

- `main` permanece siempre estable. No se commitea directo sobre ella después del commit inicial.
- Cada fase se desarrolla en su propia rama y se integra a `main` mediante Pull Request:

| Fase | Rama |
|---|---|
| 1 - EDA | `feature/eda` |
| 2 - Preprocesamiento | `feature/preprocessing` |
| 3 - Modelado | `feature/modeling` |
| 4 - Pipeline de predicción | `feature/prediction-pipeline` |

- Cada integrante commitea con su propio usuario, y `git log --oneline` debe mostrar los
  commits de todos, distribuidos en el tiempo.

## Estado por fase

- [x] Fase 0 - Entorno, control de versiones y trazabilidad
- [x] Fase 1 - Análisis exploratorio de datos (EDA)
- [x] Fase 2 - Preprocesamiento y selección de variables
- [x] Fase 3 - Entrenamiento y versionado de modelos
- [x] Fase 4 - Pipeline de predicción

## Decisiones y acuerdos

- [`docs/acuerdos-de-datos.md`](docs/acuerdos-de-datos.md): formato de los datos, funciones compartidas y reglas de trabajo del equipo.
- [`docs/decisions/`](docs/decisions/): decisiones de diseño con su justificación, una por archivo.

Selección de variables: [ADR 0015](docs/decisions/0015-seleccion-de-variables.md).

## XGBoost: resultados y revisión sin entrenar

El desarrollo de `feature/modeling-xgboost` contiene tres estudios independientes:

| Estudio | Variables | Trials | Mejor trial | MAE CV | MAE validación independiente |
|---|---:|---:|---:|---:|---:|
| xgboost-temporal-v1 | 14 | 100 | 88 | 3986.65 | No evaluada |
| xgboost-temporal-15-features-v1 | 15 | 100 | 56 | 3957.96 | 2221.76 |
| xgboost-temporal-15-features-refinado-v1 | 15 | 60 | 49 | 3941.22 | 2229.22 |

La validación independiente usa las mismas 171 filas comunes (29/10/2022–26/04/2023),
con ajuste solo sobre 830 filas de entrenamiento. El menor MAE CV del refinamiento
no se tradujo en menor MAE de validación. No se eligió un modelo de producción ni
se reentrenó con entrenamiento más validación; no se evaluó test.

Leer el [ADR 0019](docs/decisions/0019-xgboost-optuna.md), los JSON/CSV bajo
`results/xgboost*/` y `notebooks/modelado_xgboost.ipynb`. La notebook consulta
resultados guardados y muestra PNG existentes; su primera sección usa el dataset
histórico de `data/processed/seleccion_original_14/`. No ejecutar los scripts de
búsqueda ni evaluación para revisar un PR. Si falta SQLite histórico, la notebook
lo restaura desde su ZIP; no crea trials. No sobrescribe figuras al consultarlas.
Las salidas ejecutadas se conservaron; las correcciones de lectura de esta auditoría
se verificaron estáticamente y no se reejecutó la notebook.

Los scripts de búsqueda son entradas explícitas para experimentos autorizados:
`src/modelado_xgboost_15.py` y `src/modelado_xgboost_15_refinado.py`.
`src/modelado_xgboost.py` conserva el identificador histórico de 14 variables,
pero el validador vigente exige 15: no usar su CLI para reproducir la corrida
original. El código exacto original está en el commit registrado en su resumen;
los datos originales están archivados. La reanudación verifica hashes y rechaza
cambios de datos, versiones o código. Los ZIP y JSON bastan para consultar resultados
sin reanudar ni restaurar estudios nuevos.

### Entorno y portabilidad

La ejecución registrada usó Python 3.12.14 y las versiones exactas de
`requirements.txt`, incluidos Optuna 5.0.0, XGBoost 3.4.1 y scikit-learn 1.9.1.
El nombre `cervecera_env` es convencional y se puede recrear; las rutas se resuelven
respecto del repositorio, sin depender de la carpeta personal del autor.

```bash
conda create -n cervecera_env python=3.12.14
conda activate cervecera_env
python -m pip install uv==0.13.1
uv pip install -r requirements.txt
uv pip check
```

En OneDrive, `uv pip install --link-mode=copy -r requirements.txt` evita enlaces
permanentes. Para reproducir hashes históricos del código se requiere conservar
los finales de línea de Windows (`core.autocrlf=true`), además de datos y versiones.
La semilla sola no garantiza igualdad entre sistemas o versiones diferentes.

El evaluador de la primera búsqueda de 15 variables lee por defecto los resultados
ya guardados. Solo si se lo usa expresamente para una nueva evaluación y faltan
esas salidas necesita la referencia Git `origin/feature/modeling-rf-lineales-experimentos`.
Esa rama remota no es necesaria para consultar los resultados versionados; no se
recomienda eliminarla sin revisar ese mecanismo de reproducción.

### Alcance del PR individual

El PR de `feature/modeling-xgboost` apunta a `feature/modeling` y contiene el
trabajo de XGBoost y los cambios propios vinculados al indicador de fin de año.
No incorpora implementaciones, notebooks ni commits de Random Forest, Ridge o Lasso.
`git fetch` solo actualiza referencias remotas; no integra esas ramas.

El log de esta rama contiene cuatro baselines heredados de la base y 139 registros
propios de XGBoost. Las uniones de logs y los informes generales de auditoría se
mantienen fuera del repositorio y no forman parte de este PR. La unión del registro
general corresponde a la integración del equipo, conservando todas las filas originales.

`results/xgboost_15_validacion/referencias_companera.csv` y `comparacion.csv` conservan
referencias atribuidas utilizadas en la comparación histórica de XGBoost. No son
implementaciones ni nuevas corridas de los modelos de otra integrante. No se borran
para aparentar una ejecución distinta de la realizada.

## Primera búsqueda LightGBM

Configuración, ejecución y límites: [ADR 0021](docs/decisions/0021-lightgbm-optuna.md).
Ejecutar `python src/modelado_lightgbm.py` en esta copia con Python 3.12.
Produce 100 trials totales, evaluación del ganador en validación y tres PNG.
Resultados: `results/lightgbm/`; modelo: `models/lightgbm/`. No evalúa test.

## Segunda búsqueda LightGBM

Metodología y auditoría: [ADR 0022](docs/decisions/0022-lightgbm-segunda-busqueda.md).
Ejecutar python src/modelado_lightgbm_2.py con Python 3.12.
Resultados, comparación y PNG: results/lightgbm_2/. Modelo: models/lightgbm_2/.
Se conservan los 100 trials de la primera búsqueda y todos sus artefactos.


### Revisión de LightGBM sin entrenar

Leer `results/lightgbm_2/informe.md`, `comparacion.csv`, `validacion.json` y los
PNG de `results/lightgbm_2/graficos/`; la primera búsqueda permanece en
`results/lightgbm/`. No ejecutar los scripts de búsqueda para consultar resultados.
Las copias `codigo_ejecutado/` documentan lo ejecutado; se corre el código de `src/`.
Las huellas y versiones registradas permiten auditar las corridas históricas;
la semilla por sí sola no garantiza igualdad entre sistemas y versiones diferentes.

El PR individual de `feature/modeling-lightgbm` apunta a `feature/modeling` y
contiene únicamente LightGBM y su soporte de datos de 15 variables. No incorpora
implementaciones ni commits de RF/Ridge/Lasso. Su log tiene cuatro baselines
heredados y 86 registros propios. Las uniones de logs de todos los modelos y los
informes de auditoría general están fuera del repositorio; no entran en este PR.

### Conjunto oficial definitivo

El conjunto oficial del proyecto tiene **15 variables**, incluida `es_fin_de_anio`,
con idénticos nombres y orden que XGBoost y que `data/processed/data_lineage.json`.
`src/seleccion_definitiva.py` devuelve siempre esas 15 variables. El preprocesamiento
que lo consulta, los ejecutores de entrenamiento y evaluación y los modelos de
validación guardados utilizan el mismo conjunto. No se implementa compatibilidad
permanente con otra selección ni se modifican los artefactos históricos.

Esta sección corresponde al modelado, la optimización y la evaluación. `train_model.py`
entrena y evalúa el modelo final (ver "Cierre de Fase 3: Lasso con catorce entradas") y
`predict.py` es el pipeline de predicción de la Fase 4 (ADR 0025).

### Diagnóstico offline de LightGBM (15/11/2026)

El [informe de diagnóstico](results/lightgbm_diagnostico_validacion/informe.md)
completa los gráficos y residuos de las dos búsquedas LightGBM desde sus predicciones
de validación guardadas. Se conservan cinco PNG comparativos, residuos por fila y
mes, errores mayores, parámetros completos y auditoría del tracking. El notebook
propio es [modelado_lightgbm.ipynb](notebooks/modelado_lightgbm.ipynb); solo consulta
esos resultados. No carga datasets, modelos ni estudios mutables de Optuna.

Desde la raíz, con Python 3.12 y las dependencias existentes:

```powershell
python -B src/diagnostico_lightgbm.py
python -B src/actualizar_notebook_lightgbm.py
python -B -m pytest tests/test_diagnostico_lightgbm.py -q -p no:cacheprovider
```

El segundo comando requiere el kernel `cervecera_env`; si falta, registrarlo desde
ese entorno con `python -m ipykernel install --user --name cervecera_env`.
No se debe ejecutar la suite de modelado ni los scripts de búsquedas para revisar
el diagnóstico. Sus pruebas propias no contienen entrenamiento ni predicción.
Se reutiliza la aritmética pura del diagnóstico XGBoost ya integrado en la base,
sin modificar su código ni depender de una copia de trabajo externa.

La rama se actualizó sobre `feature/modeling`: el log común ahora incluye otras
familias, pero los registros LightGBM siguen siendo 86 (84 COMPLETE y dos
validaciones). La auditoría verifica su firma semántica y los originales por
SHA-256. La granularidad por experimento, los timestamps sin zona horaria y el
árbol sucio histórico quedan explícitos; no se fabrican registros retrospectivos.
Las referencias anteriores a logs separados describen las publicaciones originales.

El diagnóstico descriptivo de 3.3 queda completo para estos candidatos. La
importancia de variables del ganador global, notebook común, entrenamiento final,
versionado semántico y registry siguen pendientes de integración. No se abrió
test ni se modificaron las 15 variables, resultados originales o módulos comunes.

### Diagnóstico de validación sin ejecutar modelos (15/11/2026)

El [informe de diagnóstico](results/xgboost_diagnostico_validacion/informe.md)
completa los gráficos de predicción frente a real y el análisis de residuos de
las dos búsquedas vigentes usando exclusivamente sus CSV de validación guardados.
La carpeta contiene cinco PNG comparativos, residuos por fila y mes, errores
mayores, parámetros completos, espacios de búsqueda y auditoría del tracking.
`verificacion.json` registra las huellas de los originales y el entorno del análisis.
No se abre `dataset_final.csv`, no se cargan estimadores y no se escribe en el log.

Desde la raíz del repositorio, con `cervecera_env` (Python 3.12):

```powershell
python -B src/diagnostico_xgboost.py
python -B src/actualizar_diagnostico_notebook.py
python -B -m pytest tests/test_diagnostico_xgboost.py -q -p no:cacheprovider
```

El segundo comando requiere el kernel Jupyter `cervecera_env`: ejecuta únicamente
el apéndice autónomo de las secciones 13–16 de `notebooks/modelado_xgboost.ipynb`.
Si el entorno todavía no está registrado en Jupyter, hacerlo desde ese entorno con
`python -m ipykernel install --user --name cervecera_env --display-name cervecera_env`.
Preserva las fuentes y salidas de las 33 celdas históricas; no ejecuta el notebook
completo ni restaura estudios de Optuna. La lectura del informe y de los PNG no
requiere Jupyter. El archivo de pruebas indicado no contiene entrenamiento.
No ejecutar la suite completa ni los evaluadores para esta consulta: contienen
ajustes de modelos. No se agregaron dependencias a `requirements.txt`.

El log conserva 139 registros XGBoost, con parámetros, métricas y timestamps
originales verificados. La granularidad es por experimento completo, no por
ajuste de pliegue; los trials podados se conservan fuera del CSV. Esta limitación
de la sección 3.2 queda documentada, sin inventar registros ni tiempos históricos.
La diferencia entre métricas del CSV decimal y del log se explica por float32;
su recuperación reproduce las métricas históricas sin modificar predicciones.

La [auditoría de ambos modelos](docs/auditoria-fase3-modelos.md) contrasta las
secciones 3.1–3.4 del PDF oficial de referencias y la ubicación de los entregables.
LightGBM se inspeccionó en su rama sin modificarlo. La integración al notebook
común, elección global, importancia de variables del ganador y registro semántico
siguen pendientes. No se presenta la Fase 3 completa como finalizada.


## Cierre de Fase 3: Lasso con catorce entradas

El cierre fija Lasso (`alpha=75`, `max_iter=20000`, `random_state=42`) y pipeline A
(mediana y StandardScaler) sobre las catorce variables del linaje histórico
`data/processed/seleccion_original_14/data_lineage.json`. El CSV vigente no se cambia.
La comparación y sus fuentes están en `results/comparacion_modelos_final.md`.

Con Python 3.12, activar `conda activate cervecera_env` y usar las dependencias
congeladas de `requirements.txt`. Para un cierre nuevo autorizado, con código
commiteado y sin entrenamiento ni evaluación final previos:

```powershell
python src/train_model.py entrenar
python src/train_model.py evaluar-test-una-vez
python src/informe_lasso_final.py
python -m pytest tests -q
```

La segunda orden requiere autorización para consumir test: crea un marcador antes
de cargar sus observaciones. No borrar ese marcador para reintentar. Si ya existe
evidencia final, solo se recuperan y verifican los resultados guardados. Las órdenes
anteriores no deben usarse para crear otra evaluación tras observar el resultado.
Para auditar la versión publicada sin volver a usar test:

```powershell
python src/train_model.py consultar
python src/train_model.py verificar-entrenamiento
python src/verificar_cierre_lasso.py
python src/informe_lasso_final.py
```

La verificación de entrenamiento reproduce exclusivamente train+validation, compara
sus predicciones archivadas y no modifica el artefacto, logs ni registry. `consultar`
recalcula aritmética desde predicciones guardadas; no carga el dataset test ni predice.
El notebook `notebooks/modelado.ipynb`, ejecutado con kernel `cervecera_env`, solo lee
archivos de evidencia y puede repetirse sin consumir test. El notebook histórico de
baselines se conserva en Git; las búsquedas por familia y sus ramas se enlazan en el
informe de comparación.

Para cargar el pipeline final y recuperar su contrato de entrada:

```python
import joblib
modelo = joblib.load("models/modelo_v1.0.0.pkl")
columnas = modelo.cierre_metadata_["columnas"]
# X_nuevo es una tabla ya preparada con las mismas reglas históricas de Fase 2.
predicciones = modelo.predict(X_nuevo[columnas])
```

La ruta efectiva y versión están en `models/model_registry.json`; si 1.0.0 ya estaba
ocupada se reserva la siguiente versión patch sin sobrescribirla. Se conserva la
validación original separada del test local y de las métricas de ajuste. El hash
`commit` del registry identifica el código previo al entrenamiento, también incluido
en el artefacto. El commit posterior de resultados recibe el tag anotado `v1.0.0`
(o la versión efectiva); no se intenta incluir el hash de ese commit dentro de sí
mismo. No se publica ni integra a main automáticamente.

Resultados, predicciones auditables, residuos, coeficientes, gráficos y controles:
`results/lasso_final/`. El resultado local no acredita el umbral del test oculto.
Validation fue utilizada antes para selección de variables. DIA puede repetirse y
el horizonte certificado es el bloque siguiente; la disponibilidad real de cada
entrada a las 23:59 sigue sin demostrarse. La Fase 4 usa este modelo desde `src/predict.py` (ADR 0025).
