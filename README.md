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

`predict.py` carga el modelo desde `models/model_registry.json` y escribe las
predicciones diarias (una fila por día, correspondientes al consumo del día D+1) en
`results/predicciones.csv`.

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
- [ ] Fase 3 - Entrenamiento y versionado de modelos
- [ ] Fase 4 - Pipeline de predicción

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

### Conjunto oficial definitivo

El conjunto oficial del proyecto tiene **15 variables**, incluida `es_fin_de_anio`,
con los nombres y el orden registrados en `data/processed/data_lineage.json`.
El selector y los ejecutores vigentes de entrenamiento y evaluación utilizan ese
mismo conjunto. No se implementa compatibilidad permanente con otra selección.
Los artefactos de experimentos anteriores se conservan únicamente como evidencia
histórica, con sus identificadores y contratos originales.

Este PR corresponde al modelado, la optimización y la evaluación. Los puntos de
entrada genéricos `train_model.py` y `predict.py` siguen siendo plantillas; el
pipeline final de predicción no se presenta como implementado en este desarrollo.
