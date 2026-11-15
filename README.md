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

Este PR corresponde al modelado, la optimización y la evaluación. Los puntos de
entrada genéricos `train_model.py` y `predict.py` siguen siendo plantillas; el
pipeline final de predicción no se presenta como implementado en este desarrollo.
