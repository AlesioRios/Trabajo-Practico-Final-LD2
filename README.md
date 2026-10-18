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
│   ├── checksums.json                          (huellas SHA-256 de los Excel)
│   └── processed/
│       ├── dataset_final.csv
│       └── data_lineage.json
├── models/
│   ├── modelo_v1.0.0.pkl
│   └── model_registry.json
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocesamiento.ipynb
│   └── modelado.ipynb
├── src/
│   ├── preprocessing_pipeline.py
│   ├── train_model.py
│   ├── predict.py
│   └── auxiliar_functions.py
├── results/
│   ├── predicciones.csv
│   └── experiment_logs.csv
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

> `data/checksums.json` y la función que lo verifica se generan en la Fase 1.

## Cómo ejecutar el proyecto

Siempre con el entorno activado (`conda activate cervecera_env`) y desde la raíz del
repositorio.

| Qué | Comando | Disponible desde |
|---|---|---|
| Análisis exploratorio | `jupyter lab notebooks/eda.ipynb` | Fase 1 |
| Preprocesamiento | `jupyter lab notebooks/preprocesamiento.ipynb` y `python src/preprocessing_pipeline.py` | Fase 2 |
| Modelado y entrenamiento | `jupyter lab notebooks/modelado.ipynb` y `python src/train_model.py` | Fase 3 |
| Predicción sobre un Excel nuevo | `python src/predict.py "<ruta al archivo .xlsx>"` | Fase 4 |

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
- [ ] Fase 1 - Análisis exploratorio de datos (EDA)
- [ ] Fase 2 - Preprocesamiento y selección de variables
- [ ] Fase 3 - Entrenamiento y versionado de modelos
- [ ] Fase 4 - Pipeline de predicción
