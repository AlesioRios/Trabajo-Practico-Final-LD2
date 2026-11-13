> **Estado vigente (Fase 2.3):** Consultar [selección, resultados y limitaciones](docs/informes/seleccion_fase23_estado_actual.md). Spearman definitivo: 14 variables y 27 columnas; elegido por parsimonia dentro del 1 % del MAE de Pearson. Test reservado.

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
│   ├── Totalizadores Planta *.xlsx   (NO versionados, ver "Datos")
│   ├── checksums.json                          (huellas SHA-256 de los Excel y del CSV inicial)
│   └── processed/
│       ├── dataset_unificado_inicial.csv   (Fase 1: solo Consolidado EE, una fila por día)
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

Los nombres registrados en `data/checksums.json` y utilizados por esta copia del proyecto son:

- `Totalizadores Planta 2020-2022.xlsx`
- `Totalizadores Planta 2021-2023.xlsx`
- `Totalizadores Planta 2022-2023.xlsx`

El cargador espera exactamente los tres originales que coincidan con `Totalizadores Planta *.xlsx`. No deben mezclarse copias adicionales en `data/`.

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
| Preprocesamiento hasta ingeniería | `jupyter lab notebooks/preprocesamiento.ipynb` y `python src/preprocessing_pipeline.py` | Fase 2; selección y guardado final pendientes de conexión |
| Diagnóstico de selección | `jupyter lab notebooks/seleccion_de_variables.ipynb` | Selección temporal reproducible; ver sección siguiente |
| Modelado y entrenamiento | `jupyter lab notebooks/modelado.ipynb` y `python src/train_model.py` | Fase 3, pendiente de implementación |
| Predicción sobre un Excel nuevo | `python src/predict.py "<ruta al archivo .xlsx>"` | Fase 4, pendiente de implementación |

`src/train_model.py` y `src/predict.py` contienen actualmente solo su descripción; el notebook de modelado todavía no está en el repositorio. La carga de modelos y la escritura de predicciones son entregables futuros, no comandos operativos de esta etapa.

Para trabajar con el notebook de selección, iniciar Jupyter desde la raíz del repositorio y elegir el kernel de `cervecera_env`. Si no aparece, registrarlo con el entorno activado:

```bash
python -m ipykernel install --user --name cervecera_env --display-name "Python (cervecera_env)"
```

El notebook también puede localizar el repositorio desde un subdirectorio suyo. Lee artefactos congelados sin entrenar; no necesita Excel ni cachés. Reconstruir características sí requiere los originales. El punto de entrada usa `data/` relativo a su propio archivo por defecto; una ruta relativa proporcionada mediante `--carpeta-datos` se interpreta desde el directorio del comando.

La selección está resuelta: no repetirla ni ejecutar la comparación congelada. Consultar la [guía de consumo](docs/guia-fase2.md) y el [historial](docs/informes/metodologia_fase23.md).

## Reproducibilidad

- Semilla fija: `SEMILLA = 42`, usada en toda partición y en todo estimador que acepte
  `random_state`. Dos ejecuciones sucesivas deben dar los mismos números.
- Las dependencias están congeladas con `uv pip freeze` en `requirements.txt`.
- El modelo serializado lleva versión semántica (`models/modelo_v1.0.0.pkl`) y el commit
  correspondiente se etiqueta con `git tag -a v1.0.0`, de modo que la versión del registro
  y la del repositorio sean la misma.

La caché de tabla diaria del punto de entrada compartido es `data/interim/tabla_completa.pkl`, ignorada por Git. La versión recibida de `feature/preprocessing` decide su reutilización por fechas de modificación de los originales y del código; no ofrece el manifiesto de huellas de las ampliaciones locales anteriores. `--sin-cache` fuerza reconstrucción. La comprobación de SHA-256 de fuentes con `data/checksums.json` sigue siendo separada. El notebook de selección usa artefactos versionados y no necesita esa caché.

Las dependencias de carga, ingeniería, selección, Jupyter y pruebas están declaradas en `requirements.txt`. La auditoría estática no demuestra que esas versiones puedan instalarse o ejecutarse con Python 3.12 en otra máquina; esa validación está pendiente. No se alteraron versiones durante la corrección. El contrato oficial de selección y el mapa explícito están documentados en [el contrato de integración](docs/informes/integracion_pr_fase23.md#contrato-con-el-pipeline).

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

- [ ] Fase 0 - Entorno local verificado; cierre de Git e instalación independiente pendientes
- [x] Fase 1 - Análisis exploratorio de datos (EDA)
- [ ] Fase 2 - Preprocesamiento y selección de variables
- [ ] Fase 3 - Entrenamiento y versionado de modelos
- [ ] Fase 4 - Pipeline de predicción

## Decisiones y acuerdos

- [`docs/acuerdos-de-datos.md`](docs/acuerdos-de-datos.md): formato de los datos, funciones compartidas y reglas de trabajo del equipo.
- [`docs/decisions/`](docs/decisions/): decisiones de diseño con su justificación, una por archivo.

## Selección de variables: ejecución y trazabilidad

Los resultados congelados y su interpretación figuran en [el informe de Fase 2.3](docs/informes/seleccion_fase23_estado_actual.md). Los cuatro métodos individuales usaron cuatro pliegues externos y tres internos; Forward tradicional max60 utilizó tres pliegues temporales con gap=1. Sus errores internos no son una comparación independiente. El notebook solo lee resultados guardados.

La comparación común ya se ejecutó y sus artefactos están congelados. No repetir `src/comparar_selecciones_validation.py`. Test permanece reservado. Se requiere Python 3.12 y las dependencias de `requirements.txt`.

## Conjunto oficial para la siguiente fase

src/seleccion_definitiva.py carga nombres, orden, ingeniería y preparación desde el artefacto definitivo con verificación de huella. seleccionar_variables(X_train, y_train) consume esa lista sin recalcular selección. Las configuraciones explícitas permiten reproducir el historial. La [guía](docs/guia-fase2.md) explica cómo reconstruir sin fit.

Pearson obtuvo 2239.58 kW frente a 2247.82 de Spearman en las mismas 176 observaciones. La diferencia de 0.37 % cumple el margen previo del 1 %; Spearman usa 14 variables frente a 19. Validation participó de la elección y no estima rendimiento final independiente. Las 27 columnas incluyen 13 indicadores. Calendario físico y horarios de publicación siguen pendientes de confirmación externa. [Informe](docs/informes/seleccion_fase23_estado_actual.md). Fase 2.4 y modelos definitivos pendientes.

La integración previa al PR, reproducción y contrato con el Pipeline recibido se documentan en [integración de Fase 2.3](docs/informes/integracion_pr_fase23.md). La guía compartida conserva su estructura; la fila de selección señala Spearman como definitivo.
