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
│   ├── Totalizadores Planta *.xlsx            (NO versionados, ver "Datos")
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

   Ante errores de enlaces permanentes en Windows / OneDrive, la alternativa comprobada es:

   ```powershell
   uv pip install --link-mode copy --no-cache -r requirements.txt
   ```

   La instalación se comprobó en Conda con `cervecera_env` y Python 3.12.16.
   Una advertencia aislada de `Invalid RECORD entry in pip` no impidió completarla
   y no requiere cambiar `requirements.txt`. Las corridas históricas de modelado
   registraron Python 3.12.14; sus versiones se conservan en los manifiestos.

4. Conseguir los datos y ubicarlos en `data/` (siguiente sección).

## Datos

Los archivos Excel con los totalizadores horarios de la planta **no están en el
repositorio** (están excluidos en `.gitignore`). Se obtienen del aula virtual de la
cátedra (E-FICH, Laboratorio de Datos II, sección del Trabajo Práctico Final). Hay que
ubicarlos en `data/` con estos nombres literales:

```text
data/Totalizadores Planta 2020-2022.xlsx
data/Totalizadores Planta 2021-2023.xlsx
data/Totalizadores Planta 2022-2023.xlsx
```

Si la descarga tiene otro nombre, renombrar el archivo sin editar su contenido
ni abrirlo y guardarlo desde Excel. No adaptar las celdas del EDA a otros nombres.

| Nombre descargado | Nombre requerido en `data/` |
|---|---|
| `Totalizadores Planta 2020_2022.xlsx` | `Totalizadores Planta 2020-2022.xlsx` |
| `Totalizadores Planta - 2021_2023.xlsx` | `Totalizadores Planta 2021-2023.xlsx` |
| `Totalizadores Planta 2022_2023.xlsx` | `Totalizadores Planta 2022-2023.xlsx` |

En la prueba de reproducibilidad, los bytes de los tres originales coincidieron
con sus SHA-256 archivados: el problema era el nombre del archivo.

Para comprobar que se tienen exactamente los mismos archivos que el equipo, el repositorio
versiona `data/checksums.json`, con el hash SHA-256 de cada archivo. Si alguna huella no
coincide, el archivo es distinto y los resultados no serán comparables.

> `data/checksums.json` ya se generó en la Fase 1 (sección 26 de `notebooks/eda.ipynb`): guarda las huellas de los tres Excel y de `data/processed/dataset_unificado_inicial.csv`, que contiene solo la hoja Consolidado EE (los tres archivos unidos, una fila por día); las demás hojas se unen en la Fase 2. Se verifica con `verificar_huellas` de `src/auxiliar_functions.py`.

## Cómo ejecutar el proyecto

Activar `conda activate cervecera_env` y ejecutar desde la raíz del repositorio.
Antes de ejecutar cualquier notebook en JupyterLab, verificá que esté seleccionado
el entorno/kernel del proyecto (`cervecera_env`).

| Tarea | Comando | Alcance |
|---|---|---|
| Análisis exploratorio | `jupyter lab notebooks/eda.ipynb` | Fase 1; requiere los Excel |
| Preprocesamiento | `python src/preprocessing_pipeline.py` | Fase 2; regenera datos, huellas y linaje |
| Consulta del cierre publicado | `python src/train_model.py consultar` | Lee predicciones y métricas archivadas |
| Reproducción del entrenamiento | `python src/train_model.py verificar-entrenamiento` | Reajusta solo train+validation; exige el hash histórico del dataset |
| Auditoría histórica sin escritura | `python src/verificar_cierre_lasso.py --historico --sin-guardar` | Contrasta el cierre y los archivos del commit de referencia |
| Predicción sobre un Excel nuevo | `python src/predict.py "<ruta al archivo .xlsx>"` | Fase 4; usa el modelo publicado |

`notebooks/preprocesamiento.ipynb` explica la Fase 2 y
`notebooks/modelado.ipynb` consulta la evidencia del cierre. La predicción normal
no requiere ejecutar notebooks ni volver a entrenar.

`train_model.py` requiere una acción explícita. `python src/train_model.py entrenar`
es una entrada para un entrenamiento nuevo autorizado, no un paso de consulta;
protege los artefactos ya publicados. `python src/train_model.py evaluar-test-una-vez`
es la evaluación test protegida e irrepetible: **no ejecutarla para consultar o
auditar**. No borrar `results/lasso_final/test_consumido.json` para reintentar.

### Preprocesamiento y exportación reproducible

El preprocesamiento genera 337 columnas tras limpieza y creación de variables,
aplica las **15 predictoras oficiales** (incluida `es_fin_de_anio`) y ajusta el preprocesador solo con entrenamiento.
Guarda `data/processed/dataset_final.csv` sin escalar, su huella en
`data/checksums.json` y `data/processed/data_lineage.json`. La primera reconstrucción
desde Excel tarda unos seis minutos; las siguientes reutilizan `data/interim/`.
`--sin-guardar` omite el CSV final y el linaje, pero la carga puede crear la caché
intermedia; para preservar toda la evidencia, usar una copia de ejecución separada.

El CSV histórico tiene 1184 filas y 20 columnas. En su orden físico, las posiciones
16–18 son `consumo_dia`, `baseline_media_movil`, `es_fin_de_anio`, seguidas por
`objetivo` y `tramo`. La exportación conserva ese orden sin cambiar valores ni el
contrato de las 15 predictoras. SHA-256 esperado:

```text
3bda0f26952d01412f5970de0fb5999659c6fb6965c4f28935ed3bc538134dec
```

Regenerar en otra copia y ejecutar luego `python src/train_model.py verificar-entrenamiento`
permite comprobar el ajuste histórico: 1006 filas train+validation,
`reproduce_train_validation: true`, `test_usado: false`. Esa verificación no modifica
el modelo, registry ni logs; requiere el dataset compatible con la huella archivada.
No incorporar salidas regeneradas ni sustituir las huellas históricas para forzar
una comprobación.

### Predicción habitual

`predict.py` lee `models/model_registry.json` y usa por defecto la versión más alta,
actualmente 1.0.0 (Lasso con 14 entradas). Escribe `results/predicciones.csv`, una
predicción por día para el consumo del bloque siguiente (D+1 nominal). Lee por
nombre solo las hojas y columnas necesarias; si falta alguna, informa cuál y sale
con código 1 sin escribir el CSV. Predice también las filas con datos incompletos
y las advierte por consola, por ejemplo al faltar historia para rezagos. No usa
internet. El contrato y los límites están en el [ADR 0025](docs/decisions/0025-pipeline-de-prediccion.md).

## Reproducibilidad y modos de auditoría

La semilla compartida es `SEMILLA = 42`; las dependencias están congeladas en
`requirements.txt`. La semilla sola no garantiza igualdad entre sistemas o versiones.
Los manifiestos conservan versiones y SHA-256; los hashes históricos de código
pueden depender de CRLF/LF (`core.autocrlf=true` en las corridas Windows).

```powershell
python src/train_model.py consultar
python src/train_model.py verificar-entrenamiento
python src/verificar_cierre_lasso.py --historico --sin-guardar
```

El auditor conserva dos comprobaciones distintas:

| Modo | Qué compara | Interpretación |
|---|---|---|
| Estricto (predeterminado): `python src/verificar_cierre_lasso.py --sin-guardar` | Archivos actuales contra firmas de `preparacion_cierre.json` | Puede fallar legítimamente si código o JSON cambiaron después del cierre |
| Histórico: `python src/verificar_cierre_lasso.py --historico --sin-guardar` | Archivos protegidos recuperados con `git show` desde `commit_base`, considerando la representación histórica CRLF | Comprueba la evidencia de referencia; no afirma que el árbol actual sea idéntico |

Ambos conservan los controles de modelo, dataset, predicciones archivadas, métricas,
registry, logs y dependencias. La salida identifica `modo_auditoria`. Sin
`--sin-guardar`, el auditor escribe `results/lasso_final/verificacion_final.json`;
para consultar sin alterar evidencia usar siempre esa opción. Requiere los commits
históricos disponibles: una clonación superficial incompleta puede no alcanzarlos.
El bloque `pruebas` procede del XML archivado de 219 pruebas; **el auditor no vuelve
a ejecutarlas**. Sus textos históricos describen el alcance del cierre original.

## Modelos y comparación de validación

Se estudiaron XGBoost, LightGBM, Random Forest, Ridge y Lasso. El
[informe de comparación](results/comparacion_modelos_final.md) y su
[CSV de fuentes](results/comparacion_modelos_fuentes.csv) conservan el ranking,
configuraciones completas, timestamps y procedencia por commit. También se mantiene
el experimento adicional de [Gradient Boosting](docs/decisions/0023-gradient-boosting-optuna.md).

La tabla resume el mejor candidato de Fase 3 de cada una de las cinco familias,
con ajuste sobre 830 filas y evaluación sobre las mismas **171 filas comunes** de
validación (29/10/2022–26/04/2023). Se comparan configuraciones completas de 14 o 15
entradas, no métricas de CV contra validación. Las cifras se redondean aquí; las
fuentes conservan todos los decimales.

| Familia | Entradas | Configuración identificadora | MAE validación (kW) | RMSE validación (kW) | R² |
|---|---:|---|---:|---:|---:|
| Lasso | 14 | Pipeline A, `alpha=75`, `max_iter=20000` | 2215.7964 | 2835.8038 | 0.467609 |
| XGBoost | 15 | Búsqueda inicial, trial 56, 1908 árboles, profundidad 3 | 2221.7606 | 2844.3074 | 0.464411 |
| Ridge | 14 | Pipeline A, `alpha=30` | 2225.4432 | 2863.2289 | 0.457262 |
| LightGBM | 15 | Segunda búsqueda, trial 99, 200 árboles, 8 hojas | 2276.7354 | 2873.1357 | 0.453499 |
| Random Forest | 15 | 200 árboles, profundidad 10, `min_samples_leaf=12`, `max_features=0.5` | 2284.5211 | 2954.7497 | 0.422011 |

Todos usan `random_state=42`. La validación ya se utilizó en selección de variables
y familias; no es una estimación final independiente. La ventaja de Lasso frente
a XGBoost es pequeña y no demuestra superioridad estadística. El ranking se limita
a candidatos de Fase 3: los antecedentes Ridge de Fase 2 con selección Pearson o
Spearman tienen errores menores en el mismo soporte de 171 filas y se documentan
en el informe. Los resultados sobre 176 filas y los MAE de CV no se mezclan con
esta tabla. No se afirma que Lasso sea el mejor de toda la historia del proyecto.

### XGBoost

Se conservan tres estudios: 100 trials con 14 variables (mejor trial 88, MAE CV
3986.65, sin validación independiente); 100 con 15 (trial 56, CV 3957.96,
validación 2221.76); y 60 de refinamiento (trial 49, CV 3941.22, validación 2229.22).
El refinamiento redujo CV pero no mejoró validación. El ganador de esta familia
usa `learning_rate=0.0020702351473759497` y profundidad 3; sus parámetros completos
están en [evaluacion.json](results/xgboost_15_validacion/evaluacion.json).

El [ADR 0019](docs/decisions/0019-xgboost-optuna.md),
[notebook](notebooks/modelado_xgboost.ipynb), JSON/CSV en `results/xgboost*/` y el
[informe de diagnóstico](results/xgboost_diagnostico_validacion/informe.md) conservan
búsquedas, predicciones, residuos, gráficos y auditoría. El notebook consulta
resultados guardados; la sección histórica usa `data/processed/seleccion_original_14/`.
Puede restaurar el SQLite histórico desde ZIP, sin crear trials. Para revisar basta
leer informes y PNG; no ejecutar búsquedas ni evaluadores. El CLI histórico de
`src/modelado_xgboost.py` usa un contrato anterior de 14 entradas frente al validador
vigente de 15; el commit original está en su resumen. Los ejecutores actuales de
15 son `src/modelado_xgboost_15.py` y `src/modelado_xgboost_15_refinado.py`.

El evaluador inicial consulta resultados guardados; su reproducción explícita si
faltan salidas depende de `origin/feature/modeling-rf-lineales-experimentos`.
`results/xgboost_15_validacion/referencias_companera.csv` y `comparacion.csv`
conservan referencias atribuidas, no nuevas corridas de otras familias. El tracking
histórico conserva 139 registros XGBoost con granularidad por experimento; los
trials podados quedan fuera del CSV. El diagnóstico explica las diferencias por
float32 sin alterar predicciones ni fabricar registros retrospectivos.

### LightGBM

Las dos búsquedas de 100 trials usan 15 entradas: la inicial eligió el trial 72
(MAE CV 3982.2353; validación 2321.0388) y la segunda el 99 (CV 3948.7261;
validación 2276.7354). El segundo candidato usa profundidad 3, 8 hojas,
`min_child_samples=31` y `learning_rate=0.036985470389312915`.
Las configuraciones completas y modelos de validación permanecen en
[primera búsqueda](results/lightgbm/validacion.json),
[segunda búsqueda](results/lightgbm_2/validacion.json), `models/lightgbm/` y
`models/lightgbm_2/`; no son el modelo de producción publicado.

El [ADR 0021](docs/decisions/0021-lightgbm-optuna.md),
[ADR 0022](docs/decisions/0022-lightgbm-segunda-busqueda.md),
[informe de búsqueda](results/lightgbm_2/informe.md),
[comparación](results/lightgbm_2/comparacion.csv),
[informe de diagnóstico](results/lightgbm_diagnostico_validacion/informe.md) y
[notebook](notebooks/modelado_lightgbm.ipynb) permiten revisar sin entrenar.
Se conservan los resultados y gráficos de ambas búsquedas. El notebook solo lee
el diagnóstico; no carga datasets, modelos ni estudios mutables. Las copias
`codigo_ejecutado/` documentan las corridas y sus huellas; los scripts de nuevas
búsquedas están en `src/`. El tracking mantiene 86 registros propios (84 COMPLETE
y dos validaciones), timestamps originales sin zona horaria y la limitación del
árbol modificado durante la ejecución.

### Random Forest

La búsqueda inicial probó 20 configuraciones; se conservaron estudios iniciales
y Optuna con 14 y 15 entradas. El mejor bosque de validación fue el inicial de 15:
200 árboles, profundidad 10, hoja mínima 12 y mitad de variables por división,
`n_jobs=1`. La configuración inicial de 14 obtuvo MAE 2293.7033; Optuna obtuvo
2305.9525 con 14 y 2319.0805 con 15, sin mejorar el candidato inicial.

Sus artefactos de evidencia son el [log consolidado](results/experiment_logs.csv),
el [CSV de procedencia](results/comparacion_modelos_fuentes.csv) y los notebooks
[de búsqueda inicial](https://github.com/AlesioRios/Trabajo-Practico-Final-LD2/blob/9832116044b7f9b82cdd041f7b66efdc7338603c/notebooks/modelado_rf_lineales.ipynb)
y [Optuna](https://github.com/AlesioRios/Trabajo-Practico-Final-LD2/blob/9832116044b7f9b82cdd041f7b66efdc7338603c/notebooks/modelado_rf_lineales_optuna.ipynb)
de la rama histórica. No hay CSV independientes de sus predicciones de validación
ni un bosque publicado en el registry actual: las métricas se respaldan con el log
y los notebooks ejecutados, sin atribuirles una reproducción nueva.

### Ridge

Los lineales compararon diez valores de alpha en la búsqueda inicial y luego
Optuna, tanto con 14 como con 15 entradas. Ridge usa pipeline A (mediana y
StandardScaler aprendidos dentro de cada ajuste), `alpha=30` y semilla 42.
Con 14 entradas, búsqueda inicial y Optuna dieron MAE 2225.4432; con 15,
2284.2013. Comparte los notebooks y la procedencia histórica de Random Forest;
el [ADR 0020 histórico](https://github.com/AlesioRios/Trabajo-Practico-Final-LD2/blob/9832116044b7f9b82cdd041f7b66efdc7338603c/docs/decisions/0020-random-forest-y-modelos-lineales.md)
explica su preparación y espacios de búsqueda. No se publicó un Ridge final ni
predicciones individuales de validación en esta rama. El Ridge usado como juez
en Fase 2 y sus métricas pertenecen a otro alcance, detallado en el informe comparativo.

### Lasso y contrato de entradas

Lasso comparte las cuatro búsquedas de los lineales: la configuración de 14 entradas
con `alpha=75`, `max_iter=20000` y semilla 42 alcanzó MAE 2215.7964 tanto en la
búsqueda inicial como en Optuna; con 15 entradas alcanzó 2265.6207. El cierre
publica el candidato de 14 que ganó en Fase 3, sin atribuirle las métricas del de 15.

La selección oficial vigente de `src/seleccion_definitiva.py` tiene **15 predictoras**,
incluida `es_fin_de_anio`. El modelo publicado conserva las **14 entradas históricas**
de `data/processed/seleccion_original_14/data_lineage.json` por decisión explícita
del [ADR 0024](docs/decisions/0024-cierre-lasso.md). El indicador adicional queda en
el dataset, pero no se pasa a Lasso; su orden real se recupera del artefacto y del
registry. Las búsquedas vigentes de 15 y la predicción del cierre de 14 tienen
contratos explícitos diferentes; no se altera ninguno para igualarlos.

## Cierre de Fase 3: Lasso con catorce entradas

El [informe final](results/lasso_final/informe_final.md) describe el cierre ya
realizado: mediana, StandardScaler y Lasso reajustados con 1006 filas train+validation,
versión 1.0.0 en [registry](models/model_registry.json) y
`models/modelo_v1.0.0.pkl`. La validación original se conserva separada de las
métricas de ajuste y del test local. Predicciones archivadas, coeficientes, residuos,
gráficos y controles están en `results/lasso_final/` y el notebook común los consulta.
No volver a entrenar ni evaluar test para revisar esta entrega.

Para recuperar el contrato del pipeline final:

```python
import joblib
modelo = joblib.load("models/modelo_v1.0.0.pkl")
columnas = modelo.cierre_metadata_["columnas"]
# X_nuevo debe prepararse con las mismas reglas históricas de Fase 2.
predicciones = modelo.predict(X_nuevo[columnas])
```

El `commit` del registry identifica el código previo al entrenamiento y coincide
con el incluido en el artefacto. El commit posterior de resultados recibe el tag
anotado de versión; no se incluye su propio hash dentro de sí mismo. Para versiones
nuevas se reserva el siguiente patch sin sobrescribir artefactos anteriores.
`src/informe_lasso_final.py` genera informes desde evidencia y escribe salidas;
no hace falta ejecutarlo para consultar el informe publicado.

El resultado local no acredita el umbral del test oculto. Validation intervino antes
en selección; `DIA` puede repetirse y el horizonte certificado es el bloque siguiente.
La disponibilidad física de las entradas a las 23:59 no está demostrada. Estos
límites se mantienen aunque los controles técnicos de reproducibilidad sean correctos.

## Estado por fase y trazabilidad

Estado de `main` revisado al 15/11/2026, después de integrar el cierre Lasso y la
predicción (PR #27):

- [x] Fase 0: entorno, control de versiones y trazabilidad.
- [x] Fase 1: EDA.
- [x] Fase 2: preprocesamiento y selección oficial de 15 predictoras.
- [x] Fase 3: comparación, cierre Lasso de 14 entradas, serialización y registry.
- [x] Fase 4: pipeline de predicción desde un Excel nuevo.

Las menciones a integración, ganador global, notebook común o Fase 4 pendientes en
los informes históricos describen su fecha y alcance originales, anteriores al
cierre; no son tareas pendientes del código actual. Se preservan esos informes
como evidencia. El [informe comparativo](results/comparacion_modelos_final.md)
termina en la selección provisional; el [informe final](results/lasso_final/informe_final.md)
y el ADR 0025 documentan las etapas posteriores. La
[auditoría de modelos](docs/auditoria-fase3-modelos.md) conserva la revisión previa
de las secciones 3.1–3.4 y sus límites. El test oculto y la certificación operativa
siguen fuera de lo demostrado por estas entregas.

Los PR individuales XGBoost y LightGBM describían sus ramas antes de la integración:
contenían sus propias búsquedas y cuatro baselines heredados. El log común actual
conserva la unión histórica y el cierre, con procedencia en el CSV de fuentes; no
se deben interpretar aquellas descripciones como el contenido actual de `main`.
Los diagnósticos conservan gráficos y auditorías de tracking sin fabricar corridas.
Sus scripts `src/diagnostico_xgboost.py`, `src/diagnostico_lightgbm.py` y los
actualizadores de notebooks escriben salidas; para consulta leer los informes y PNG.
Las pruebas focalizadas de esos diagnósticos están en
`tests/test_diagnostico_xgboost.py` y `tests/test_diagnostico_lightgbm.py`.

## Estrategia de ramas y acuerdos

`main` permanece estable: cada cambio se desarrolla en una rama y se integra por
Pull Request revisado. Las ramas históricas por fase son `feature/eda`,
`feature/preprocessing`, `feature/modeling` y `feature/prediction-pipeline`.
Cada integrante registra sus propios commits. Crear un PR no autoriza su merge.

- [Consigna](consigna.md).
- [Acuerdos de datos](docs/acuerdos-de-datos.md): formatos y funciones compartidas.
- [Decisiones de diseño](docs/decisions/): justificación y alcance por etapa.
- [Selección de variables, ADR 0015](docs/decisions/0015-seleccion-de-variables.md).
