# Diagnóstico archivado de XGBoost — Fase 3

Fecha del análisis: 15/11/2026. Fuentes: predicciones y evaluaciones de
`results/xgboost_15_validacion/` y `results/xgboost_15_refinado_validacion/`,
estudios exportados de `results/xgboost*/` y `results/experiment_logs.csv`.
No se abrió el dataset ni el conjunto de prueba, no se cargó un estimador y no
se ejecutaron entrenamiento, predicción, optimización ni una nueva evaluación.
Los SHA-256 de las fuentes están en `verificacion.json` y permanecen intactos.

## 1. Convención y comparación

Se conservan las mismas 171 filas e índices CSV de validación, con fecha rotulada
de emisión entre 29/10/2022 y 26/04/2023. DIA identifica el bloque de emisión;
el objetivo corresponde al bloque siguiente, no necesariamente a DIA + 1 físico
debido al desfase del calendario documentado. Las fechas pueden repetirse;
la identidad y el orden se comprueban por indice_csv. No se interpolan las filas
ausentes. La línea temporal une solo observaciones disponibles y no implica
que se haya evaluado cada día intermedio.

El residuo es real menos predicho: positivo significa subestimación y negativo,
sobreestimación. Se presentan tanto el experimento inicial (trial 56) como el
refinado (trial 49), sin elegir al ganador global del trabajo.

El MAE registrado inicial es 2221.76 kW y el refinado,
2229.22 kW. El refinamiento reduce el MAE de validación
cruzada en 16.74 kW, pero aumenta el MAE de
validación en 7.46 kW.
La mejora del inicial frente a la media móvil de 2497.65 kW es
11.05 % en estas filas de validación;
no es un resultado de prueba ni satisface por sí sola la condición de aprobación.
Las tablas completas están en `comparacion_validacion.csv`, `hiperparametros.csv`
y `espacios_busqueda.csv`; incluyen parámetros fijos y optimizados de ambos ganadores.

## 2. Lectura de las predicciones y los residuos

El gráfico `prediccion_vs_real.png` muestra asociación con el consumo observado,
pero también dispersión alrededor de la diagonal. Las predicciones suavizan
algunos cambios bruscos, como se aprecia en `predicciones_temporales.png`.
No hay un ajuste nuevo de una recta ni calibración de las predicciones.

La distribución (`distribucion_residuos.png`) está cerca de cero en promedio:
sesgo inicial -79.93 kW y refinado
-53.41 kW. Eso no implica errores pequeños:
el desvío inicial es 2851.53 kW y el percentil 95 del error
absoluto es 5595.68 kW. El inicial subestima en
76 filas y sobreestima en 95;
el refinado lo hace en 79 y 92.
No se afirma normalidad ni independencia de los residuos.

En `residuos_vs_predicciones.png` los errores no forman una nube completamente
homogénea: visualmente hay predominio de errores negativos en la zona de
predicciones altas. Es un diagnóstico descriptivo, sin una prueba formal de
heterocedasticidad ni una inferencia causal.

El sesgo global oculta diferencias temporales (`residuos_temporales.png`):
en enero de 2023 el inicial subestima 1599.66 kW en promedio sobre
29 filas, mientras que en abril sobreestima 1673.62 kW
sobre 22. Octubre tiene solo dos filas y no permite generalizar
una comparación mensual. `residuos_por_mes.csv` conserva tamaños, MAE y sesgos.

El mayor error inicial es una sobreestimación de 7672.62 kW
en el índice CSV 953, fecha de emisión 2023-03-02:
real 10274.00 y predicho 17946.62 kW. Los diez mayores errores
concentran 16.89 % del error absoluto total.
Se conservan las diez filas de cada experimento en `errores_mayores.csv`, sin
eliminarlas ni atribuirles fallas de planta que las predicciones no pueden probar.

## 3. Optimización, tracking y precisión

La búsqueda inicial vigente conserva 100 trials (47 completos, 53 podados);
el refinado, 60 (45 completos, 15 podados). Ambos usan TPE con semilla 42,
MedianPruner y cinco pliegues temporales con gap=1 sobre entrenamiento.
El refinado se definió con los diez mejores trials anteriores: amplió árboles
a 1750–3000, mantuvo profundidad 3 y tasa 0.0020702351473759497, y acotó peso
de hijos y fracciones de filas/columnas. Se minimizó el MAE conjunto de las
filas comunes, no el promedio no ponderado por pliegue. Ver ADR 0019.

La auditoría contrasta 137 COMPLETE (incluidos 45 históricos de 14 variables)
y dos validaciones con 139 filas propias del log. Compara parámetros completos,
métricas, filas, semilla, commit y timestamps ISO originales; no faltan ni se
duplican esos registros. `auditoria_tracking.csv` es evidencia derivada, no un
reemplazo ni una ampliación retrospectiva de `experiment_logs.csv`.

La sección 3.2 queda parcialmente cubierta bajo su lectura literal: el log guarda
un experimento por COMPLETE, agrupando cinco ajustes; los podados quedan en
Optuna/JSON, sin fila en el CSV. Los timestamps son de registro, no de inicio del
ajuste, y no llevan zona horaria. No hay evidencia para inventar tiempos por
pliegue ni métricas finales de podados. Resolver el contrato futuro corresponde
a la integración, sin modificar las ejecuciones históricas.

Los CSV contienen representaciones decimales de predicciones float32. Las
métricas calculadas sobre esos decimales difieren mínimamente del log; recuperar
float32 reproduce las tres métricas históricas con tolerancia estricta. El cambio
máximo de predicción es 0.00093750 kW.
Los residuos usan los decimales originales; el resumen separa las métricas del CSV
de las columnas terminadas en `_registrado`. No se corrige ningún resultado previo.

## 4. Reproducción y entrega

Con Python 3.12 y las dependencias congeladas, desde la raíz del repositorio:

```powershell
python -B src/diagnostico_xgboost.py
python -B -m pytest tests/test_diagnostico_xgboost.py -q -p no:cacheprovider
```

El primer comando solo deriva CSV, PNG y verificaciones dentro de esta carpeta.
No importa XGBoost, LightGBM, scikit-learn, Optuna ni joblib. El segundo archivo
de pruebas se limita a aritmética, validaciones de evidencia y controles estáticos.
No se autoriza la suite completa, que contiene ajustes de estimadores.
La sección 13 en adelante de `notebooks/modelado_xgboost.ipynb` es autónoma y
consulta este diagnóstico; las primeras secciones son históricas y no deben
ejecutarse para esta revisión. Las dependencias de estado de las celdas históricas
no se certifican como reproducción integral del notebook desde un kernel vacío.

## 5. Alcance y pendientes

Quedan completos para estos candidatos XGBoost los gráficos de predicción frente
a real y el análisis descriptivo de residuos de 3.3, con salidas ejecutadas.
Los cinco PNG y las tablas quedan preparados para integrar al notebook común,
que no se modifica. Se preservan las 15 variables y todos los resultados previos.

No hay un modelo XGBoost serializado del que extraer importancia de variables.
La f-ANOVA existente trata hiperparámetros y no cubre ese requisito. La consigna
pide importancia de variables del mejor modelo global; queda pendiente de su
elección y de un artefacto real. No se reconstruye un estimador para obtenerla.
También quedan para la integración: comparación de todas las familias, selección
por MAE de validación, entrenamiento final con train+validation, versión semántica,
registry, etiqueta Git y punto de entrada común. Todos los pasos que requieren
ejecutar modelos están fuera del alcance autorizado.

La selección previa de variables no está anidada y validación ya fue consultada
en decisiones anteriores. Estos gráficos añaden inspección del mismo período;
no son una nueva estimación independiente ni justifican ajustar hiperparámetros.
Los cambios de sesgo por mes no demuestran causa operativa ni generalización.
