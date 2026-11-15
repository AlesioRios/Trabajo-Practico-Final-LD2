# Consigna oficial y almacenamiento de XGBoost y LightGBM

Revisión: 15/11/2026. Se consultó el documento `referencias/Trabajo Práctico Final
2026.pdf` del espacio de trabajo, páginas 5–6 (Fase 3) y 8–9 (aprobación,
reproducibilidad y estructura). Coincide con `consigna.md`, secciones 3.1–3.4.
Los apuntes de interpretación no se utilizan como requisitos adicionales.

La edición se limita a `feature/modeling-xgboost`. LightGBM se inspeccionó en la
copia `modelado-lightgbm`, rama `feature/modeling-lightgbm`, sin modificarla ni
ejecutar sus scripts o modelos. Las rutas LightGBM de esta tabla pertenecen a esa
rama; no se copian a XGBoost ni se consideran disponibles en un clon de esta rama.
Este inventario prepara la integración, no sustituye la revisión del otro PR.

| Exigencia oficial | XGBoost y ubicación | LightGBM y ubicación | Estado / pendiente |
|---|---|---|---|
| 3.1: baselines y candidatos | `notebooks/modelado.ipynb`, celdas 5–7, y `results/experiment_logs.csv`; estudios en `results/xgboost_15/` y `results/xgboost_15_refinado/` | Baselines comunes y estudios en `results/lightgbm/` y `results/lightgbm_2/` | Cumplidos para estas dos familias. Comparación conjunta de al menos cuatro familias pendiente de integración. |
| 3.2: parámetros, MAE/RMSE/R² y timestamps | 139 registros propios: 137 COMPLETE y dos validaciones; `results/xgboost_diagnostico_validacion/auditoria_tracking.csv` concilia el log | 86 propios: 84 COMPLETE y dos validaciones; ADR 0022, sección de tracking | Parcial en ambos bajo lectura literal de cada entrenamiento: CSV por experimento, no por ajuste; podados en JSON/Optuna. No se inventan registros ni fechas. |
| 3.2: espacio y análisis de optimización | `docs/decisions/0019-xgboost-optuna.md`, notebook específico y `espacios_busqueda.csv` del diagnóstico | ADR 0021/0022, `results/lightgbm_2/informe.md`, comparaciones y límites | Cumplidos para candidatos, con parámetros completos y evidencia ejecutada. La justificación del ganador global se reserva a integración. |
| 3.3: métricas de validación | `results/xgboost_15_validacion/evaluacion.json` y `results/xgboost_15_refinado_validacion/evaluacion.json` | `results/lightgbm/validacion.json` y `results/lightgbm_2/validacion.json` | Cumplido: 830 filas de ajuste y las mismas 171 filas de evaluación. Mejor MAE XGBoost inicial: 2221,76; LightGBM segunda búsqueda: 2276,74 kW. |
| 3.3: predicción frente a real | Cinco PNG comparativos y CSV en `results/xgboost_diagnostico_validacion/`; secciones 13–16 del notebook propio | `predicciones_validacion.csv` de las dos búsquedas contiene real/predicho; no hay gráficos equivalentes guardados | XGBoost cumplido con el diagnóstico nuevo. LightGBM pendiente, resoluble con sus CSV sin entrenar. |
| 3.3: residuos y conclusiones | `informe.md`, residuos por fila/mes, errores mayores y tres gráficos específicos en el diagnóstico | El informe compara búsquedas y brechas; no analiza ni grafica residuos | XGBoost cumplido a nivel descriptivo. LightGBM pendiente. No se exige un test estadístico ni un gráfico particular por cada trial. |
| 3.3: importancia de variables del mejor modelo | No hay estimador serializado; no se inventan importancias | Hay `models/lightgbm/modelo_validacion.joblib` y `models/lightgbm_2/modelo_validacion.joblib`, más texto nativo, sin extraer importancias aquí | No aplica todavía al candidato: el requisito es del ganador global. Los PNG f-ANOVA existentes en ambos son importancia de hiperparámetros. |
| Entregable notebook común | `modelado.ipynb` conserva baselines; notebook XGBoost contiene búsquedas y nuevo diagnóstico | `modelado.ipynb` conserva baselines; no existe notebook LightGBM específico; resultados en ADR e informe | Parcial. Integrar al común y actualizar salidas históricas de 14 variables sin confundirlas con el contrato vigente de 15. El común no se edita en este trabajo. |
| 3.4: selección, train+validation, modelo semántico y registry | No hay modelo final ni registry; `src/train_model.py` es plantilla | Los modelos serializados son de validación, sin versión semántica ni registry; `train_model.py` es plantilla | Pendiente de elección global; no se puede completar sin ejecutar modelos y editar módulos compartidos, acciones no autorizadas. |
| Reproducibilidad y trazabilidad | Python 3.12.14, semilla 42, hashes/versiones y Git; nuevo diagnóstico no requiere datos originales ni modelo | Versiones y hashes en contratos; snapshots `codigo_ejecutado/` conservan código de corridas con árbol sucio | Parcial como entrega integral desde clon limpio. No confundir pruebas históricas con pruebas ejecutadas en esta revisión. |
| Git y PR | Rama propia; publicación histórica de PR #19 hacia `feature/modeling` | Rama propia; publicación histórica de PR #20 hacia `feature/modeling` | No se fusiona ni publica automáticamente. El conector devolvió 404 para PR #19 al consultar; no permite afirmar su estado remoto actual. |

## Alcance preciso de la consigna

La sección 3.3 exige análisis de residuos y gráficos de predicción frente a real
sin restringirlos explícitamente al ganador. Para dejar evaluados estos dos
candidatos, corresponde documentar ambos. No exige repetir el diagnóstico para
cada trial. Solo la importancia de variables está expresamente restringida al
mejor modelo. Los histogramas, residuos frente a predicción y evolución temporal
son la forma elegida de cubrir el análisis, no una lista literal de gráficos del PDF.
SHAP, QQ-plots, pruebas de normalidad o nuevos ajustes no son condiciones añadidas.

Los CSV originales, resúmenes, estudios comprimidos, hiperparámetros, predicciones
y logs están almacenados en carpetas propias y versionados en sus ramas. Los
resultados nuevos de diagnóstico se separan de ellos. No se altera el log histórico
para simular tiempos por pliegue ni se cambian valores por discrepancias mínimas de
precisión decimal. Los timestamps históricos no tienen zona horaria; no se les
atribuye retrospectivamente una zona no registrada.

## Pendientes que no deben declararse cumplidos

La Fase 3 global aún requiere comparar todas las familias, elegir por MAE de
validación, documentar importancia de variables del elegido, implementar el punto
de entrada común, reentrenar solo el ganador con train+validation y registrar su
versión semántica, parámetros, métricas de validación, ruta y commit en
`models/model_registry.json`, con etiqueta Git equivalente. Los modelos LightGBM
de validación no sustituyen ese registro final.

Las condiciones eliminatorias de las páginas 8–9 incluyen baselines y modelo
final sobre el mismo test, mejora de ambos baselines, MAE menor a 3000 en el test
oculto y reproducción integral con `predict.py`. Ninguna puede acreditarse solo
con métricas de validación. No se accedió a prueba para esta revisión.

La corrección inmediata de LightGBM es documental y gráfica a partir de sus CSV;
queda identificada sin alterar esa rama. La evidencia de este documento corresponde
al inventario local de la fecha indicada, no a una certificación de merges remotos.
