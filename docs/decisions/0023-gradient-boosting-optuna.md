# ADR 0023: Gradient Boosting con validación temporal y pipeline A

- **Fecha:** 15/11/2026
- **Estado:** aceptada

## Contexto

La Fase 3 requiere optimización, tracking, comparación con ambos baselines y diagnóstico de residuos. Se incorpora GradientBoostingRegressor como candidato adicional, con el dataset final y las quince variables existentes. La rama parte de `feature/modeling` en e596b79, con los cierres de XGBoost y LightGBM integrados. No se mezclan ramas de otros modelos.

## Decisión

Reutilizar `modelado_comun.py` y crear un pipeline nuevo por fold con `armar_pipeline("A", columnas=columnas).set_params(regressor=GradientBoostingRegressor(...))`. La imputación por mediana y StandardScaler permanecen dentro del pipeline; no se transforma el objetivo. No se modifica código de Fase 2.

Ejecutar exactamente cien trials totales secuenciales, con TPESampler(seed=42), MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1, n_min_trials=3), TimeSeriesSplit(n_splits=5, gap=1) y MAE conjunto de predicciones fuera de muestra. Esta configuración se verificó en `src/modelado_xgboost.py` y `src/modelado_lightgbm.py`; no se infiere solo de documentación. El espacio solicitado queda en `docs/gradient-boosting.md` y la tabla 4 del notebook propio. Alpha es condicional a huber. Se fijan random_state=42 y n_iter_no_change=None.

Persistir un estudio SQLite exclusivo y exportar resultados después de cada trial. Guardar en el esquema común una fila por ajuste de fold y otra por estado final del trial, con identificadores en notas. Las métricas agregadas se reservan a COMPLETE; un podado conserva las métricas de sus folds terminados, nunca se presenta como experimento completo. Conservar timestamps reales, tiempos por fold, versiones, semilla, hashes, commit y estado de Git. La recuperación de RUNNING requiere la opción explícita documentada y conserva su número como FAIL, sin repetirlo.

Después del estudio, elegir por MAE CV, ajustar solo con entrenamiento y evaluar sobre la misma validación independiente que los baselines y los resultados archivados de ambos modelos. Guardar el candidato sin versión final ni entrada en el registry global. Producir desde el principio los cinco diagnósticos de residuos, errores mayores, historial e importancia de variables del estimador. El notebook propio lee evidencia y se ejecuta desde cero, sin depender de sesiones ni relanzar entrenamiento.

## Justificación

Las tablas 1, 5, 6 y 7 del notebook documentan estados, configuración elegida, métricas por fold y comparación independiente. El informe generado `results/gradient_boosting/informe.md` aporta las cifras de sesgo, extremos, patrones mensuales y brecha de entrenamiento. La elección de parámetros es empírica: minimiza el MAE CV entre las configuraciones terminadas; no se afirma que cada parámetro aislado cause una mejora.

## Alternativas descartadas

Se descarta entrenar un modelo preliminar porque la solicitud comienza directamente con cien trials. Imputar globalmente o usar validación para seleccionar el trial introduciría información ajena al ajuste permitido. Se descarta omitir StandardScaler pese a que los árboles no lo necesitan, porque la variante A es el contrato solicitado. Se descarta modificar los estudios anteriores o los puntos de entrada compartidos.

## Limitaciones

Gradient Boosting es un modelo adicional y no sustituye Random Forest ni Ridge/Lasso entre los cuatro modelos obligatorios. La selección previa de variables no es anidada; las fechas pueden repetirse y estar desfasadas, por lo que se comparan índices de fila. La disponibilidad operacional de ciertas mediciones sigue pendiente de certificar según ADR 0017. La validación pertenece a otra época y no prueba ausencia de sobreajuste. La importancia por impureza puede repartirse entre variables correlacionadas; no representa causalidad. Selección global, train+validation, test, versión semántica y registry se reservan al cierre conjunto de 3.4.

## Resultados ejecutados

La tabla 1 confirma cien trials: 38 COMPLETE, 62 PRUNED y cero FAIL. La tabla 5 identifica el trial 76 y sus parámetros:

```json
{
  "random_state": 42,
  "n_iter_no_change": null,
  "n_estimators": 550,
  "learning_rate": 0.01592324888670594,
  "max_depth": 2,
  "min_samples_leaf": 4,
  "subsample": 0.8,
  "max_features": 0.8,
  "loss": "absolute_error"
}
```

El MAE CV es 3759.637254 kW. La tabla 7 reporta MAE de validación 2316.309953, RMSE 2948.038902 y R² 0.424633 en 171 filas comunes. El mejor baseline alcanza 2497.653578 kW: la mejora es 7.26 %. El candidato queda detrás de XGBoost inicial (2221.760646) y de LightGBM segunda búsqueda (2276.735434). No se lo declara ganador global.

La configuración ganadora combina profundidad 2 (límite inferior), 550 árboles y learning_rate cercano a 0.016; limita cada árbol y acumula correcciones pequeñas. min_samples_leaf=4 está cerca del mínimo permitido; subsample y max_features de 0.8 introducen aleatoriedad reproducible. La pérdida absoluta es coherente con la métrica optimizada y reduce la influencia de errores extremos sobre el ajuste. Alpha no interviene porque la pérdida no es huber. Se elige esta combinación por su menor MAE CV, sin atribuir causalmente la mejora a parámetros aislados ni ampliar el espacio después de mirar validación.

El sesgo global es +13.61 kW, pero enero subestima +2080.86 kW en 29 filas y abril sobreestima −2056.18 kW en 22 filas. El mayor error absoluto es 8748.57 kW. La brecha validación menos entrenamiento es −86.00 kW; por períodos y distribuciones diferentes no demuestra ausencia de sobreajuste. La importancia del estimador destaca dia_anio_coseno, frio_media_7_dias y Frio (Kw); no constituye interpretación causal.

El código de entrenamiento corresponde al commit real 2e16156fb04078f83518df9d16b7b0507ae2529b, con árbol limpio al iniciar. Se verificó la reapertura y reconstrucción de las cien propuestas sin agregar trials; contrato y versiones coinciden. Se agregaron 483 registros propios (382 folds completos, 100 estados de trial y una validación), conservando el prefijo exacto del log anterior.
