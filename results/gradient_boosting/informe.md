# Gradient Boosting: estudio y validación del candidato

El estudio gradient_boosting-temporal-v1 finalizó con 100 trials. Ganó el trial 76, por MAE conjunto fuera de muestra 3759.637254 kW. Sus parámetros se guardan en validacion.json; la validación independiente no intervino en su selección.

La validación contiene 171 observaciones idénticas por índice, fecha y objetivo a las cuatro referencias XGBoost y LightGBM. El MAE es 2316.309953, RMSE 2948.038902 y R² 0.424633. La mejora relativa frente al mejor baseline es 7.26 %. Comparación completa en comparacion_validacion.csv.

El residuo se define como real menos predicho. El sesgo medio es 13.61 kW; valores positivos indican subestimación. El percentil 95 del error absoluto es 6070.09 y el máximo 8748.57 kW. Los diez mayores errores concentran 17.47 % del error absoluto. No se eliminan observaciones extremas.

El mes con mayor MAE es 2022-10: 5321.34 kW en 2 filas, sesgo -5321.34 kW. residuos_por_mes.csv y los gráficos temporales muestran heterogeneidad: un promedio global no representa todos los meses. Las fechas rotuladas pueden repetirse y estar desfasadas; el índice CSV es la identidad, conforme al ADR 0002. No se infiere causalidad ni independencia de los residuos.

Entre los meses con al menos veinte observaciones, el mayor sesgo positivo aparece en 2023-01: 2080.86 kW en 29 filas; el menor aparece en 2023-04: -2056.18 kW en 22 filas. Estos cambios de signo muestran por qué el sesgo global cercano a cero puede ocultar patrones temporales. Octubre contiene solo dos filas y no permite una conclusión mensual estable.

El MAE dentro de entrenamiento es 2402.31 kW en 813 filas comunes. La brecha validación menos entrenamiento es -86.00 kW. Es un diagnóstico de sobreajuste limitado por el cambio de período y tamaño de muestra. El MAE CV usa períodos anteriores y ajustes menores; no es directamente comparable con el error de validación. No se asegura ausencia de sobreajuste ni significación estadística de diferencias pequeñas.

La importancia de las 15 variables proviene del estimador ajustado (reducción de impureza), no de Optuna. Es descriptiva, puede repartirse entre variables correlacionadas y no demuestra causalidad. Se conserva mediana y StandardScaler mediante la variante A; se aprende de nuevo dentro de cada fold. No se transforma el objetivo.

El tracking común conserva cada fold ajustado y cada trial con su estado. Los trials PRUNED/FAIL tienen métricas finales vacías; los folds completos conservan métricas parciales identificadas. detalle.json conserva inicios, finales, duraciones y métricas por fold; SQLite y progreso.jsonl permiten auditar la secuencia. Los timestamps de fold usan UTC; los de Optuna y del registrador común conservan el reloj local real. No se alteran fechas históricas.

La sección 3.1 incorpora un candidato adicional: Gradient Boosting no reemplaza los cuatro modelos obligatorios, y su cumplimiento conjunto exige también Random Forest y Ridge/Lasso. Se completa 3.2 para este estudio y 3.3 para este candidato. En 3.4 solo se guarda el candidato entrenado con train; selección global, entrenamiento con train+validation, evaluación final de test, versión semántica y registry quedan pendientes del cierre conjunto. No se modifican train_model.py ni el notebook compartido.

Persisten las limitaciones heredadas: selección previa de variables no anidada, consultas anteriores de validación y disponibilidad operacional de ciertas mediciones pendiente de certificar (ADR 0017). La evaluación es histórica, no una garantía del test oculto.
