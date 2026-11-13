# Resultados definitivos de Fase 2.3

Fuente: [resumen congelado](../../results/seleccion/comparacion_validation_congelada/resumen.csv). Mismo Ridge y las mismas 176 filas de Validation. No se recalcularon modelos para este documento.

| Método | Variables | Columnas efectivas | MAE (kW) |
|---|---:|---:|---:|
| pearson | 19 | 37 | 2239.58 |
| spearman | 14 | 27 | 2247.82 |
| mutua | 311 | 619 | 2736.27 |
| lasso | 268 | 532 | 2723.35 |
| forward | 30 | 59 | 2377.22 |

Pearson minimiza MAE. Spearman se elige por parsimonia dentro del margen previo del 1 %. Lista y preparación en [el artefacto](../../results/seleccion/comparacion_validation_congelada/seleccion_definitiva.json); cifras completas y predicciones en resultados.json.

Los baselines históricos utilizaron 171 filas comunes, no las 176: persistencia 2668.3274853801167 y media móvil 2497.6535783904205 kW. Compararlos solo con mae_en_filas_baseline: Spearman 2173.39634430503 kW. No mezclar soportes ni presentar esta comparación como rendimiento final.

[Interpretación y auditoría](seleccion_fase23_estado_actual.md). Validation se usó para elegir; Test reservado.
