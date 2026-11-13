> Estado de integración actualizado: [revisión previa al PR](integracion_pr_fase23.md). Se recibió origin/feature/preprocessing en 2cfd478 sin crear commit; se conserva el punto de entrada compartido. Los estados de Git y prerrequisitos descritos en revisiones anteriores son históricos.

# Continuación tras Fase 2.3

Spearman definitivo: 14 variables, 27 columnas efectivas, MAE de Validation 2247.817074339489 kW. Pearson tuvo el mínimo absoluto; el margen previo del 1 % permite preferir Spearman. Los cinco métodos se compararon sobre 176 filas. No repetir búsquedas, selección ni comparación. Test reservado.

La integración está disponible en src/seleccion_definitiva.py y en el valor predeterminado de seleccionar_variables. El notebook solo lee resultados. [Guía](../guia-fase2.md), [informe](seleccion_fase23_estado_actual.md), [historial](metodologia_fase23.md), [resultados](resultados_fase23.md).

Persisten la confirmación externa de calendario físico y publicación de mediciones, y la alineación de etiquetas y bloques incompletos del punto de entrada compartido para Fase 2.4. La advertencia Lasso de un pliegue externo es histórica; el ajuste final registrado satisface tolerancia. No se certifica publicación exacta a las 23:59.

Las pruebas, cambios preexistentes y propuesta exacta de commit se registran en cierre_fase23.md y results/seleccion/cierre_fase23/. No se hizo commit, push ni staging.
