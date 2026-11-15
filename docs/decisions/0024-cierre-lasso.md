# ADR 0024: cierre de Lasso con catorce variables

- **Fecha:** 15/11/2026
- **Estado:** aceptada para la ejecución final autorizada

## Contexto

La comparación de Fase 3 seleccionó Lasso con catorce entradas y pipeline A por MAE de validación 2215.796375483132. El usuario fijó algoritmo, variables, orden y parámetros y autorizó una sola evaluación de test local después del reentrenamiento. La tabla de validación del notebook y results/comparacion_modelos_final.md respaldan esa selección.

## Decisión

Reajustar mediana, StandardScaler y Lasso(alpha=75, max_iter=20000, random_state=42) con todas las filas etiquetadas de entrenamiento más validación del CSV procesado. Recuperar las catorce entradas desde el linaje histórico original, sin cambiar el CSV vigente de quince columnas. Reutilizar armar_pipeline("A", columnas). Verificar dos ajustes deterministas y la carga joblib antes de abrir test. Conservar las métricas originales de validación como selección histórica.

Separar las órdenes entrenar, evaluar-test-una-vez y consultar. Crear un marcador exclusivo antes de abrir observaciones test. Archivar predicciones y métricas; las consultas y el notebook reutilizan esos archivos. Ante interrupción no borrar automáticamente el marcador. Evaluar modelo y ambos baselines sobre el mismo soporte común.

El commit previo al ajuste identifica el código entrenado en el artefacto, el manifiesto y el registry. Un commit posterior contiene resultados y recibe el tag anotado de versión. No se intenta incluir el hash del commit de resultados dentro de sí mismo. No se hace push ni merge a main.

## Justificación

Las tablas de comparación del notebook conservan el MAE original y muestran separadamente entrenamiento, test y baselines. La verificación de dos ajustes se realiza antes del acceso test. Las predicciones archivadas permiten auditar el resultado y generar residuos sin volver a predecir. La versión semántica y SHA-256 vinculan el pipeline serializado con la ejecución. Los ensayos históricos y el registro anterior se conservan.

## Alternativas descartadas

Cambiar a quince entradas alteraría el candidato elegido. Elegir otro modelo después de observar test contaminaría la evaluación. Usar el notebook como punto de entrada de entrenamiento o test permitiría ejecuciones accidentales. Borrar etiquetas de frontera eliminadas por Fase 2 cambiaría la preparación original.

## Limitaciones

Validation ya intervino en la selección de variables. DIA puede repetirse y el horizonte comprobado es el bloque siguiente: no se certifica correspondencia calendario física ni disponibilidad operativa a las 23:59. La evaluación local no demuestra el umbral del test oculto. El código de Fase 4 permanece fuera de alcance. Las métricas y limitaciones efectivas del cierre se publican en results/lasso_final/informe_final.md y se muestran en el notebook sin cifras manuales.
