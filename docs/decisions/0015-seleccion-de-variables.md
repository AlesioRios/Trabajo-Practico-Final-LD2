# ADR 0015: Spearman como conjunto definitivo

- **Fecha:** 13/11/2026
- **Estado:** aceptada para el contrato de selección; disponibilidad operativa pendiente

## Contexto

Se necesita una entrada interpretable para predecir el frío del bloque diario siguiente usando solo información hasta el cierre del actual. Train contiene 848 bloques, Validation 182 y Test 182. Se eligieron cinco candidatos inicialmente solo con Train, con separación temporal y aprendizaje por prefijo. La comparación posterior utiliza las mismas 176 observaciones etiquetadas de Validation, preparación de Train y `Ridge(alpha=100.0, solver="svd")`, semilla 42. No se emplea Test para esta selección ni para las verificaciones de este PR.

## Decisión

Consumir las 14 variables de Spearman desde [seleccion_definitiva.json](../../results/seleccion/comparacion_validation_congelada/seleccion_definitiva.json), única fuente operativa de nombres, orden, ingeniería y preparación. `seleccionar_variables(X_train, y_train)` devuelve esa lista congelada sin recalcular ranking ni búsqueda y sin consultar y_train. `SEMILLA = 42` corresponde a los experimentos y a las configuraciones explícitas; cargar el contrato no requiere aleatoriedad.

## Justificación

La tabla 2 del notebook contiene la comparación completa: Pearson tuvo el menor MAE absoluto, 2239.58 kW, con 19 variables y 37 columnas; Spearman obtuvo 2247.82 kW con 14 variables y 27 columnas. La diferencia de 8.24 kW (aproximadamente 0.37 %) está dentro del margen práctico del 1 %. Spearman conserva cinco variables menos y se elige por parsimonia. No se afirma que tenga menor error ni superioridad estadística. Con 176 observaciones y esa diferencia no se hizo un análisis que permita declarar significación o equivalencia estadística.

El registro interno anterior a la evaluación es [candidatos.json](../../results/seleccion/candidatos_congelados_validation/candidatos.json): fecha 2026-11-13T22:35:18.737995+00:00 y campo criterio_previo. El comparador verifica la huella del candidato antes de ajustar los modelos. resultados.json conserva candidatos_sha256 de la versión original; [portabilidad.json](../../results/seleccion/cierre_fase23/portabilidad.json) relaciona esa huella con la versión que solo relativiza metadatos. El notebook muestra fecha, criterio y vínculo entre huellas. Esta es evidencia interna del procedimiento, sin sello de tiempo independiente; no se inventó evidencia ni se cambió el criterio histórico para justificar el resultado. El margen expresa una decisión de ingeniería, no una prueba estadística.

Las 27 columnas efectivas son 14 variables más 13 indicadores de ausencia aprendidos con Train; solo dia_anio_coseno no necesita indicador. La preparación histórica imputa medianas, estandariza las 14 variables y agrega indicadores sin escalar. La variante A del Pipeline compartido no agrega esos indicadores: puede consumir la misma lista de 14 entradas, pero su matriz efectiva y sus métricas no deben confundirse con las de la comparación de selección. Las tablas 3 y 4 del notebook muestran el contrato y los controles de jerarquía.

La tabla 5 compara soportes idénticos: Spearman 2173.40 kW, persistencia 2668.33 kW y media móvil de siete días 2497.65 kW sobre las mismas 171 filas disponibles. La mejora frente a la media móvil, el mejor baseline, es aproximadamente 13 %. No se compara el MAE de 2247.82 sobre 176 filas con baselines sobre 171. Los índices y errores proceden de los artefactos originales.

## Alternativas descartadas

La tabla 2 conserva Forward 2377.22 kW (30 variables, 59 columnas), Lasso 2723.35 (268/532) e información mutua 2736.27 (311/619). Pearson tiene el mínimo absoluto, pero el margen práctico previo permite preferir la parsimonia de Spearman.

Forward tradicional comienza vacío, agrega por nombre la candidata que minimiza el MAE medio con Ridge alpha=100 y solver svd, tres pliegues temporales y gap=1. El máximo es 60 columnas efectivas incluyendo indicadores, no 60 variables directas. La mejor ronda es la 30; la 31 no mejora. Su MAE 2826.52 kW es interno y se utilizó para elegir variables, no una evaluación independiente. El registro progreso.json mide 449.88 segundos, aproximadamente 450. La tabla 6 muestra ronda, error, dimensiones y tiempo.

Lasso tuvo una advertencia de convergencia en un ajuste externo histórico. La comprobación final posterior registrada en candidatos.json verificó la misma selección y dual_gap=6102.317616208471 por debajo de la tolerancia efectiva 6108.6231615631195, con 8630 iteraciones. La tabla 7 muestra esa evidencia; no se repitió el ajuste en este cierre ni se invalida automáticamente todo el método por la advertencia previa.

Unión de selectores y Forward 30/15 quedaron fuera del flujo vigente por la dependencia del prefiltro y la falta de comparación homogénea. Se conservan como evidencia descartada en los paquetes de [historial](../../results/seleccion/historial/README.md), sin imports activos ni resultados reescritos.

## Limitaciones

Validation se usó para elegir entre cinco conjuntos: el MAE de Spearman puede ser optimista y no es rendimiento predictivo final independiente. Test permanece reservado para una única evaluación final posterior.

No se certifica que las seis mediciones originales estuvieran publicadas a las 23:59. Faltan zona horaria, latencia, revisiones y correspondencia bloque-fecha física de Consolidado EE, Consolidado Agua y Totalizadores Energia. Los rezagos usan bloques anteriores y el coseno usa DIA+1; el horizonte demostrado es el bloque siguiente, no un calendario físico validado. Disponibilidad matemática del dataset no demuestra disponibilidad operativa.

Ridge alpha=100 fue el evaluador común de esta selección. El Pipeline del equipo prueba una grilla de alpha; otro valor de regularización o modelo podría favorecer otro conjunto. Es una limitación de alcance, no un error ni una autorización para cambiar las variables congeladas en este PR.

El punto de entrada compartido conserva su política de etiqueta de frontera y días incompletos. La selección usa las reglas congeladas y protege la frontera del objetivo por tramo. La conexión final del equipo debe aplicar esa política antes de aprender con Train, sin modificar silenciosamente el código compartido.

## Reproducción

El notebook ejecuta solo lectura y controles de artefactos versionables. La tabla 8 identifica fuentes, scripts, comandos y tiempos registrados; Forward max60 duró aproximadamente 450 segundos. Los tiempos de etapas individuales constan en resumen_resultados.csv dentro del paquete de evidencia. No se registró duración verificable de la comparación común; no se inventa una estimación. La regeneración costosa es explícita y usa directorios nuevos; no se sobrescribe el conjunto oficial ni se ejecuta automáticamente durante la lectura del notebook. Ver [procedimiento y contrato](../informes/integracion_pr_fase23.md).
