# Metodología e historial de Fase 2.3

La metodología vigente se detalla en el [informe](seleccion_fase23_estado_actual.md). La entrada futura se carga con seleccion_definitiva; los motores experimentales requieren invocación explícita y no forman parte del notebook final.

Se evaluaron Spearman, Pearson, información mutua, Lasso y Forward. Los cuatro primeros están en results/seleccion/ejecucion-20261113T184058Z-969d4545/: configuraciones.json, datos.json, selecciones_por_metodo.json, resúmenes, estabilidad, preparacion/, experimentos/ y codigo_revision/. Las reglas por pliegue, índices, configuraciones y código permiten auditar lo aprendido por cada ajuste. Los MAE internos no son una comparación independiente entre métodos.

La unión de selectores y Forward 30/15 quedaron descartados: restringían la búsqueda a un prefiltro, mezclaban procedimientos y no proporcionaban una comparación homogénea de cinco conjuntos independientes. Sus ejecuciones siguen en ejecucion-20261113T194441Z-union-a9b63e4f y ejecucion-forward30-20261113-01, con código y registros en auditoria/. No importar ni ejecutar sus borradores desde el flujo principal; no representan la elección final.

Las correcciones incluyeron objetivo por bloque siguiente, excluir la etiqueta de frontera de Train, gap=1 entre pliegues, invalidación de bloques con menos de 24 lecturas, limpieza e ingeniería aprendidas por prefijo, distinción de jerarquía e indicadores y Forward tradicional independiente por nombre. Se revisó Forward con límites intermedios 30/40 y finalmente 60 columnas efectivas. La evidencia final está en forward_max60_artefactos/: configuración, resultado, rondas, checkpoints de rondas 30 y 31, preparaciones, código ejecutado y huellas. Las rondas previas y logs siguen en auditoria/.

candidatos_congelados_validation/ conserva los cinco conjuntos, sus fuentes y el criterio previo del 1 % con sha256.txt. comparacion_validation_congelada/ conserva resultados.json, resumen.csv y seleccion_definitiva.json con sha256.json. El comparador histórico es src/comparar_selecciones_validation.py: no ejecutarlo en el cierre ni sobrescribir resultados. Su nota de ausencia de selección con Validation se refiere a la generación inicial de candidatos; la elección posterior entre ellos sí usó Validation. El artefacto definitivo es la fuente de verdad operativa; los candidatos históricos son evidencia congelada.

Lasso final registró dual_gap=6102.317616208471 frente a tolerancia efectiva 6108.6231615631195, 8630 iteraciones y selección reproducida según candidatos.json. Eso no elimina la advertencia histórica de no convergencia de un pliegue externo. No se repite Lasso en este cierre.

Los originales se recuperan del aula virtual indicada en el README y se validan con data/checksums.json. No versionar Excel, cachés, matrices joblib ni tablas compartidas derivadas. La decisión necesita los resultados congelados y reglas finales; la auditoría completa requiere configuraciones, registros experimentales JSON, preparaciones, perfiles, código y logs. Las tablas compartidas y cachés omitidas se regeneran con los motores y el código archivado, solo con Train. Puede ser costoso y no es parte del consumo oficial. No borrar copias locales ni ejecutar automáticamente ese procedimiento.

La portabilidad del artefacto definitivo se corrigió pasando referencias a rutas relativas; cierre_fase23/portabilidad.json conserva huella y metadatos anteriores. Las rutas absolutas históricas son procedencia, nunca rutas que el consumidor deba abrir; resolver sus sufijos desde la raíz del proyecto. No se alteraron métricas experimentales.

## Historial preparado para Git

Las carpetas completas permanecen locales e ignoradas. La evidencia versionable se distribuye en [paquetes compactos con manifiesto y recuperación](../../results/seleccion/historial/README.md). Incluyen resúmenes completos, código, reglas y registros de las configuraciones elegidas; el detalle repetitivo de otros ajustes y las matrices compartidas se recuperan localmente o se regeneran con las configuraciones y código conservados. No se eliminaron archivos originales ni se alteraron valores históricos.
