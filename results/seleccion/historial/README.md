# Evidencia de selección

El único paquete necesario para el notebook es `ejecucion-20261113T184058Z-969d4545-evidencia.zip`. Contiene el resumen de etapas que lee el notebook y conserva las configuraciones, reglas por pliegue, perfiles, estabilidad, resúmenes de candidatos, código y registros de las configuraciones elegidas. No se cambiaron los bytes de esas entradas. Se retiraron del paquete logs operativos e intentos fallidos con rutas personales, que no son insumos de reproducción.

`manifiesto_evidencia.json` registra tamaño, cantidad de entradas y SHA-256 del paquete publicado, junto con la huella anterior y las entradas omitidas. Antes de extraerlo, verificar su huella; usar un directorio vacío para no sobrescribir evidencia local. Las rutas internas comienzan por la carpeta de ejecución. Los originales de los datos se obtienen por el mecanismo del proyecto, no desde archivos privados de esta copia.

Los paquetes de unión de selectores, Forward 30/15 y diagnósticos anteriores se retiraron del árbol versionado porque no intervienen en la selección vigente. Se conserva su descripción y SHA-256 en el manifiesto. No son dependencias operativas ni se prometen como parte de un clon nuevo. Sus copias locales permanecen ignoradas; la historia de Git conserva además las versiones publicadas anteriormente.

Los resultados y checkpoints del Forward tradicional definitivo permanecen en `../forward_max60_artefactos/`. El código exacto del comparador histórico se conserva en `codigo_comparador_validation_20261113.py`. Estos insumos son distintos de los procedimientos descartados. La explicación metodológica vigente está en el ADR de selección y en el notebook, sin duplicar aquí sus cifras.

Las tablas compartidas, matrices intermedias, bytecode, copias locales y detalles repetitivos de ajustes descartados no se versionan. Su ausencia no impide cargar el contrato oficial ni ejecutar el notebook. Los comandos explícitos de reproducción están en la [guía de conexión](../../../docs/informes/integracion_pr_fase23.md); no se ejecutan automáticamente.
