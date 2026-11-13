# Evidencia histórica de Fase 2.3

Los cuatro paquetes *-evidencia.zip son los archivos destinados a Git. manifiesto_evidencia.json guarda huellas, cantidades y tamaños. Se verificó la integridad de los tres paquetes de ejecuciones; todos sus contenidos conservan los bytes originales.

El paquete de métodos individuales incluye código, configuraciones, datos de particiones, reglas por pliegue, perfiles, estabilidad, resúmenes CSV completos de la búsqueda, eventos y registros JSON de las configuraciones finales elegidas. Estos resúmenes conservan los resultados de los candidatos descartados. Los paquetes de unión y Forward 30/15 conservan código, preparaciones, manifiestos, eventos y checkpoints de procedimientos abandonados. auditoria-y-diagnosticos-evidencia.zip guarda logs, decisiones previas, incidencias y diagnósticos anteriores.

Para inspeccionar sin ejecutar búsquedas, verificar SHA-256 contra manifiesto_evidencia.json y extraer el paquete elegido en una carpeta vacía. Los nombres internos comienzan por la carpeta de ejecución. Para reconstruir las rutas históricas de una copia nueva, extraer desde results/seleccion/. No sobrescribir evidencia local modificada; usar una carpeta aparte para comparar.

```python
from zipfile import ZipFile
from pathlib import Path
base = Path("results/seleccion")
# nombre debe pertenecer a manifiesto_evidencia.json; verificar huella antes.
# with ZipFile(base / "historial" / nombre) as archivo:
#     archivo.extractall(base)
```

Se omiten de Git tablas_compartidas/ (cachés de matrices derivadas), bytecode, temporales y el detalle JSON repetitivo de configuraciones no elegidas. Las configuraciones y resúmenes completos, preparaciones y código permiten regenerar ese detalle con los motores archivados, exclusivamente con Train y los Excel del aula virtual validados con data/checksums.json. Puede ser costoso y no forma parte del consumo oficial. No ejecutar automáticamente ni usar Test.

Además de los archivos originales que siguen intactos, esta copia local conserva tres ZIP íntegros de ejecuciones, con los mismos nombres de carpeta y extensión .zip, ignorados por Git. sha256.json registra sus huellas. Esos paquetes incluyen todo el detalle JSON, salvo tablas compartidas regenerables y bytecode. Ruta de recuperación local: results/seleccion/historial/<carpeta-de-ejecución>.zip. No se presupone que estén en un clon nuevo: para ese clon, los paquetes compactos versionados y el código/configuraciones archivados son la ruta reproducible de recuperación. No se borró evidencia local.

Las rutas absolutas de los documentos históricos son procedencia. Resolver el sufijo desde la raíz del proyecto; el consumidor oficial no abre esos caminos. La única fuente operativa es comparacion_validation_congelada/seleccion_definitiva.json.
