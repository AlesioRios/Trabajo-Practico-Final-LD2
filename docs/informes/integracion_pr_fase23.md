# Integración de selección antes del Pull Request

## Rama compartida y preservación

Se obtuvo origin/feature/preprocessing en 2cfd478 (PR #10, precedido por #9 y #8) mediante fetch. Se aplicó git merge --no-commit --no-ff origin/feature/preprocessing. No se creó un commit, no se hizo push ni se abrió el Pull Request: MERGE_HEAD identifica el commit integrado y el merge espera la confirmación de los archivos. El destino sigue siendo feature/preprocessing, nunca main. No se reescribió el historial.

Se detectó un merge anterior pendiente hacia 902d79a. Sus metadatos, índice, parches preparados/sin preparar y archivos se respaldaron fuera del repositorio antes de usar git merge --quit, que conserva archivos e índice. Se guardaron además dos stashes, que no se eliminaron. El primer stash encontró directorios sincronizados que no pudo retirar; la recuperación final se hizo desde el respaldo de bytes verificado, sin sobrescribir contribuciones nuevas de la rama compartida.

src/preprocessing_pipeline.py y notebooks/preprocesamiento.ipynb coinciden con origin/feature/preprocessing. Las ampliaciones locales previas del cargador compartido se apartaron del flujo de selección y siguen en respaldo/stashes. Los scripts propios pasan explícitamente data/ a la firma pública del cargador. Las dos pruebas de caché que exigían aquellas ampliaciones ajenas se retiraron de la suite de selección; sus versiones previas permanecen en el respaldo. No se quitó una prueba de selección ni de fuga temporal por ese motivo.

La guía conserva el contenido de origin y cambia únicamente la fila de selección. El cambio previamente preparado que actualizaba 87 a 88 pruebas ya existe en origin. El Pipeline, su notebook, ADR 0016 y pruebas se conservan como contribuciones recibidas.

## Contrato con el Pipeline

```python
from seleccion_definitiva import cargar_seleccion_definitiva
from seleccion_de_variables import seleccionar_variables
from pipeline_sklearn import separar_xy, armar_pipeline
reglas = cargar_seleccion_definitiva()
# tabla_con_variables debe usar las reglas congeladas de reglas["ingenieria"].
# columnas = seleccionar_variables(tabla_con_variables, None)
# X, y = separar_xy(tabla_con_variables, columnas)
# pipeline = armar_pipeline(columnas=columnas)  # sin fit
```

El selector de dos argumentos carga Spearman sin usar etiquetas ni recalcular selección. La semilla de los experimentos es SEMILLA=42. El contrato tiene 14 entradas ordenadas y todas las interacciones conservan ambos padres. La prueba de integración aplica transform_variables de origin a veinte filas sintéticas con el esquema original congelado; obtiene 337 columnas y conserva las 14 elegidas, orden e índices al separar matrices de tramos sintéticos. No ejecuta Train/Validation/Test reales ni ajusta un modelo en esta comprobación.

La comparación histórica produjo 27 columnas: 14 entradas más 13 indicadores. La variante A del Pipeline compartido acepta las mismas 14 entradas y no agrega esos indicadores. No es una reproducción de la matriz histórica ni garantiza el mismo MAE; su ajuste y la grilla de alpha pertenecen a la siguiente etapa. La selección se hizo con Ridge(alpha=100.0, solver="svd"); otros modelos o alpha podrían preferir otras variables.

El punto de entrada compartido no conecta todavía selección y guardado en su main. Su responsable deberá llamar al selector congelado, conservar las reglas de ingeniería originales y alinear etiqueta de frontera y política de días incompletos antes de aprender con Train. No se alteró ese archivo para introducir estos cambios.

## Reproducibilidad

Activar cervecera_env (Python 3.12), instalar requirements.txt y abrir notebooks/seleccion_de_variables.ipynb desde la raíz. El notebook carga JSON y CSV archivados incluidos en la propuesta para Git. No requiere Excel ni cachés, y no ejecuta los comandos costosos de esta sección. Las tablas muestran resultados y controles leídos, no cifras inventadas en celdas.

Para nuevas reproducciones explícitas, los Excel se obtienen del aula virtual según README y se verifican con data/checksums.json. Cada salida nueva debe ser un directorio distinto: no sobrescribir resultados congelados.

```bash
# Cuatro métodos individuales: nueva ejecución y salidas identificadas automáticamente.
python src/experimentos_seleccion.py
# Forward tradicional independiente: directorio nuevo y límite efectivo explícito.
python src/forward_tradicional_independiente.py --max-features 60 --salida results/seleccion/reproduccion-forward-max60
# Comparación y baselines: lee candidatos ya congelados; no se ejecutó en este PR.
python src/comparar_selecciones_validation.py --salida results/seleccion/reproduccion-validation
```

El primer comando conserva resultados de búsqueda interna; su ganador interno no reemplaza la elección oficial basada después en Validation. Los candidatos congelados se materializaron desde selecciones_por_metodo.json y el mejor subconjunto registrado de Forward. Sus listas y procedencia están versionadas y se verifican en el notebook; no hay una nueva selección automática al cargar el JSON definitivo. Para estudiar la ejecución original exacta, los paquetes de evidencia conservan configuraciones, código y registros; codigo_comparador_validation_20261113.py conserva el comparador original y coincide con codigo_sha256 del contrato. Los scripts actuales solo adaptan rutas/firmas y exponen salida nueva; esa adaptación no modifica resultados originales.

Los tiempos por etapa de métodos individuales constan en resumen_resultados.csv del ZIP y aparecen en la tabla 8. Forward max60 registró 449.88389778137207 segundos, aproximadamente 450. No hay una duración registrada de la comparación común: no se inventa una estimación. Cargar el contrato y leer el notebook es independiente de esos experimentos costosos; su duración actual se registra al ejecutarlo.

El 1 % consta en el candidato fechado antes de evaluar; el comparador verificaba su hash antes del ajuste y resultados.json conserva la huella de aquella versión. portabilidad.json vincula los metadatos relativos con esa huella original. Es evidencia interna, sin certificación externa de fecha. Las limitaciones de publicación a las 23:59 y del calendario físico siguen vigentes.

## Pruebas y reserva de Test

La suite se ejecuta con python -m pytest tests -q. Incluye pruebas sintéticas explícitas de independencia de etiquetas de evaluación, frontera shift(-1), columnas inexistentes y contrato de esquema con el Pipeline recibido. tests/conftest.py omite únicamente la prueba compartida que procesa los tres tramos reales completos: se informa como omitida para cumplir la reserva de Test, no como aprobada. Las pruebas unitarias con Ridge usan tablas sintéticas y no constituyen modelos definitivos ni una comparación de Validation real.

## Archivos para revisión

archivos_para_commit.txt delimita selección e incluye las compatibilidades existentes de src/variables_interacciones.py y src/limpieza.py, necesarias para mapa y reglas JSON. No se reimplementaron esas funciones. src/auxiliar_functions.py, .gitattributes de la raíz, el CSV inicial y ver_progreso.ps1 conservan su trabajo previo, fuera de la propuesta de selección. Los cambios legítimos del punto de entrada y sus pruebas ya llegan por el merge de origin; no deben presentarse como desarrollos de selección. Los artefactos, paquetes compactos y código histórico necesarios están en el manifiesto; matrices masivas y copias locales quedan fuera.

Falta la revisión del usuario de los archivos y autorización del commit de merge y selección. Luego podrá publicarse la rama y abrirse el PR con base feature/preprocessing usando la plantilla del equipo. Otra persona debe revisarlo y usar Create a merge commit; no se mergea el PR propio.

## Resultado ejecutado y propuesta final

Notebook reiniciado y ejecutado con cervecera_env: nueve celdas de código, cero errores, 8.86 segundos. Pruebas tras el merge: python -m pytest tests -q en Python 3.12.14, 131 aprobadas, cero fallidas y una omitida para proteger Test real, en 17.49 segundos. Las pruebas de frontera, independencia de etiquetas, columnas ausentes y esquema compartido están incluidas en ese resultado. Los comandos de regeneración Forward y comparador se comprobaron con --help, sin ejecutar experimentos.

El manifiesto actualizado contiene 64 archivos: 62 propios de selección y dos compatibilidades compartidas explícitas (mapa y JSON). Son una propuesta para revisión, no archivos ya preparados por esta tarea. El índice contiene las siete contribuciones recibidas del merge; sus archivos conservan la versión de origin, excepto la fila de selección de la guía que permanece sin preparar. Los cambios propios siguen sin preparar. No hay conflictos sin resolver ni archivos eliminados. El commit del merge está pendiente porque el usuario pidió confirmar antes qué archivos se incluirán.

PR propuesto: base feature/preprocessing, revisión por otra persona y Create a merge commit. No se hizo commit, push ni apertura del PR. Falta confirmar el conjunto de archivos y finalizar el merge; después podrá publicarse la rama. El punto de entrada compartido conserva la conexión final como tarea de quien integra; el contrato de selección está disponible y comprobado sin modificarlo.
