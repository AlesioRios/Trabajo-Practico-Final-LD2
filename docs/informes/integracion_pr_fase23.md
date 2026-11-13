# Selección: reproducción y conexión

La decisión metodológica y sus limitaciones están exclusivamente en el [ADR de selección](../decisions/0015-seleccion-de-variables.md). Las cifras y comprobaciones se muestran en el [notebook](../../notebooks/seleccion_de_variables.ipynb), ejecutable con el kernel cervecera_env. Esta es la única guía vigente de selección en este directorio.

## Cargar y conectar

Desde la raíz del repositorio, este comando imprime los nombres oficiales, sin datasets ni selección nueva:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from seleccion_definitiva import cargar_seleccion_definitiva; print('\n'.join(cargar_seleccion_definitiva()['variables']))"
```

En el punto de conexión del equipo, para tablas ya construidas con las reglas congeladas:

```python
from seleccion_de_variables import seleccionar_variables
from pipeline_sklearn import separar_xy
columnas = seleccionar_variables(X_train, y_train)
X_train, y_train = separar_xy(tabla_train, columnas)
```

La lista, su orden y las reglas provienen únicamente de `seleccion_definitiva.json`. Si hace falta reconstruir ingeniería, usar las reglas guardadas con las funciones compartidas, conservando la historia antes de filtrar filas. No aprender reglas con Validation o Test. La conexión debe alinear la etiqueta de frontera y la política de días incompletos antes de aprender con Train; la disponibilidad operativa sigue pendiente, como explica el ADR. El punto de entrada compartido no se modifica en este PR.

## Reproducir artefactos

El notebook carga los JSON versionados y el resumen CSV del paquete publicado en [historial](../../results/seleccion/historial/README.md). No necesita Excel ni una caché privada. Sus tablas identifican la procedencia y los tiempos registrados.

Las reproducciones costosas son explícitas y requieren los originales obtenidos mediante el mecanismo del proyecto, validados con `data/checksums.json`, y las dependencias de requirements.txt. Usar directorios nuevos, sin sobrescribir los artefactos oficiales:

```bash
python src/experimentos_seleccion.py
python src/forward_tradicional_independiente.py --max-features 60 --salida results/seleccion/reproduccion-forward-max60
python src/comparar_selecciones_validation.py --salida results/seleccion/reproduccion-validation
```

El ganador interno del primer comando no reemplaza el contrato definitivo. Los candidatos y la decisión oficial permanecen congelados; el notebook comprueba su coherencia. El código histórico archivado distingue lo ejecutado originalmente de los adaptadores actuales de rutas y firmas. No es necesario repetir búsquedas para consumir la selección.

## Compatibilidad compartida y revisión

`src/limpieza.py` serializa límites como diccionario de números o None; la aplicación los convierte a una serie numérica y admite también el formato histórico Series. `src/variables_interacciones.py` obtiene el mapa de componentes desde las mismas bases y nombres de los productos; crear_interacciones lo adjunta en attrs sin cambiar columnas ni valores. Las pruebas de compatibilidad cubren ambos contratos.

Antes de revisar, ejecutar el notebook desde cero y `python -m pytest tests -q`. La suite conserva la omisión de la prueba que procesa datos reales reservados; una comprobación estructural completa requiere autorización explícita y no debe entrenar ni evaluar modelos. El PR apunta a feature/preprocessing, lo revisa otro integrante y la integración posterior utiliza Create a merge commit. No aprobar ni fusionar el PR propio.
