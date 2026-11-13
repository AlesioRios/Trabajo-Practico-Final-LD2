> Estado de integración actualizado: [revisión previa al PR](integracion_pr_fase23.md). Se recibió origin/feature/preprocessing en 2cfd478 sin crear commit; se conserva el punto de entrada compartido. Los estados de Git y prerrequisitos descritos en revisiones anteriores son históricos.

# Cierre de Fase 2.3 para revisión

Spearman definitivo: 14 variables y 27 columnas efectivas. Pearson obtuvo el menor MAE absoluto; Spearman se elige con el margen práctico previo del 1 %. La lista oficial y sus reglas están en seleccion_definitiva.json. [Informe completo](seleccion_fase23_estado_actual.md), [resultados](resultados_fase23.md), [historial](metodologia_fase23.md) y [guía de consumo](../guia-fase2.md).

## Cambios de este cierre

Se agregaron src/seleccion_definitiva.py, src/verificar_seleccion_definitiva.py y tests/test_seleccion_definitiva.py. Se conectó el valor predeterminado de seleccionar_variables al artefacto, conservando las configuraciones experimentales explícitas; las pruebas antiguas se adaptaron a esa separación. Se actualizaron README, guía, ADR 0015, notebook y documentos de informe, metodología, resultados y continuación. Se corrigieron referencias absolutas del artefacto definitivo y se actualizó su huella, sin cambiar lista, orden, métricas ni reglas. Se agregó evidencia de verificación y una política de Git que conserva paquetes históricos compactos y deja cachés y paquetes íntegros fuera del commit. No se eliminaron archivos.

## Validación ejecutada

Python 3.12.14 de cervecera_env. Suite completa: 126 aprobadas, 0 fallidas (18.46 segundos). Después del control adicional de columnas duplicadas y convertir el verificador en comando explícito: 28 pruebas afectadas aprobadas (8.67 segundos). pip check: sin dependencias rotas. El notebook se ejecutó de cero: cinco celdas seguras sin errores; validado con nbformat. Los controles de artefactos dieron todos True. Se revisaron los enlaces internos existentes.

El verificador reconstruyó exclusivamente el prefijo Train: 848 bloques, 829 filas utilizables y matriz 829 x 27 finita. Coincidieron nombres, orden, 13 indicadores, medianas y escalas con el contrato. La reconstrucción del prefijo real fue idéntica con y sin bloques posteriores. Fuentes y artefactos comparados coincidieron en SHA-256. No se recalculó Spearman, no hubo búsqueda ni modelos definitivos, no se reevaluó Validation y no se evaluó Test. El cargador existente lee los originales completos o su caché, pero el verificador consume solo Train; las otras comprobaciones leen resultados históricos o datos sintéticos.

Evidencia en results/seleccion/cierre_fase23/verificacion_train.json y pruebas_y_documentacion.json. Comando de reproducción seguro: python src/verificar_seleccion_definitiva.py.

## Git y archivos para commit

Rama feature/preprocessing-seleccion. No se hizo commit, push ni staging. El índice conserva exactamente los bytes iniciales. Ya estaban preparados docs/guia-fase2.md, src/preprocessing_pipeline.py y tests/test_preprocessing_pipeline.py. La guía ahora contiene también cambios de este cierre: el índice conserva su versión anterior, por lo que requiere revisión del diff antes de preparar la versión final.

[Lista exacta propuesta para el commit de selección](../../results/seleccion/cierre_fase23/archivos_para_commit.txt). Incluye artefactos previos de esta fase que todavía no estaban versionados, además de los cambios actuales. [Dependencias preexistentes para revisar por separado](../../results/seleccion/cierre_fase23/dependencias_preexistentes.txt). No ejecutar git add . ni git commit -a: mezclarían el trabajo de integración del equipo. Los archivos de estas dependencias permanecen intactos desde el inicio de este cierre. seleccion_de_variables.py y las pruebas de selección ya tenían modificaciones previas de esta misma fase; se conservan.

También eran preexistentes data/processed/dataset_unificado_inicial.csv y ver_progreso.ps1; se excluyen de la propuesta. El CSV conserva los bytes iniciales y coincide con la huella registrada; el indicador de modificación de Git requiere revisar la normalización de líneas junto con .gitattributes, sin restaurar datos. El manifiesto no incorpora Excel, cachés, checkpoints de Jupyter, bytecode, tablas compartidas ni los tres ZIP íntegros locales. Los checkpoints metodológicos de Forward sí son evidencia necesaria y están incluidos. Los paquetes compactos pesan unos once megabytes en total; ningún archivo propuesto supera cinco megabytes. La ausencia de archivos eliminados y el detalle de estado se registran en revision_git.json.

## Integración y límites pendientes

La siguiente fase puede cargar la lista y reconstruir variables mediante el adaptador, sin repetir selección. La preparación histórica es compatible y quedó reproducida. La conexión del entrenamiento definitivo, serialización, dataset_final.csv y data_lineage.json pertenece a Fase 2.4 y posteriores. No se editaron preprocessing_pipeline.py, preprocesamiento.ipynb, train_model.py ni predict.py en este cierre.

No se detectó fuga computacional en el conjunto oficial con el protocolo por bloques. No se certifica disponibilidad real a las 23:59: faltan zona horaria, latencia de publicación, revisiones y correspondencia bloque-fecha física para Consolidado EE, Consolidado Agua y Totalizadores Energia. Las lecturas admiten segundos posteriores a las 23:59:00; emitir tras recibir el cierre. El calendario usa DIA+1, y solo se demuestra predicción del bloque siguiente. El punto de entrada compartido conserva diferencias de frontera de etiqueta y días incompletos que deben alinearse antes de Fase 2.4. Lasso conserva una advertencia histórica de convergencia de un pliegue externo; su ajuste final registrado cumple tolerancia. Validation se usó para elegir y no es rendimiento final independiente; Test reservado.

## Revisión mínima previa al commit

La propuesta pasa de 58 a 59 archivos por results/seleccion/.gitattributes: conserva los bytes exactos de artefactos con huellas, también con core.autocrlf=true. Se relativizaron exclusivamente metadatos personales de candidatos.json y portabilidad.json; no se modificaron conjuntos, fecha, criterio previo, métricas ni reglas. sha256.txt corresponde al candidato con metadatos portables; resultados.json conserva la huella original de la evaluación, registrada en portabilidad.json. Los bytes previos se conservan en copias locales *.original-local.json ignoradas.

La lista aislada todavía no es autosuficiente sobre HEAD: seleccionar_variables importa construir_mapa_interacciones, función presente solo en el cambio preexistente de src/variables_interacciones.py, excluido de la propuesta. Se comprobó el ImportError usando la versión confirmada de ese módulo, sin datos ni modelos. La limpieza con reglas JSON también depende de la compatibilidad preexistente de src/limpieza.py. Es necesario coordinar la integración separada de esas dependencias; no se incorporan silenciosamente al commit de selección ni se modifica su código. La guía contiene cambios previos ya preparados y requiere coordinar qué versión se integra. El índice quedó intacto.

No se repitieron pruebas generales, datasets, ingeniería ni evaluaciones. Se comprobó únicamente carga del artefacto, importación, rutas vigentes, atributos de Git y conservación de valores durante la corrección de metadatos.

## Resolución delimitada de dependencias de integración

No se modificó ningún módulo compartido ni la guía. La implementación necesaria ya existe en el árbol de trabajo; debe incorporarse mediante un commit coordinado previo o junto a la integración, separado de los 59 archivos propios. dependencias_preexistentes.txt delimita ahora los tres archivos necesarios, no una autorización para incluir cambios ajenos automáticamente.

En src/variables_interacciones.py se necesita el import de itertools.combinations, construir_mapa_interacciones(reglas) y la propagación de componentes_interacciones en crear_interacciones. Se conserva exactamente la implementación existente: el mapa deriva de variables_base y nombres_de_interacciones, sin recalcular selección ni cambiar productos. El selector importa ese símbolo y lo usa para preparar matrices, validar reglas y comprobar jerarquía. Sin ese cambio, HEAD no satisface el import.

En src/limpieza.py la compatibilidad de lectura requiere que marcar_valores_absurdos convierta limite_superior con pd.Series(limite_superior, dtype=float).reindex(columnas) antes de comparar: interpreta dict JSON y None, además de Series históricas. La versión existente incorpora también salida JSON estándar en aprender_reglas_de_limpieza, con dict de números finitos o None e import de math.isfinite. No se reemplaza por otra lógica ni se vuelve a aprender limpieza con datos reales.

En src/preprocessing_pipeline.py debe coordinarse la versión de trabajo con cargar_tabla_completa(carpeta_datos=CARPETA_DATOS_POR_DEFECTO, usar_cache=True) y el contrato de caché con manifiesto/huellas que utilizan el verificador y las pruebas de reproducibilidad incluidas. La versión de HEAD exige carpeta_datos, mientras el verificador incluido llama al cargador sin argumentos: se comprobó esa incompatibilidad de firma sin ejecutar el cargador. Deben conservarse tanto los cambios ya preparados del punto de entrada como las mejoras no preparadas del cargador; este cierre no modifica ninguno.

La única diferencia preparada de docs/guia-fase2.md respecto de HEAD actualiza un recuento histórico de pruebas de 87 a 88. La versión final de trabajo actualiza el estado de la fase y no debe restaurar ese número histórico como resultado vigente. No hay dos cambios metodológicos incompatibles que fusionar. Puede integrarse la guía final manteniendo el registro histórico del índice/commit anterior; no requiere duplicar texto. La preparación de la versión final sigue siendo una acción posterior autorizable: el índice conserva la versión anterior y quedó intacto.

Comprobaciones ejecutadas únicamente para esta integración: importación del selector, carga del artefacto oficial, mapa con padres oficiales y lectura de límites JSON oficiales con tres filas sintéticas. Tres pruebas aprobadas (16.00 segundos): test_mapa_desde_reglas_sin_interpretar_nombres_y_json, test_reglas_limpieza_json_estandar_y_compatibilidad_series y test_reglas_tienen_prioridad_sobre_attrs_y_rechazan_mapa_contradictorio. No se ejecutaron modelos, selección, datasets, Validation, Test ni la suite general.

Archivos para versionar: los 59 del manifiesto propio, más los tres módulos compartidos del listado de dependencias en un commit coordinado. Son 62 rutas distintas para la integración completa, sin mezclar responsabilidades. .gitattributes de la raíz, src/auxiliar_functions.py y tests/test_preprocessing_pipeline.py conservan sus cambios preexistentes; quedan fuera de la propuesta propia y bajo responsabilidad de su integración correspondiente. También quedan fuera el CSV inicial modificado, ver_progreso.ps1 y los temporales/locales ya excluidos. La corrección JSON y el mapa están resueltos localmente; la incorporación sobre HEAD requiere versionar esos prerrequisitos, no otro desarrollo del selector.

Versiones exactas comprobadas de los prerrequisitos (SHA-256 de los archivos de trabajo, no identificadores de commit):

- `src/variables_interacciones.py`: `1b0165f793ac857564dcc9df3ef1038fb0630754b26999fddb3fb2683134ece7`.

- `src/limpieza.py`: `287ce057dd5d7d7d9f0aa8abafc9800ce3cadeda64b8de745ac1234362b4a2d8`.

- `src/preprocessing_pipeline.py`: `06cde441f7e1aa3c1e12ea56d109851f2baf67bb28970987f3b6a850a0195929`.
