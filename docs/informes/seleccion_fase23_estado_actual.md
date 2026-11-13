> Estado de integración actualizado: [revisión previa al PR](integracion_pr_fase23.md). Se recibió origin/feature/preprocessing en 2cfd478 sin crear commit; se conserva el punto de entrada compartido. Los estados de Git y prerrequisitos descritos en revisiones anteriores son históricos.

# Fase 2.3: Spearman definitivo

Se cierra la selección de características, no el entrenamiento definitivo ni la Fase 2.4. Se busca una entrada interpretable y parsimoniosa para predecir el frío del bloque diario siguiente al cierre del actual. La validez temporal operativa queda condicionada a las confirmaciones de fuentes y calendario detalladas abajo.

## Separación y procedimiento ejecutado

La tabla contiene 1212 bloques: Train 848 (70 %), Validation 182 (15 %) y Test 182 (15 %). Las etiquetas de Train van de 2020-07-01 a 2022-10-28; las de Validation, de 2022-10-29 a 2023-04-27. Son etiquetas originales, no fechas físicas certificadas. No se inspeccionó Test en este cierre. Cada etiqueta de entrenamiento se obtiene con consumo_dia.shift(-1) dentro de su tramo; se excluye la última para no importar el objetivo de Validation. Se invalidan bloques con menos de 24 lecturas. El registro histórico informa 829 etiquetas de Train utilizables.

Los cinco candidatos se eligieron inicialmente solo con Train. Spearman ordena por asociación monotónica absoluta y Pearson por asociación lineal absoluta, con cobertura mínima del 60 % y 20 pares. Información mutua estima dependencia con vecinos y semilla fija. Lasso selecciona coeficientes distintos de cero, incluyendo variables cuyos indicadores son activos. Para estos cuatro métodos se usaron cuatro pliegues externos, tres internos, gap=1 y Ridge de referencia; cantidad, escala y transformación se eligieron en los pliegues internos. Se conserva la jerarquía de las interacciones. Spearman usa k=11 directas y agrega tres componentes: 14 variables finales; Pearson k=15 termina en 19. Mutua termina en 311; Lasso, en 268, con alpha=3.1622776601683795. Las configuraciones exactas están en selecciones_por_metodo.json.

Forward tradicional empieza vacío y agrega por nombre la candidata de menor MAE medio con Ridge(alpha=100, solver=svd), tres pliegues temporales y gap=1. El máximo era 60 columnas efectivas, incluidos indicadores. La ronda 30 fue la mejor: 30 variables, MAE interno 2826.5222178434105 y 39/59/59 columnas por pliegue; la ronda 31 no mejoró. Ese MAE guio la búsqueda y no es rendimiento independiente. Los protocolos internos difieren: la comparación final usa preparación y evaluador comunes.

## Comparación congelada y decisión

Los cinco conjuntos se evaluaron sobre las mismas 176 observaciones con objetivo disponible en Validation, con mediana, indicadores y StandardScaler aprendidos en Train, sin transformación, y Ridge(alpha=100, solver=svd, semilla=42). Se conservaron índices y predicciones. No se repite esa evaluación en este cierre.

| Método | Variables | Columnas efectivas | MAE (kW) |
|---|---:|---:|---:|
| pearson | 19 | 37 | 2239.58 |
| spearman | 14 | 27 | 2247.82 |
| mutua | 311 | 619 | 2736.27 |
| lasso | 268 | 532 | 2723.35 |
| forward | 30 | 59 | 2377.22 |

Pearson obtuvo el menor MAE absoluto: sus predicciones guardadas tienen el menor error medio en esas filas. No se demostró una causa física ni superioridad general; su criterio lineal es compatible con Ridge, pero eso no demuestra la causa de la diferencia. Spearman quedó a 8.237117374156 kW, aproximadamente 0.3678 %, y utiliza cinco variables menos. El criterio de equivalencia práctica del 1 % se fijó antes de evaluar: figura en candidatos_congelados_validation/candidatos.json, fechado 2026-11-13T22:35:18.737995+00:00, con huella. No es una prueba de equivalencia estadística. Se elige Spearman por parsimonia. Validation participó en esta elección final, por lo tanto su MAE no es una estimación final independiente. Test sigue reservado para una única evaluación final posterior.

## Variables y reconstrucción exacta

La única fuente operativa de nombres, orden, reglas y preparación es [seleccion_definitiva.json](../../results/seleccion/comparacion_validation_congelada/seleccion_definitiva.json). Esta lista es una vista documental generada desde ese archivo, no una lista mantenida en código:

1. `Frio (Kw)`
2. `Frio (Kw) x frio_hace_2_dias`
3. `frio_media_7_dias`
4. `Sala Maq (Kw)`
5. `Frio (Kw) x Consolidado Agua | Agua Planta (Hl)`
6. `Servicios (Kw)`
7. `Frio (Kw) x Totalizadores Energia | KW Trafo 9`
8. `dia_anio_coseno`
9. `frio_hace_2_dias x Consolidado Agua | Agua Planta (Hl)`
10. `dia_anio_coseno x Consolidado Agua | Agua Planta (Hl)`
11. `KW Gral Planta`
12. `frio_hace_2_dias`
13. `Consolidado Agua | Agua Planta (Hl)`
14. `Totalizadores Energia | KW Trafo 9`

Se reutilizan limpiar y transform_variables con las reglas congeladas; nunca se aprende ingeniería nuevamente. frio_hace_2_dias es shift(2) por bloques; frio_media_7_dias usa rolling(7, min_periods=3), incluido el bloque actual. dia_anio_coseno es cos(2*pi*dayofyear(DIA + 1 día)/365.25). Los productos provienen del mapa construir_mapa_interacciones y de crear_interacciones; no se interpretan nombres para inventar reglas.

Las 27 columnas son las 14 mediciones o derivadas más 13 indicadores de ausencia observada en Train. Solo dia_anio_coseno no necesita indicador. Primero se imputan las medianas de Train, se estandarizan las 14 columnas con los centros y escalas guardados, y se agregan los 13 indicadores sin escalar, en el orden del artefacto. No hay 27 variables elegidas. El número puede variar en un preprocesador reajustado con otros datos; 27 describe exactamente esta comparación.

## Auditoría temporal y disponibilidad a las 23:59

No se detectó uso computacional de predictores de bloques futuros en las funciones revisadas. Las ventanas terminan hoy, los rezagos son positivos y la referencia al calendario futuro es determinista. Limpieza, selección de bases, umbrales, medianas, indicadores y escalas se aprenden en prefijos de Train. gap=1 protege las fronteras de pliegues de etiquetas desplazadas. La lista inicial no se aprende con Validation ni Test; Validation compara candidatos congelados.

| Fuente | Variables dependientes | Evidencia | Supuesto y confirmación faltante |
|---|---|---|---|
| Consolidado EE: Frio (Kw), Sala Maq (Kw), Servicios (Kw), KW Gral Planta | Mediciones directas, rezago, media y productos del frío | Cierre por bloque, ADR 0001, controles de validez y horas en Excel | Se supone publicado al emitir. Falta confirmar zona horaria, hora física, latencia de captura/publicación y revisiones posteriores para cada una de las cuatro columnas |
| Consolidado Agua: Agua Planta (Hl) | Agua y sus tres productos | Unión por bloque por cantidad y fecha; cierre >=23:59, ADR 0006 | Falta confirmar que corresponde al bloque actual y se publica al emitir, sin revisiones posteriores; zona horaria y latencia sin evidencia |
| Totalizadores Energia: KW Trafo 9 | Trafo y producto con frío | Unión por bloque y filtro de cierre de la ingeniería | Falta confirmar intervalo físico, reinicio, zona horaria y publicación; se supone disponible después de registrar el cierre |
| DIA y secuencia de bloques | Coseno, producto con agua e interpretación de rezagos/horizonte | ADR 0002 y 0012: fechas repetidas/faltantes, desfase de pocos días | Falta correspondencia externa bloque-fecha física y confirmar días consecutivos; el coseno usa etiqueta+1, no calendario físico demostrado |

Registrar 23:59 no demuestra disponibilidad antes de las 23:59:00: se aceptan segundos hasta 23:59:59. El instante operativo debe ser después de recibir todas las lecturas necesarias del cierre. No hay prueba de publicación en tiempo real ni ausencia de revisiones retrospectivas. La predicción demostrada por el código es del bloque siguiente usando bloques hasta t; no se certifica que sea exactamente el día civil t+1 a las 23:59. Estas limitaciones afectan al conjunto por sus dependencias; requieren confirmación de la cátedra o del proveedor. No se inventan horarios ni se declara certificada esa disponibilidad.

El punto de entrada compartido ejecutar_preprocesamiento todavía aprende con una última etiqueta global que puede provenir de Validation y no aplica la invalidación estricta de menos de 24 lecturas de los experimentos. No usar esa ruta para repetir selección ni aprender las reglas oficiales: usar el adaptador congelado. No se modificó ese trabajo ajeno. La integración del adaptador no depende de resolverlo; el cierre del pipeline general de Fase 2.4 sí requiere alinear esas políticas.

## Integración y evidencia

[Guía de ejecución](../guia-fase2.md), [metodología e historial](metodologia_fase23.md), [resultados](resultados_fase23.md), [ADR 0015](../decisions/0015-seleccion-de-variables.md) y [continuación](continuacion_fase23.md). El notebook solo lee artefactos y calcula tablas desde resultados guardados, sin búsquedas ni modelos. seleccion_definitiva reconstruye desde historia cruda y aplica las 27 columnas guardadas sin fit. seleccionar_variables sin configuración consume la lista oficial y rechaza nombres ausentes o reglas contradictorias.
