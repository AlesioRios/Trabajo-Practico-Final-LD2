# ADR 0015: procedimiento configurable de selección de variables

* **Fecha:** 12/11/2026
* **Estado:** propuesta
* **Autor:** Juan Cruz Brown

## Contexto

La selección debe comparar procedimientos dentro de entrenamiento y conservar los componentes de las interacciones. La tabla ingenierizada global contiene reglas aprendidas que no pueden reutilizarse para validar sobre sus propios datos de ajuste.

## Decisión

Se propone evaluar configuraciones fijas reaprendiendo limpieza e ingeniería por pliegue, además del selector y del estimador evaluador. El evaluador recibe la tabla previa a limpieza, TimeSeriesSplit y un estimador con su preprocesamiento. El supuesto de gap cero corresponde a emisión al cierre: el consumo actual ya permite etiquetar la fila anterior. Los índices se conservan antes de máscaras. No se aprueba aún un método ganador, cantidad ni umbral final.

Las reglas de selección se devuelven como diccionario JSON y se aplican sin reaprender. La funcion seleccionar\_variables(X\_train, y\_train) falla explícitamente mientras no exista configuración aprobada; admite configuración explícita para experimentos. Los auxiliares de selección usan mediana y Lasso estandariza; es una implementación experimental, no el Pipeline final de integración.

## Justificación

Las dimensiones y candidatas provienen de la tabla 1 del notebook seleccion\_de\_variables; el inventario y la finitud, de las tablas 2 y 3; las transiciones y disponibilidad de etiquetas, de las tablas 4 a 6. Los rankings y redundancias de las tablas 7 y 8 son descriptivos y no justifican todavía una selección final.

## Alternativas descartadas

Validar una única tabla ingenierizada con todas las reglas aprendidas en entrenamiento permite que los pliegues evaluados influyan en esas reglas. Adoptar un método por defecto convertiría una decisión metodológica pendiente en una elección implícita. Se implementan Forward y Backward con candidatas y detención explícitas. Cada subconjunto cierra jerarquía antes de medir MAE interno; limpieza, ingeniería y preprocesamiento se reajustan por pliegue. La evaluación externa pasa solo su entrenamiento a la búsqueda. No se usa SequentialFeatureSelector porque no integra estas reglas por sí solo. RFE queda excluido por ahora.

## Limitaciones

No se ejecutaron comparaciones predictivas sobre la planta ni búsquedas de hiperparámetros. Los parámetros fijos requieren aprobación. Comparar procedimientos ajustados necesita evaluación temporal adicional. Las fechas rotuladas son inciertas y la interpretación por bloques es una convención heredada. Los padres agregados por jerarquía pueden aumentar el tamaño solicitado. La integración con el Pipeline definitivo sigue pendiente. Las pruebas sintéticas no prueban rendimiento real.

## Actualización del 13/11/2026

Se conservan Spearman, Pearson, información mutua y Lasso; se agregan Forward y Backward. Cada búsqueda requiere cantidad objetivo directa, máximo de pasos, máximo de ajustes, mejora mínima explícita y conjunto inicial para Forward. La cantidad efectiva puede aumentar por jerarquía. Los empates respetan el orden declarado. Una ronda incompleta por presupuesto no actualiza el conjunto vigente. Si una interacción no es reproducible en un pliegue se informa un error, sin cambiar automáticamente candidatas. La tabla histórica sí recibió una etiqueta procedente de validación externa; excluirla no cambia las bases. El notebook reaprende las bases del ranking sin esa etiqueta y conserva la copia histórica. Gap cero continúa provisionalmente aceptado bajo emisión al cierre y horizonte por bloques. Ningún wrapper se ejecutó sobre la planta.

