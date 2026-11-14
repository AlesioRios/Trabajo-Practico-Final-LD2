# ADR 0010: la transformación por asimetría y el escalado se aplican después de crear variables

- **Fecha:** 08/11/2026
- **Estado:** aceptada
- **Autor:** Alesio Ríos

## Contexto
El orden clásico de preprocesamiento es: tipos de dato, duplicados, valores faltantes y atípicos, transformación para corregir la asimetría (por ejemplo Box-Cox), escalado y, recién después, ingeniería de características. Hay que decidir si la transformación y el escalado van antes o después de crear las variables nuevas de la consigna 2.2 (rezagos, promedios móviles, razones e interacciones).

Hechos medidos en `X_train` limpio (sección 4 de `notebooks/preprocesamiento.ipynb`):
- De 262 columnas de medición, 138 tienen asimetría fuerte (mayor a 1 en valor absoluto): 113 positivas (cola hacia los valores altos) y 25 negativas (cola hacia los valores bajos, por ejemplo `CO 2 Linea 4 / Hl` con −4,5 o `KW Iluminacion L3` con −4,0). 90 columnas superan 2 en valor absoluto. Las más asimétricas son las razones por hectolitro de Consolidado KPI (52 de 54 con asimetría mayor a 1; mediana 6,8; máximo 14,4) y las de Totalizadores Efluentes (mediana 12,4).
- 173 columnas tienen algún cero y 1 tiene valores negativos, así que Box-Cox (solo acepta valores estrictamente positivos) no se puede aplicar tal cual.
- `Frio (Kw)`, el objetivo, tiene asimetría −0,31 (casi simétrico).

## Decisión
1. **Se mide la asimetría ahora** (diagnóstico, sin modificar datos). Los tipos de dato, los duplicados, los faltantes y los valores absurdos ya se resolvieron antes de crear variables (ADR 0003, 0006 y 0008).
2. **La transformación y el escalado se aplican después de crear las variables nuevas**, dentro de un `Pipeline` de scikit-learn ajustado solo con entrenamiento (consigna 2.4), y solo para los modelos que lo necesitan (Ridge y Lasso). Los modelos de árboles (Random Forest, XGBoost, LightGBM) no los necesitan.
3. **Transformación elegida: Yeo-Johnson** (acepta ceros y valores negativos y corrige tanto la asimetría positiva como la negativa), con `PowerTransformer` de scikit-learn; log(1+x) queda como alternativa a comparar solo para las columnas de asimetría positiva (no corrige la negativa).
4. **El objetivo no se transforma:** `Frio (Kw)` queda en kW porque el error (MAE) se mide en kW.
5. Las variables nuevas se calculan con los **valores reales** (kW, Hl, etc.), no con los transformados. `dataset_final.csv` se guarda sin transformar ni escalar.

## Justificación
- **Las razones cambian de significado.** El frío sobre el total de la planta, con kW reales, es el porcentaje que consume el frío. Con columnas ya estandarizadas, el denominador pasa por valores cercanos a 0 y negativos: la razón explota o cambia de signo.
- **El promedio de valores transformados no es el valor transformado del promedio.** Con tres días de 1.000, 1.000 y 100.000 kW, el promedio real es 34.000 y el promedio de los logaritmos equivale a unos 4.400. Para energía interesa el consumo real. El rezago, en cambio, sí conmuta con una transformación, así que para los rezagos el orden da lo mismo.
- **Las variables nuevas traen su propia asimetría.** Si se transformara antes, habría que transformar de nuevo después; la transformación tiene que ir al final para cubrir todo el conjunto.
- **Los parámetros se aprenden.** Yeo-Johnson y el escalador estiman valores con los datos. Ajustarlos una vez con todo `X_train` antes de validar hace que cada tramo de validación de `TimeSeriesSplit` ya haya influido en ellos; dentro del `Pipeline` se reajustan en cada tramo.
- **Lo que sí se resolvió antes es lo que las variables nuevas necesitan limpio.** Un `Sala Maq (Kw)` de 4.208.132 dentro de un promedio móvil de 7 días arruinaría la variable durante una semana; por eso los valores absurdos se anulan antes (ADR 0008). La asimetría no tiene ese efecto.
- **`predict.py` reutiliza el mismo `Pipeline`** ajustado, sin reescribir la transformación.

## Alternativas descartadas
- **Transformar antes de crear variables (orden clásico):** deja las variables nuevas sin tratar, rompe el significado de razones y promedios, obliga a transformar dos veces y ajusta los parámetros fuera de la validación cruzada.
- **No transformar nunca:** válida para los modelos de árboles, pero perjudica a Ridge y Lasso, que se comparan en la Fase 3. Se decide por modelo y se compara con y sin transformación.
- **Box-Cox:** descartada por los ceros y los valores negativos.

## Limitaciones
- La transformación no se probó todavía sobre el modelo final; su efecto se mide en la Fase 3.
- La asimetría se midió solo con `X_train`; las variables nuevas se vuelven a medir cuando existan.
