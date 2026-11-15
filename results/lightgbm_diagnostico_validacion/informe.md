# Diagnóstico archivado de LightGBM — Fase 3

Análisis del 15/11/2026, basado exclusivamente en `results/lightgbm/`,
`results/lightgbm_2/` y el log de experimentos. No se abrió el dataset, ningún modelo
serializado ni el conjunto de prueba. No se ejecutaron entrenamientos, predicciones
ni búsquedas. Las fuentes históricas se preservaron por SHA-256 en `verificacion.json`.

## 1. Datos, convención y comparación

Las dos búsquedas usaron las mismas 15 variables, nombres y orden, según sus
contratos archivados, cinco pliegues temporales con gap=1 y semilla 42. Cada ganador
se ajustó originalmente con 830 filas y se evaluó en las mismas 171 filas comunes
de validación. En este cierre solo se leyeron las predicciones ya guardadas.

El residuo se define como real menos predicho: positivo indica subestimación y
negativo, sobreestimación. La identidad se comprueba por indice_csv, no por DIA,
porque hay fechas repetidas. DIA es la fecha rotulada de emisión D; el objetivo es
el bloque siguiente, que no necesariamente coincide con el día físico D+1. Las
líneas temporales unen solo observaciones disponibles; no se imputan días omitidos.

| Búsqueda / referencia | MAE CV | MAE validación | RMSE validación | R² validación |
|---|---:|---:|---:|---:|
| Primera, trial 72 | 3982,24 | 2321,04 | 2937,76 | 0,4286 |
| Segunda, trial 99 | 3948,73 | 2276,74 | 2873,14 | 0,4535 |
| Persistencia | — | 2668,33 | 3319,43 | 0,2705 |
| Media móvil | — | 2497,65 | 3368,08 | 0,2490 |

La segunda búsqueda reduce MAE CV en 33,51 kW (0,84 %) y MAE de validación en
44,30 kW (1,91 %) frente a la primera. Es una mejora descriptiva pequeña; no se
afirma significación estadística. Frente a la media móvil, el segundo candidato
mejora aproximadamente 8,85 % en validación, menos del 10 %. Esto no acredita la
mejora ni el MAE exigidos en el test oculto. No se elige al ganador global aquí.

`comparacion_validacion.csv` conserva las métricas de candidatos y baselines;
`hiperparametros.csv` incluye parámetros fijos y optimizados completos, y
`espacios_busqueda.csv` conserva los rangos efectivos de los contratos originales.

## 2. Predicciones y residuos

![Predicción frente a real](prediccion_vs_real.png)

La diagonal representa predicción exacta. Ambos candidatos acompañan el nivel
del consumo, con dispersión visible y dificultad para anticipar algunas caídas
bruscas. No se recalibró ni se ajustó una recta sobre las predicciones.

![Real y predicho en el tiempo](predicciones_temporales.png)

Las series conservan los cambios de nivel y los desaciertos de ambas búsquedas.
La similitud visual no significa que sus errores sean iguales: las métricas se
verificaron por separado contra los JSON de validación y sus registros del log.

![Distribución de residuos](distribucion_residuos.png)

| Resumen | Primera | Segunda |
|---|---:|---:|
| Sesgo promedio (kW) | +8,50 | −2,57 |
| Mediana del residuo (kW) | −236,27 | −293,34 |
| Desvío del residuo (kW, muestral) | 2946,37 | 2881,57 |
| Percentil 95 del error absoluto (kW) | 6334,03 | 5742,03 |
| Mayor error absoluto (kW) | 7642,35 | 7285,36 |
| Subestimaciones / sobreestimaciones | 79 / 92 | 79 / 92 |
| Fracción del error absoluto en las diez peores filas | 17,12 % | 16,86 % |

El promedio casi nulo no representa precisión perfecta: conviven sobreestimaciones
y subestimaciones grandes. La segunda búsqueda reduce dispersión y cola del error
absoluto, aunque mantiene errores superiores a 7000 kW y el mismo conteo por signo.
El histograma no demuestra normalidad ni independencia. Los percentiles son
descriptivos de estas 171 observaciones, no intervalos de predicción futuros.

![Residuos frente a predicciones](residuos_vs_predicciones.png)

Se aprecia predominio de residuos negativos en la zona de predicciones altas y
errores positivos importantes en niveles intermedios o bajos. La nube no parece
completamente homogénea. Es una lectura visual, sin prueba formal de
heterocedasticidad ni atribución causal al funcionamiento de la planta.

![Evolución de residuos](residuos_temporales.png)

El sesgo global cancela diferencias temporales: en enero de 2023, sobre 29 filas,
la primera búsqueda subestima 1970,65 kW en promedio y la segunda 1763,20 kW. En
abril, sobre 22 filas, sobreestiman 1650,53 y 1706,69 kW respectivamente. La segunda
no mejora uniformemente en cada período. Octubre tiene solo dos filas y no es una
comparación mensual estable. `residuos_por_mes.csv` conserva los tamaños y valores.

## 3. Errores mayores

El máximo error de la primera búsqueda es una sobreestimación de 7642,35 kW en
indice_csv 994, emisión 14/04/2023: real 13306,00 y predicho 20948,35 kW. La fila
siguiente del ranking, índice 954 y emisión 03/03/2023, es una subestimación de
7442,32 kW: real 20679,00 y predicho 13236,68 kW. Hay errores grandes de ambos signos.

`errores_mayores.csv` conserva las diez mayores desviaciones de cada búsqueda,
incluyendo índices, fechas, valores reales, predichos y residuos. No se eliminan
filas según su error ni se atribuyen a fallas operativas que estas predicciones
no permiten comprobar. Los residuos completos se guardan en dos CSV propios.

## 4. Optimización y tracking

Ambos estudios tienen 100 trials: 38 COMPLETE y 62 PRUNED en el primero; 46 COMPLETE
y 54 PRUNED en el segundo, sin fallidos. Se conserva TPESampler(seed=42),
MedianPruner y el MAE conjunto de las 677 filas comunes de los cinco pliegues, en
vez de promediar sin ponderación los MAE de pliegues de distinto tamaño.

La segunda búsqueda amplió árboles a 150–900, tasa a 0,01–0,10, mínimo por hoja a
10–60 y fracción de columnas a 0,65–1,00. Mantuvo profundidad 3–6 y el límite
condicionado num_leaves <= min(31, 2**max_depth). El segundo ganador tiene 200
árboles, profundidad 3, ocho hojas y tasa 0,036985470389312915. No hay imputación,
escalado, transformación del objetivo ni early stopping adicionales. Ver ADR 0021/0022.

`auditoria_tracking.csv` concilia los 84 COMPLETE y las dos validaciones con las
86 filas LightGBM de `results/experiment_logs.csv`. Verifica hiperparámetros
completos, MAE/RMSE/R², filas, semilla, commits y timestamps originales. Las métricas
de los CSV de validación coinciden con las históricas con rtol=1e-10 y atol=1e-8,
sin convertir las predicciones LightGBM a float32 ni modificar resultados.

El cumplimiento literal de 3.2 sigue siendo parcial: el CSV contiene un registro
por experimento COMPLETE que agrupa cinco ajustes; los podados están en los estudios
y sus JSON. Los timestamps son de registro, no del comienzo de cada ajuste, y no
tienen zona horaria. Las corridas tienen arbol_sin_commitear=True: se preserva ese
hecho, respaldado por snapshots de código y hashes. No se fabrican registros,
fechas ni métricas finales para completar retrospectivamente el tracking.

## 5. Reproducción y límites

Con Python 3.12 y las dependencias existentes, desde la raíz del repositorio:

```powershell
python -B src/diagnostico_lightgbm.py
python -B src/actualizar_notebook_lightgbm.py
python -B -m pytest tests/test_diagnostico_lightgbm.py -q -p no:cacheprovider
```

El primer comando genera solo las tablas, figuras y verificación del diagnóstico;
este informe interpretativo se conserva como documentación. El segundo genera y
ejecuta únicamente el notebook LightGBM de consulta, que no contiene estimadores.
Requiere el kernel cervecera_env. El tercero no ejecuta entrenamiento ni Optuna.
No ejecutar las suites de modelado ni `comparar_lightgbm.py`: contienen operaciones
con estimadores que quedan fuera del alcance autorizado de este cierre.

Se reutilizan las funciones puras de `src/diagnostico_xgboost.py`, ya integrado
en la base, sin modificarlo. Las fuentes originales se verifican por SHA-256.
La firma semántica de las filas LightGBM permite verificar su preservación aunque
el log común reciba registros de otras familias. No se agregaron dependencias.

Con estos gráficos y conclusiones queda cubierto el diagnóstico descriptivo de
3.3 para ambos candidatos LightGBM. La importancia de variables corresponde al
mejor modelo global; la f-ANOVA existente es importancia de hiperparámetros. No
se extraen importancias de los modelos guardados ni se inventan valores aquí.
Quedan para integración el notebook común, comparación de familias, selección,
reentrenamiento final, registry, versión semántica y etiqueta Git. Los modelos
guardados actualmente son de validación, no sustituyen esos entregables.

La selección de variables no está anidada y validación ya intervino en decisiones
anteriores. Este análisis añade inspección del mismo período, sin transformarlo
en una evaluación independiente ni justificar otro ajuste. Los cambios mensuales
no prueban causa operativa o generalización. El test permanece reservado.
