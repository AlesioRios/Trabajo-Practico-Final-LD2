"""Diagnósticos del candidato; compara archivos históricos por identidad de fila."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from diagnostico_xgboost import preparar_residuos, resumir_residuos, comprobar_metricas
from modelado_comun import RAIZ, cargar_dataset_final, filas_comunes, calcular_metricas


def comparar_predicciones(tabla, raiz=RAIZ):
    """Exige iguales índices, fechas y objetivos antes de comparar errores."""
    rutas = {
        'XGBoost inicial': 'xgboost_15_validacion/predicciones.csv',
        'XGBoost refinado': 'xgboost_15_refinado_validacion/predicciones.csv',
        'LightGBM inicial': 'lightgbm/predicciones_validacion.csv',
        'LightGBM segunda': 'lightgbm_2/predicciones_validacion.csv',
    }
    filas = [{'modelo': 'Gradient Boosting', **calcular_metricas(tabla.real, tabla.predicho), 'filas': len(tabla)}]
    for nombre, ruta in rutas.items():
        otra = pd.read_csv(raiz / 'results' / ruta, parse_dates=['DIA'])
        pd.testing.assert_frame_equal(tabla[['indice_csv', 'DIA', 'real']].reset_index(drop=True),
                                      otra[['indice_csv', 'DIA', 'real']].reset_index(drop=True), check_dtype=False)
        filas.append({'modelo': nombre, **calcular_metricas(otra.real, otra.predicho), 'filas': len(otra)})
    val = cargar_dataset_final()['validacion']
    comunes = filas_comunes(val)
    assert tabla.indice_csv.tolist() == val['X'].index[comunes].tolist()
    np.testing.assert_allclose(tabla.real, val['y'].loc[comunes])
    for nombre in ('persistencia', 'media_movil'):
        filas.append({'modelo': nombre, **calcular_metricas(val['y'].loc[comunes], val[nombre].loc[comunes]), 'filas': int(comunes.sum())})
    return pd.DataFrame(filas)


def graficar(tabla, estudio, importancia, salida):
    """Cinco diagnósticos, mayores errores, historial e importancia del estimador."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 120, 'savefig.dpi': 180})
    salida.mkdir(parents=True, exist_ok=True)
    for nombre in ('prediccion_vs_real', 'predicciones_temporales', 'distribucion_residuos',
                   'residuos_vs_predicciones', 'residuos_temporales', 'errores_mayores',
                   'historial_optimizacion', 'importancia_variables'):
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        if nombre == 'prediccion_vs_real':
            ax.scatter(tabla.real, tabla.predicho, alpha=.7, s=20)
            limites = [tabla[['real', 'predicho']].min().min(), tabla[['real', 'predicho']].max().max()]
            ax.plot(limites, limites, '--', color='black')
            ax.set(xlabel='Real (kW)', ylabel='Predicho (kW)')
        elif nombre == 'predicciones_temporales':
            ax.plot(tabla.DIA, tabla.real, label='Real', linewidth=1.1)
            ax.plot(tabla.DIA, tabla.predicho, label='Predicho', linewidth=1.1)
            ax.set(xlabel='Fecha rotulada D; objetivo del bloque siguiente', ylabel='Consumo (kW)')
            ax.legend()
        elif nombre == 'distribucion_residuos':
            ax.hist(tabla.residuo, bins=18, edgecolor='white')
            ax.axvline(0, color='black', linestyle='--')
            ax.axvline(tabla.residuo.mean(), color='#bd541a', label='Media')
            ax.set(xlabel='Residuo = real − predicho (kW)', ylabel='Filas')
            ax.legend()
        elif nombre in ('residuos_vs_predicciones', 'residuos_temporales'):
            temporal = nombre == 'residuos_temporales'
            ax.scatter(tabla.DIA if temporal else tabla.predicho, tabla.residuo, alpha=.7, s=20)
            ax.axhline(0, color='black', linestyle='--')
            ax.set(xlabel='Fecha rotulada D' if temporal else 'Predicho (kW)', ylabel='Real − predicho (kW)')
        elif nombre == 'errores_mayores':
            mayor = tabla.nlargest(10, 'error_absoluto').sort_values('error_absoluto')
            ax.barh(mayor.indice_csv.astype(str), mayor.error_absoluto)
            ax.set(xlabel='Error absoluto (kW)', ylabel='Índice de fila CSV')
        elif nombre == 'historial_optimizacion':
            completos = [t for t in estudio.trials if t.state.name == 'COMPLETE']
            ax.scatter([t.number for t in completos], [t.value for t in completos], label='Completo')
            ax.plot([t.number for t in completos], np.minimum.accumulate([t.value for t in completos]), label='Mejor acumulado')
            podados = [t for t in estudio.trials if t.state.name == 'PRUNED']
            ax.scatter([t.number for t in podados], [t.value for t in podados], marker='x', alpha=.5, label='Podado: MAE parcial')
            ax.set(xlabel='Trial (desde cero)', ylabel='MAE fuera de muestra (kW)')
            ax.legend()
        else:
            orden = importancia.sort_values('importancia')
            ax.barh(orden.variable, orden.importancia)
            ax.set(xlabel='Importancia por reducción de impureza', ylabel='Variable')
            fig.set_size_inches(14, 8)
        ax.grid(alpha=.2)
        ax.set_title('Gradient Boosting · ' + nombre.replace('_', ' '))
        fig.savefig(salida / f'{nombre}.png', facecolor='white')
        plt.close(fig)


def generar_diagnostico(estudio):
    """Lee el candidato ajustado, verifica predicciones y produce evidencia e informe."""
    salida = RAIZ / 'results/gradient_boosting'
    evaluacion = json.loads((salida / 'validacion.json').read_text(encoding='utf-8'))
    tabla = preparar_residuos(pd.read_csv(salida / 'predicciones_validacion.csv'))
    resumen = resumir_residuos(tabla)
    comprobar_metricas(resumen, evaluacion['metricas'])
    comparacion = comparar_predicciones(tabla)
    modelo = joblib.load(RAIZ / evaluacion['modelo'])
    datos = cargar_dataset_final()
    val = datos['validacion']
    np.testing.assert_allclose(modelo.predict(val['X'].loc[filas_comunes(val)]), tabla.predicho, rtol=1e-12)
    train = datos['entrenamiento']
    comunes = filas_comunes(train)
    metricas_train = calcular_metricas(train['y'].loc[comunes], modelo.predict(train['X'].loc[comunes]))
    importancia = pd.DataFrame({'variable': modelo.feature_names_in_, 'importancia': modelo['regressor'].feature_importances_})
    meses = tabla.assign(mes=tabla.DIA.dt.strftime('%Y-%m')).groupby('mes').agg(
        filas=('residuo', 'size'), sesgo=('residuo', 'mean'), mae=('error_absoluto', 'mean')).reset_index()
    productos = {'residuos': tabla, 'comparacion_validacion': comparacion, 'residuos_por_mes': meses,
                 'errores_mayores': tabla.nlargest(10, 'error_absoluto'), 'importancia_variables': importancia,
                 'resumen_residuos': pd.DataFrame([resumen]), 'metricas_entrenamiento': pd.DataFrame([metricas_train])}
    for nombre, t in productos.items():
        t.to_csv(salida / f'{nombre}.csv', index=False, lineterminator='\n')
    graficar(tabla, estudio, importancia, salida / 'graficos')
    preservacion = json.loads((salida / 'fuentes_previas_sha256.json').read_text(encoding='utf-8'))
    for nombre, huella in preservacion['archivos'].items():
        if hashlib.sha256((RAIZ / nombre).read_bytes()).hexdigest() != huella:
            raise ValueError(f'Se modificó una fuente histórica: {nombre}')
    log = RAIZ / 'results/experiment_logs.csv'
    prefijo = log.read_bytes()[:preservacion['log_previo_bytes']]
    if hashlib.sha256(prefijo).hexdigest() != preservacion['log_previo_sha256']:
        raise ValueError('El log anterior no se conservó como prefijo exacto.')
    propios = pd.read_csv(log).query("modelo == 'gradient_boosting'")
    assert propios.notas.is_unique
    esperados = sum(1 + len(t.user_attrs.get('pliegues', [])) for t in estudio.trials) + 1
    assert len(propios) == esperados
    propios.to_csv(salida / 'experiment_logs.csv', index=False, lineterminator='\n')
    verificacion = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'controles': {'100_trials_terminados': len(estudio.trials) == 100 and all(t.state.is_finished() for t in estudio.trials),
                                 'mismas_filas_comparacion': True, 'originales_intactos': True,
                                 'log_previo_preservado': True, 'candidato_reproduce_predicciones': True,
                                 'tracking_por_ajuste_y_trial': len(propios) == esperados, 'test_no_evaluado': True},
                    'registros_propios': len(propios), 'metricas_entrenamiento': metricas_train,
                    'modelo_sha256': hashlib.sha256((RAIZ / evaluacion['modelo']).read_bytes()).hexdigest()}
    (salida / 'verificacion.json').write_text(json.dumps(verificacion, indent=2), encoding='utf-8')
    peor_mes = meses.loc[meses.mae.idxmax()]
    representativos = meses.loc[meses.filas >= 20]
    if representativos.empty:
        representativos = meses
    subestima = representativos.loc[representativos.sesgo.idxmax()]
    sobreestima = representativos.loc[representativos.sesgo.idxmin()]
    mejor_base = comparacion.loc[comparacion.modelo.isin(['persistencia', 'media_movil']), 'mae'].min()
    texto = f'''# Gradient Boosting: estudio y validación del candidato

El estudio {estudio.study_name} finalizó con {len(estudio.trials)} trials. Ganó el trial {estudio.best_trial.number}, por MAE conjunto fuera de muestra {estudio.best_value:.6f} kW. Sus parámetros se guardan en validacion.json; la validación independiente no intervino en su selección.

La validación contiene {len(tabla)} observaciones idénticas por índice, fecha y objetivo a las cuatro referencias XGBoost y LightGBM. El MAE es {resumen['mae']:.6f}, RMSE {resumen['rmse']:.6f} y R² {resumen['r2']:.6f}. La mejora relativa frente al mejor baseline es {100*(1-resumen['mae']/mejor_base):.2f} %. Comparación completa en comparacion_validacion.csv.

El residuo se define como real menos predicho. El sesgo medio es {resumen['sesgo_real_menos_predicho']:.2f} kW; valores positivos indican subestimación. El percentil 95 del error absoluto es {resumen['p95_error_absoluto']:.2f} y el máximo {resumen['max_error_absoluto']:.2f} kW. Los diez mayores errores concentran {100*resumen['fraccion_error_top10']:.2f} % del error absoluto. No se eliminan observaciones extremas.

El mes con mayor MAE es {peor_mes['mes']}: {peor_mes['mae']:.2f} kW en {int(peor_mes['filas'])} filas, sesgo {peor_mes['sesgo']:.2f} kW. residuos_por_mes.csv y los gráficos temporales muestran heterogeneidad: un promedio global no representa todos los meses. Las fechas rotuladas pueden repetirse y estar desfasadas; el índice CSV es la identidad, conforme al ADR 0002. No se infiere causalidad ni independencia de los residuos.

Entre los meses con al menos veinte observaciones, el mayor sesgo positivo aparece en {subestima['mes']}: {subestima['sesgo']:.2f} kW en {int(subestima['filas'])} filas; el menor aparece en {sobreestima['mes']}: {sobreestima['sesgo']:.2f} kW en {int(sobreestima['filas'])} filas. Estos cambios de signo muestran por qué el sesgo global cercano a cero puede ocultar patrones temporales. Octubre contiene solo dos filas y no permite una conclusión mensual estable.

El MAE dentro de entrenamiento es {metricas_train['mae']:.2f} kW en {int(comunes.sum())} filas comunes. La brecha validación menos entrenamiento es {resumen['mae']-metricas_train['mae']:.2f} kW. Es un diagnóstico de sobreajuste limitado por el cambio de período y tamaño de muestra. El MAE CV usa períodos anteriores y ajustes menores; no es directamente comparable con el error de validación. No se asegura ausencia de sobreajuste ni significación estadística de diferencias pequeñas.

La importancia de las 15 variables proviene del estimador ajustado (reducción de impureza), no de Optuna. Es descriptiva, puede repartirse entre variables correlacionadas y no demuestra causalidad. Se conserva mediana y StandardScaler mediante la variante A; se aprende de nuevo dentro de cada fold. No se transforma el objetivo.

El tracking común conserva cada fold ajustado y cada trial con su estado. Los trials PRUNED/FAIL tienen métricas finales vacías; los folds completos conservan métricas parciales identificadas. detalle.json conserva inicios, finales, duraciones y métricas por fold; SQLite y progreso.jsonl permiten auditar la secuencia. Los timestamps de fold usan UTC; los de Optuna y del registrador común conservan el reloj local real. No se alteran fechas históricas.

La sección 3.1 incorpora un candidato adicional: Gradient Boosting no reemplaza los cuatro modelos obligatorios, y su cumplimiento conjunto exige también Random Forest y Ridge/Lasso. Se completa 3.2 para este estudio y 3.3 para este candidato. En 3.4 solo se guarda el candidato entrenado con train; selección global, entrenamiento con train+validation, evaluación final de test, versión semántica y registry quedan pendientes del cierre conjunto. No se modifican train_model.py ni el notebook compartido.

Persisten las limitaciones heredadas: selección previa de variables no anidada, consultas anteriores de validación y disponibilidad operacional de ciertas mediciones pendiente de certificar (ADR 0017). La evaluación es histórica, no una garantía del test oculto.
'''
    (salida / 'informe.md').write_text(texto, encoding='utf-8')
    return verificacion
