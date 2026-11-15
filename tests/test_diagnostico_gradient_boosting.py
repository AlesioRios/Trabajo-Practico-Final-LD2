"""Auditoría del estudio real y de los diagnósticos de Gradient Boosting."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import diagnostico_gradient_boosting as dg
from modelado_comun import calcular_metricas

SALIDA = dg.RAIZ / 'results/gradient_boosting'

def test_estudio_real_tracking_y_tiempos():
    detalle = json.loads((SALIDA / 'detalle.json').read_text(encoding='utf-8'))
    trials = detalle['trials']
    assert len(trials) == 100
    assert [t['trial'] for t in trials] == list(range(100))
    log = pd.read_csv(SALIDA / 'experiment_logs.csv')
    assert log.notas.is_unique
    for trial in trials:
        assert trial['estado'] in {'COMPLETE', 'PRUNED', 'FAIL'}
        assert pd.Timestamp(trial['fin']) >= pd.Timestamp(trial['inicio'])
        estado = log.loc[log.notas.str.contains(f"; trial={trial['trial']}; estado=")]
        assert len(estado) == 1
        if trial['estado'] == 'COMPLETE':
            metricas = calcular_metricas(trial['predicciones']['real'], trial['predicciones']['predicho'])
            for k, v in metricas.items():
                assert estado.iloc[0][k] == pytest.approx(v)
        else:
            assert estado[['mae', 'rmse', 'r2']].isna().all().all()
        for fold in trial['pliegues']:
            fila = log.loc[log.notas.str.contains(f"; trial={trial['trial']}; fold={fold['paso']};")]
            assert len(fila) == 1
            assert fold['ultimo_indice_ajuste'] + 1 < fold['primer_indice_evaluacion']
            assert pd.Timestamp(fold['fin_utc']) >= pd.Timestamp(fold['inicio_utc'])
            assert fold['duracion_segundos'] > 0
            for k in ['mae', 'rmse', 'r2']:
                assert fila.iloc[0][k] == pytest.approx(fold[k])
    evaluacion = json.loads((SALIDA / 'validacion.json').read_text(encoding='utf-8'))
    completos = [t for t in trials if t['estado'] == 'COMPLETE']
    assert evaluacion['trial'] == min(completos, key=lambda t: t['metricas']['mae'])['trial']

def test_diagnosticos_comparacion_y_controles():
    tabla = pd.read_csv(SALIDA / 'residuos.csv', parse_dates=['DIA'])
    comparacion = dg.comparar_predicciones(tabla)
    assert len(comparacion) == 7 and comparacion.filas.eq(171).all()
    np.testing.assert_allclose(tabla.residuo, tabla.real - tabla.predicho)
    assert len(list((SALIDA / 'graficos').glob('*.png'))) == 8
    verificacion = json.loads((SALIDA / 'verificacion.json').read_text())
    assert all(verificacion['controles'].values())
    rota = tabla.copy()
    rota.loc[0, 'indice_csv'] += 1
    with pytest.raises(AssertionError):
        dg.comparar_predicciones(rota)
