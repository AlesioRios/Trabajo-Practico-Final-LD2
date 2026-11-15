"""Contratos de Gradient Boosting, aislamiento, métricas y persistencia real."""
import json
import sys
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import modelado_gradient_boosting as gb
from modelado_comun import calcular_metricas, validacion_cruzada_temporal

@pytest.fixture
def entrenamiento():
    X = pd.DataFrame({f'x{i}': np.arange(36, dtype=float) for i in range(15)})
    X.iloc[::4, 0] = np.nan
    return dict(X=X, y=pd.Series(np.arange(1., 37.)),
                fechas=pd.Series(pd.date_range('2026-01-01', periods=36)),
                persistencia=pd.Series(1., index=X.index), media_movil=pd.Series(1., index=X.index))

class ModeloCero:
    def __init__(self, **params):
        self.params = params
    def fit(self, X, y):
        self.ultimo = X.index.max()
        return self
    def predict(self, X):
        assert self.ultimo + 1 < X.index.min()
        return np.zeros(len(X))

@pytest.mark.parametrize('loss', ['squared_error', 'absolute_error', 'huber'])
def test_espacio_y_condicion(loss):
    estudio = optuna.create_study()
    estudio.enqueue_trial({'loss': loss})
    trial = estudio.ask()
    params = gb.proponer_parametros(trial)
    assert ('alpha' in params) == (loss == 'huber')
    for nombre, d in trial.distributions.items():
        rango = gb.ESPACIO[nombre]
        if nombre == 'loss':
            assert list(d.choices) == rango['choices']
        else:
            assert d.low == rango['low'] and d.high == pytest.approx(rango['high'])
            assert d.log == rango.get('log', False)
            assert d.step == rango.get('step', 1 if rango['tipo'] == 'int' else None)
    assert gb.PARAMS_FIJOS == {'random_state': 42, 'n_iter_no_change': None}

def test_folds_imputacion_y_escalado_exclusivos(entrenamiento, monkeypatch):
    modelos = []
    def fabrica(**params):
        m = gb.crear_pipeline(entrenamiento['X'].columns, **params)
        modelos.append(m)
        return m
    estudio = optuna.create_study(pruner=optuna.pruners.NopPruner())
    estudio.enqueue_trial({'n_estimators': 100, 'loss': 'squared_error'})
    estudio.optimize(gb.crear_objective(entrenamiento, fabrica), n_trials=1)
    assert len({id(m) for m in modelos}) == 5
    for m, (ajuste, evaluacion) in zip(modelos, validacion_cruzada_temporal().split(entrenamiento['X'])):
        X = entrenamiento['X'].iloc[ajuste]
        pasos = m['preprocessor'].named_transformers_['numericas']
        np.testing.assert_allclose(pasos['imputar'].statistics_, X.median())
        np.testing.assert_allclose(pasos['escalar'].mean_, X.fillna(X.median()).mean())
        assert ajuste[-1] + 1 < evaluacion[0]
        assert list(m.feature_names_in_) == list(X.columns)
    pred = estudio.best_trial.user_attrs['predicciones']
    assert estudio.best_value == pytest.approx(calcular_metricas(pred['real'], pred['predicho'])['mae'])
    assert entrenamiento['X'].isna().any().any()

def test_cargador_orden_oficial_sin_test():
    datos = gb.cargar_dataset_final()
    assert set(datos) == {'entrenamiento', 'validacion'}
    columnas = json.loads(gb.RUTA_LINAJE.read_text(encoding='utf-8'))['columnas']
    assert len(columnas) == 15
    assert list(gb.cargar_entrenamiento()['X'].columns) == columnas

def test_reanuda_tpe_condicional_y_tracking(tmp_path, entrenamiento):
    objetivo = gb.crear_objective(entrenamiento, ModeloCero)
    ruta, log = tmp_path / 'estudio.sqlite3', tmp_path / 'log.csv'
    estudio = gb.abrir_estudio(ruta, {'test': True})
    gb.ejecutar_busqueda(estudio, objetivo, total=7, ruta_log=log)
    estudio = gb.abrir_estudio(ruta, {'test': True})
    gb.ejecutar_busqueda(estudio, objetivo, total=12, ruta_log=log)
    previo = log.read_bytes()
    gb.ejecutar_busqueda(estudio, objetivo, total=12, ruta_log=log)
    assert log.read_bytes() == previo
    entero = gb.abrir_estudio(tmp_path / 'entero.sqlite3', {'test': True})
    gb.ejecutar_busqueda(entero, objetivo, total=12, ruta_log=tmp_path / 'otro/log.csv')
    assert [(t.params, t.value, t.state) for t in estudio.trials] == [(t.params, t.value, t.state) for t in entero.trials]
    tabla = pd.read_csv(log)
    assert tabla.notas.is_unique
    assert len(tabla) == sum(1 + len(t.user_attrs['pliegues']) for t in estudio.trials)
    assert pd.to_datetime(tabla.fecha_hora, format='mixed', utc=True).notna().all()
    assert (tmp_path / 'gradient_boosting/detalle.json').exists()
    with pytest.raises(ValueError, match='incompatible'):
        gb.abrir_estudio(ruta, {'test': False})
    with pytest.raises(ValueError, match='100'):
        gb.ejecutar_busqueda(estudio, objetivo, total=101, ruta_log=log)

@pytest.mark.parametrize('estado', ['PRUNED', 'FAIL'])
def test_estados_incompletos_sin_metricas_falsas(tmp_path, entrenamiento, estado):
    class Podar(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return trial.last_step == 2
    class Roto(ModeloCero):
        def fit(self, X, y):
            raise ValueError('fallo deliberado')
    estudio = optuna.create_study(pruner=Podar())
    estudio.optimize(gb.crear_objective(entrenamiento, Roto if estado == 'FAIL' else ModeloCero),
                     n_trials=1, catch=(ValueError,))
    assert estudio.trials[0].state.name == estado
    gb.sincronizar_registro(estudio, tmp_path / 'log.csv')
    tabla = pd.read_csv(tmp_path / 'log.csv')
    assert tabla.iloc[-1][['mae', 'rmse', 'r2']].isna().all()
    assert f'estado={estado}' in tabla.iloc[-1].notas
    assert 'metricas' not in estudio.trials[0].user_attrs

def test_running_rechazado_y_limite_evaluacion(tmp_path):
    estudio = gb.abrir_estudio(tmp_path / 'estudio.sqlite3', {'test': True})
    estudio.ask()
    with pytest.raises(ValueError, match='RUNNING'):
        gb.abrir_estudio(tmp_path / 'estudio.sqlite3', {'test': True})

def test_reanuda_con_podas_y_fallos_reales(tmp_path, entrenamiento):
    class Podador(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return trial.number % 3 == 1 and trial.last_step == 2
    base = gb.crear_objective(entrenamiento, ModeloCero)
    def objetivo(trial):
        if trial.number % 4 == 2:
            gb.proponer_parametros(trial)
            trial.set_user_attr('motivo', 'fallo controlado')
            raise ValueError('fallo controlado')
        return base(trial)
    contrato = {'test': 'estados mixtos'}
    ruta = tmp_path / 'mixto.sqlite3'
    estudio = gb.abrir_estudio(ruta, contrato)
    estudio.pruner = Podador()
    gb.ejecutar_busqueda(estudio, objetivo, total=6, ruta_log=tmp_path / 'mixto/log.csv')
    estudio = gb.abrir_estudio(ruta, contrato)
    estudio.pruner = Podador()
    gb.ejecutar_busqueda(estudio, objetivo, total=12, ruta_log=tmp_path / 'mixto/log.csv')
    entero = gb.abrir_estudio(tmp_path / 'entero.sqlite3', contrato)
    entero.pruner = Podador()
    gb.ejecutar_busqueda(entero, objetivo, total=12, ruta_log=tmp_path / 'entero/log.csv')
    assert {t.state.name for t in estudio.trials} == {'COMPLETE', 'PRUNED', 'FAIL'}
    assert [(t.params, t.state, t.value) for t in estudio.trials] == [(t.params, t.state, t.value) for t in entero.trials]
