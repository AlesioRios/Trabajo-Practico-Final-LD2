"""Contratos de búsqueda, aislamiento temporal, poda, métricas y reanudación."""

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modelado_xgboost as mx
from modelado_comun import calcular_metricas, filas_comunes, validacion_cruzada_temporal


@pytest.fixture
def entrenamiento():
    indices = pd.RangeIndex(100, 136)
    X = pd.DataFrame({f"x{i}": np.arange(36, dtype=float) for i in range(15)}, index=indices)
    X.iloc[::4, 0] = np.nan
    parte = {"X": X, "y": pd.Series(np.arange(1., 37.), index=indices),
             "fechas": pd.Series(pd.date_range("2026-01-01", periods=18).repeat(2), index=indices),
             "persistencia": pd.Series(1., index=indices), "media_movil": pd.Series(1., index=indices)}
    parte["persistencia"].iloc[[7, 8, 9, 20]] = np.nan
    return parte


class ModeloCero:
    """Devuelve cero y permite comprobar las métricas a mano."""
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.ultimo_indice = X.index.max()
        assert X.index.equals(y.index)
        return self

    def predict(self, X):
        assert self.ultimo_indice + 1 < X.index.min()
        return np.zeros(len(X))


def test_espacio_exacto():
    trial = optuna.trial.FixedTrial({"n_estimators": 100, "learning_rate": .001, "max_depth": 3,
                                    "min_child_weight": .1, "subsample": .5, "colsample_bytree": .5})
    mx.proponer_parametros(trial)
    assert set(trial.distributions) == {"n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree"}
    esperado = {"n_estimators": (100, 2000, False, optuna.distributions.IntDistribution),
                "learning_rate": (.001, .3, True, optuna.distributions.FloatDistribution),
                "max_depth": (3, 10, False, optuna.distributions.IntDistribution),
                "min_child_weight": (.1, 10., True, optuna.distributions.FloatDistribution),
                "subsample": (.5, 1., False, optuna.distributions.FloatDistribution),
                "colsample_bytree": (.5, 1., False, optuna.distributions.FloatDistribution)}
    for nombre, (low, high, log, tipo) in esperado.items():
        distribucion = trial.distributions[nombre]
        assert isinstance(distribucion, tipo)
        assert (distribucion.low, distribucion.high, distribucion.log) == (low, high, log)


def test_objetivo_metricas_acumuladas_modelos_nuevos_y_nan_intactos(entrenamiento):
    modelos, ajustes, evaluaciones = [], [], []
    class Espia(ModeloCero):
        def __init__(self, **params):
            super().__init__(**params)
            modelos.append(self)
        def fit(self, X, y):
            ajustes.append(X.copy())
            return super().fit(X, y)
        def predict(self, X):
            evaluaciones.append(X.copy())
            return super().predict(X)
    estudio = optuna.create_study(pruner=optuna.pruners.NopPruner())
    estudio.optimize(mx.crear_objective(entrenamiento, Espia), n_trials=1)
    trial = estudio.trials[0]
    assert len(modelos) == 5 and len({id(m) for m in modelos}) == 5
    fijos = {"reg_alpha": 0, "reg_lambda": 1, "gamma": 0, "objective": "reg:squarederror",
             "eval_metric": "mae", "tree_method": "hist", "random_state": 42, "n_jobs": 1}
    for modelo in modelos:
        assert {k: modelo.params[k] for k in fijos} == fijos
        assert "early_stopping_rounds" not in modelo.params
    reales = []
    for paso, (ajuste, evaluacion) in enumerate(validacion_cruzada_temporal().split(entrenamiento["X"])):
        pd.testing.assert_frame_equal(ajustes[paso], entrenamiento["X"].iloc[ajuste])
        parte = {k: v.iloc[evaluacion] for k,v in entrenamiento.items()}
        pd.testing.assert_frame_equal(evaluaciones[paso], parte["X"].loc[filas_comunes(parte)])
        reales.extend(parte["y"].loc[filas_comunes(parte)].tolist())
        assert trial.intermediate_values[paso] == pytest.approx(np.mean(reales))
        assert entrenamiento["fechas"].iloc[ajuste].max() <= parte["fechas"].min()
    assert list(trial.intermediate_values) == list(range(5))
    assert trial.user_attrs["metricas"] == pytest.approx(calcular_metricas(reales, np.zeros(len(reales))))
    assert trial.value == pytest.approx(np.mean(reales))
    assert any(X.isna().any().any() for X in ajustes)


def test_cargador_solo_entrega_entrenamiento(monkeypatch, entrenamiento):
    class NoConsultar(dict):
        def __getitem__(self, clave):
            assert clave == "entrenamiento"
            return super().__getitem__(clave)
    monkeypatch.setattr(mx, "cargar_dataset_final", lambda: NoConsultar(entrenamiento=entrenamiento))
    assert mx.cargar_entrenamiento() is entrenamiento


def test_podado_sin_metricas_finales_ni_registro(entrenamiento, tmp_path):
    class PodarTercero(optuna.pruners.BasePruner):
        def prune(self, estudio, trial):
            return trial.last_step == 2
    estudio = optuna.create_study(pruner=PodarTercero())
    estudio.optimize(mx.crear_objective(entrenamiento, ModeloCero), n_trials=1)
    trial = estudio.trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert list(trial.intermediate_values) == [0, 1, 2]
    assert "metricas" not in trial.user_attrs and "predicciones" not in trial.user_attrs
    mx.sincronizar_registro(estudio, tmp_path / "log.csv")
    assert not (tmp_path / "log.csv").exists()
    mx.exportar_resultados(estudio, tmp_path)
    tabla = pd.read_csv(tmp_path / "trials.csv")
    assert tabla.loc[0, "estado"] == "PRUNED" and "mae" not in tabla.columns


def test_fallo_conserva_motivo(entrenamiento):
    class ModeloRoto(ModeloCero):
        def fit(self, X, y):
            raise ValueError("fallo deliberado")
    estudio = optuna.create_study()
    estudio.optimize(mx.crear_objective(entrenamiento, ModeloRoto), n_trials=1, catch=(ValueError,))
    assert estudio.trials[0].state == optuna.trial.TrialState.FAIL
    assert "fallo deliberado" in estudio.trials[0].user_attrs["motivo"]
    assert "metricas" not in estudio.trials[0].user_attrs


def test_sqlite_reanuda_sin_duplicados_y_reproduce_tpe(tmp_path, entrenamiento):
    contrato = {"prueba": True}
    ruta, log = tmp_path / "estudio.sqlite3", tmp_path / "log.csv"
    objective = mx.crear_objective(entrenamiento, ModeloCero)
    estudio = mx.abrir_estudio(ruta, contrato)
    assert estudio.pruner._n_startup_trials == 5
    assert estudio.pruner._n_warmup_steps == 2
    assert estudio.pruner._interval_steps == 1
    assert estudio.pruner._n_min_trials == 3
    mx.ejecutar_busqueda(estudio, objective, total=7, ruta_log=log)
    del estudio
    estudio = mx.abrir_estudio(ruta, contrato)
    mx.ejecutar_busqueda(estudio, objective, total=12, ruta_log=log)
    mx.ejecutar_busqueda(estudio, objective, total=12, ruta_log=log)
    assert len(estudio.trials) == 12
    assert len(pd.read_csv(log)) == sum(t.state.name == "COMPLETE" for t in estudio.trials)
    entero = mx.abrir_estudio(tmp_path / "entero.sqlite3", contrato)
    mx.ejecutar_busqueda(entero, objective, total=12, ruta_log=tmp_path / "entero.csv")
    assert [t.params for t in estudio.trials] == [t.params for t in entero.trials]
    assert [t.value for t in estudio.trials] == [t.value for t in entero.trials]
    with pytest.raises(ValueError, match="incompatible"):
        mx.abrir_estudio(ruta, {"prueba": False})
    with pytest.raises(ValueError, match="100"):
        mx.ejecutar_busqueda(estudio, objective, total=101, ruta_log=log)


def test_estudio_interrumpido_no_continua_silenciosamente(tmp_path):
    ruta = tmp_path / "estudio.sqlite3"
    estudio = mx.abrir_estudio(ruta, {"prueba": True})
    estudio.ask()
    with pytest.raises(ValueError, match="RUNNING"):
        mx.abrir_estudio(ruta, {"prueba": True})


def test_xgboost_real_acepta_nan_y_respeta_cantidad_arboles(entrenamiento):
    modelos = []
    def fabrica(**params):
        modelo = mx.XGBRegressor(**params)
        modelos.append(modelo)
        return modelo
    trial = optuna.trial.FixedTrial({"n_estimators": 100, "learning_rate": .01, "max_depth": 3,
                                    "min_child_weight": 1., "subsample": 1., "colsample_bytree": 1.})
    valor = mx.crear_objective(entrenamiento, fabrica)(trial)
    assert np.isfinite(valor)
    assert all(m.get_booster().num_boosted_rounds() == 100 for m in modelos)


def test_reanudar_reproduce_podados_y_fallidos(tmp_path, entrenamiento):
    class ModeloVariable(ModeloCero):
        def fit(self, X, y):
            if self.params["n_estimators"] == 679:
                raise ValueError("fallo controlado después de proponer parámetros")
            return super().fit(X, y)
        def predict(self, X):
            super().predict(X)
            return np.full(len(X), self.params["n_estimators"])
    objective = mx.crear_objective(entrenamiento, ModeloVariable)
    contrato = {"prueba": "estados mixtos"}
    ruta = tmp_path / "mixto.sqlite3"
    estudio = mx.abrir_estudio(ruta, contrato)
    mx.ejecutar_busqueda(estudio, objective, total=7, ruta_log=tmp_path / "mixto.csv")
    estudio = mx.abrir_estudio(ruta, contrato)
    mx.ejecutar_busqueda(estudio, objective, total=14, ruta_log=tmp_path / "mixto.csv")
    entero = mx.abrir_estudio(tmp_path / "entero.sqlite3", contrato)
    mx.ejecutar_busqueda(entero, objective, total=14, ruta_log=tmp_path / "entero.csv")
    assert {t.state.name for t in estudio.trials} == {"COMPLETE", "PRUNED", "FAIL"}
    assert [(t.params, t.value, t.state) for t in estudio.trials] == [
        (t.params, t.value, t.state) for t in entero.trials]
