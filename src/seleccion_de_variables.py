"""Selección experimental explícita y consumo predeterminado del conjunto oficial."""
import copy
import time
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from variables_interacciones import columnas_candidatas, construir_mapa_interacciones
from auxiliar_functions import SEMILLA
from limpieza import aprender_reglas_de_limpieza, limpiar, COLUMNAS_DE_CONTROL, _exigir_columnas

# Política de compatibilidad provisional; no representa un resultado experimental.
CONFIGURACION_PROVISIONAL = {"metodo": "spearman", "k": 1, "provisional": True}


def fitear_reglas_de_variables(entrenamiento_limpio):
    """Delega al pipeline al llamar, conservando el nombre público sin un ciclo al importar."""
    from preprocessing_pipeline import fitear_reglas_de_variables as aprender
    return aprender(entrenamiento_limpio)


def transform_variables(tabla_limpia, reglas):
    """Delega la transformación al pipeline mediante una importación local."""
    from preprocessing_pipeline import transform_variables as transformar
    return transformar(tabla_limpia, reglas)


def cerrar_jerarquia(elegidas, componentes, disponibles):
    """Conserva los padres de cada interacción, sin duplicados y con orden estable."""
    salida = list(dict.fromkeys(elegidas))
    for nombre in salida:
        for padre in componentes.get(nombre, []):
            if padre not in disponibles:
                raise ValueError(f"Falta componente: {padre}")
            if padre not in salida:
                salida.append(padre)
    return salida


def aprender_seleccion(X, y, configuracion, componentes=None, *,
                       tabla_bruta=None, estimador=None, particion_interna=None):
    """Aprende una regla JSON con entrenamiento alineado y objetivo finito.

    Configuración explícita: metodo y k; para Lasso admite criterio=no_cero.
    Mutua requiere vecinos y discretas. Lasso requiere parametros.
    Wrappers requieren tabla_bruta, estimador y particion_interna explícitos.
    Los selectores que requieren datos completos usan mediana; Lasso estandariza.
    No modifica entradas ni ajusta hiperparámetros. Los tiempos son informativos.
    """
    componentes = componentes or {}
    if any(c in COLUMNAS_DE_CONTROL for nombre, padres in componentes.items()
           for c in [nombre, *padres]):
        raise ValueError("El mapa de componentes contiene objetivo o columnas de control.")
    if configuracion["metodo"] in ("forward", "backward"):
        return _buscar_secuencial(tabla_bruta, configuracion, componentes or {},
                                  estimador, particion_interna)
    inicio = time.perf_counter()
    cfg = copy.deepcopy(configuracion)
    bases = {p for padres in componentes.values() for p in padres}
    sin_mapa = [c for c in X.columns if " x " in c and c not in componentes and c not in bases]
    if sin_mapa:
        raise ValueError(f"Falta mapa de componentes: {sin_mapa}. "
                         "Proporcionar componentes=construir_mapa_interacciones(reglas_variables) "
                         "o usar seleccionar_variables con las reglas o attrs de ingeniería.")
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("X e y deben tener el mismo índice único.")
    if not X.columns.is_unique:
        raise ValueError("Hay nombres de columnas repetidos.")
    if "objetivo" in X:
        raise ValueError("El objetivo debe estar separado de X.")
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("El objetivo debe ser finito; no se imputa.")
    columnas = columnas_candidatas(X)
    if not columnas:
        raise ValueError("No hay candidatas variables.")
    valores = X[columnas].astype(float)
    if np.isinf(valores.to_numpy()).any():
        raise ValueError("Hay infinitos en las candidatas.")
    metodo = cfg["metodo"]
    if metodo in ("spearman", "pearson"):
        puntajes = valores.corrwith(y, method=metodo).abs()
    elif metodo == "mutua":
        datos = SimpleImputer(strategy="median").fit_transform(valores)
        mascara = [c in cfg["discretas"] for c in columnas]
        puntajes = pd.Series(mutual_info_regression(
            datos, y, discrete_features=mascara,
            n_neighbors=cfg["vecinos"], random_state=SEMILLA), index=columnas)
    elif metodo == "lasso":
        parametros = dict(cfg["parametros"])
        parametros["random_state"] = SEMILLA
        modelo = make_pipeline(SimpleImputer(strategy="median"),
                               StandardScaler(), Lasso(**parametros))
        modelo.fit(valores, y)
        puntajes = pd.Series(abs(modelo[-1].coef_), index=columnas)
    else:
        raise ValueError("Método no implementado.")
    ranking = puntajes.dropna().sort_values(ascending=False, kind="stable")
    if cfg.get("criterio") == "no_cero":
        if metodo != "lasso":
            raise ValueError("no_cero se reserva para Lasso.")
        directas = ranking[ranking > 0].index.tolist()
    else:
        k = cfg["k"]
        if not isinstance(k, int) or not 1 <= k <= len(ranking):
            raise ValueError("k fuera del rango disponible.")
        directas = ranking.head(k).index.tolist()
    if not directas:
        raise ValueError("La configuración no conserva variables.")
    elegidas = cerrar_jerarquia(directas, componentes, X.columns)
    return {"configuracion": cfg, "candidatas": columnas,
            "puntajes": {c: float(v) if np.isfinite(v) else None for c, v in puntajes.items()},
            "directas": directas, "seleccionadas": elegidas,
            "por_jerarquia": [c for c in elegidas if c not in directas],
            "componentes": componentes, "segundos": time.perf_counter() - inicio,
            "semilla": SEMILLA}


def preparar_datos_seleccion(tabla_train, reglas_variables):
    """Prepara X, y y mapa explícito desde entrenamiento ya ingenierizado, sin aprender.

    Exige DIA creciente, índice y columnas únicos y las reglas originales de ingeniería.
    Conserva constantes que puedan ser padres. Quita controles de X y filas sin objetivo
    solo después de crear rezagos; no imputa ni escala. Rechaza objetivos infinitos,
    componentes ausentes o interacciones incompatibles con las reglas, sin usar attrs.
    """
    _exigir_columnas(tabla_train, ["DIA", "objetivo"], "para preparar la selección")
    if not tabla_train.index.is_unique or not tabla_train.columns.is_unique:
        raise ValueError("La preparación de selección requiere índice y columnas únicos.")
    if not tabla_train["DIA"].is_monotonic_increasing:
        raise ValueError("El entrenamiento debe conservar el orden cronológico de DIA.")
    componentes = construir_mapa_interacciones(reglas_variables)
    necesarias = list(dict.fromkeys([*componentes, *(p for padres in componentes.values() for p in padres)]))
    _exigir_columnas(tabla_train, necesarias, "de ingeniería descritas por las reglas de interacciones")
    utilizables = tabla_train.loc[tabla_train["objetivo"].notna()]
    y = utilizables["objetivo"].copy()
    if y.empty or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Se requiere al menos un objetivo utilizable y todos deben ser finitos; no se imputan.")
    X = utilizables.drop(columns=COLUMNAS_DE_CONTROL, errors="ignore").copy()
    return X, y, componentes


def seleccionar_variables(X_train, y_train, configuracion=None, componentes=None, *,
                          reglas_variables=None, **contexto):
    """Carga nombres oficiales por defecto; configuración explícita permite experimentar.

    Las reglas explícitas son la fuente principal del mapa. Si también se da un mapa,
    ambos deben coincidir. Sin reglas admite el mapa explícito y, como complemento,
    attrs["componentes_interacciones"]. Nunca deduce componentes separando nombres.
    La preparación recomendada es preparar_datos_seleccion(tabla_train, reglas_variables).
    Sin configuración carga la lista oficial sin usar y_train ni recalcular selección.
    Las configuraciones explícitas conservan la interfaz experimental histórica.
    """
    if configuracion is None:
        from seleccion_definitiva import cargar_seleccion_definitiva, aplicar_conjunto_definitivo
        oficiales = cargar_seleccion_definitiva()
        if reglas_variables is not None and reglas_variables != oficiales["ingenieria"]:
            raise ValueError("Las reglas de ingeniería no coinciden con el conjunto definitivo")
        if componentes is not None:
            mapa = construir_mapa_interacciones(oficiales["ingenieria"])
            if componentes != mapa:
                raise ValueError("El mapa no coincide con la ingeniería definitiva")
        return aplicar_conjunto_definitivo(X_train, oficiales).columns.tolist()
    cfg = copy.deepcopy(configuracion)
    if not isinstance(cfg, dict) or "metodo" not in cfg:
        raise ValueError("La configuración de selección debe ser un diccionario con metodo.")
    if cfg.get("provisional", False):
        warnings.warn(
            "La configuración provisional no es definitiva ni representa un método ganador. "
            "k=1 no está respaldado por una comparación experimental. Debe sustituirse por "
            "la configuración elegida después de comparar métodos.", UserWarning, stacklevel=2)
    if reglas_variables is not None:
        mapa_reglas = construir_mapa_interacciones(reglas_variables)
        if componentes is not None and componentes != mapa_reglas:
            raise ValueError("El mapa explícito no coincide con las reglas de ingeniería proporcionadas.")
        componentes = mapa_reglas
    elif componentes is None:
        componentes = getattr(X_train, "attrs", {}).get("componentes_interacciones", {})
    if not isinstance(componentes, dict):
        raise ValueError("El mapa de componentes debe ser un diccionario de listas de dos nombres.")
    for nombre, padres in componentes.items():
        if (not isinstance(nombre, str) or not isinstance(padres, list) or len(padres) != 2
                or any(not isinstance(p, str) or not p for p in padres)
                or len(set(padres)) != 2 or nombre in padres):
            raise ValueError(f"Componentes inválidos para la interacción {nombre!r}: se requieren dos padres distintos.")
        if X_train is not None and nombre in X_train.columns:
            cerrar_jerarquia([nombre], componentes, X_train.columns)
    return aprender_seleccion(X_train, y_train, cfg, componentes, **contexto)["seleccionadas"]


def aplicar_seleccion(X, reglas):
    """Aplica una lista aprendida sin reaprender ni alterar filas o índices."""
    faltan = set(reglas["seleccionadas"]) - set(X.columns)
    if faltan:
        raise ValueError(f"Faltan columnas seleccionadas: {sorted(faltan)}")
    return X.loc[:, reglas["seleccionadas"]].copy()


def evaluar_temporal(tabla_train, particion, estimador, configuracion=None, *,
                     particion_interna=None, estimador_selector=None, columnas_fijas=None):
    """Evalúa configuración fija sobre tabla previa a limpieza, solo entrenamiento.

    Conserva posiciones antes de máscaras. El llamador fija TimeSeriesSplit y
    suministra un estimador con su preprocesamiento. None es referencia sin selección.
    gap=0 supone pronóstico al cierre y etiquetas del bloque actual disponibles.
    No busca parámetros: comparar mínimos ajustados requiere evaluación adicional.
    """
    if not isinstance(particion, TimeSeriesSplit):
        raise ValueError("Se requiere TimeSeriesSplit explícito.")
    if not tabla_train.index.is_unique:
        raise ValueError("Se requiere índice único.")
    if not tabla_train["DIA"].is_monotonic_increasing:
        raise ValueError("La tabla no está ordenada.")
    resultados = []
    for numero, (i, j) in enumerate(particion.split(tabla_train)):
        inicio = time.perf_counter()
        if i[-1] >= j[0]:
            raise ValueError("Entrenamiento y evaluación se superponen.")
        bruto = tabla_train.iloc[i]
        reglas_limpieza = aprender_reglas_de_limpieza(bruto)
        train_limpio = limpiar(bruto, reglas_limpieza)
        reglas_variables = fitear_reglas_de_variables(train_limpio)
        historia = limpiar(tabla_train.iloc[:j[-1] + 1], reglas_limpieza)
        creada = transform_variables(historia, reglas_variables)
        entrenamiento = creada.iloc[i]
        evaluacion = creada.iloc[j]
        entrenamiento = entrenamiento.loc[entrenamiento.objetivo.notna()]
        evaluacion = evaluacion.loc[evaluacion.objetivo.notna()]
        if entrenamiento.empty or evaluacion.empty:
            raise ValueError("Pliegue sin etiquetas utilizables.")
        candidatas = columnas_candidatas(entrenamiento)
        # Los padres constantes pueden ser necesarios para respetar jerarquía.
        X = entrenamiento.drop(columns=COLUMNAS_DE_CONTROL, errors="ignore")
        y = entrenamiento["objetivo"]
        componentes = construir_mapa_interacciones(reglas_variables)
        if columnas_fijas is not None:
            if configuracion is not None:
                raise ValueError("No combinar selección y columnas fijas.")
            faltan = set(columnas_fijas) - set(creada.columns)
            if faltan:
                raise ValueError(f"Candidatas no reproducibles en este pliegue: {sorted(faltan)}")
            elegidas, reglas = list(columnas_fijas), None
        elif configuracion is None:
            elegidas, reglas = candidatas, None
        else:
            reglas = aprender_seleccion(X, y, configuracion, componentes,
                tabla_bruta=bruto, estimador=estimador_selector,
                particion_interna=particion_interna)
            elegidas = reglas["seleccionadas"]
        ajustado = _clonar_con_semilla(estimador)
        ajustado.fit(entrenamiento[elegidas], y)
        prediccion = ajustado.predict(evaluacion[elegidas])
        resultados.append({"pliegue": numero,
            "mae": float(mean_absolute_error(evaluacion.objetivo, prediccion)),
            "cantidad": len(elegidas), "seleccionadas": elegidas,
            "indices_evaluados": evaluacion.index.tolist(),
            "fin_entrenamiento": int(i[-1]), "inicio_evaluacion": int(j[0]),
            "segundos": time.perf_counter() - inicio, "reglas": reglas})
    return resultados


def resumir_estabilidad(resultados):
    """Resume inclusión por pliegue y Jaccard entre conjuntos, sin fijar umbrales.

    Devuelve dos DataFrames: frecuencia de cada variable y similitud por par.
    Las bases pueden variar entre pliegues; se comparan nombres, no posiciones.
    """
    if not resultados:
        raise ValueError("No hay pliegues para resumir.")
    conjuntos = [set(r["seleccionadas"]) for r in resultados]
    nombres = sorted(set().union(*conjuntos))
    frecuencias = pd.DataFrame({"Variable": nombres,
        "Frecuencia": [sum(c in grupo for grupo in conjuntos) / len(conjuntos) for c in nombres]})
    pares = []
    for i, j in combinations(range(len(conjuntos)), 2):
        union = conjuntos[i] | conjuntos[j]
        pares.append({"Pliegue A": i, "Pliegue B": j,
            "Jaccard": len(conjuntos[i] & conjuntos[j]) / len(union) if union else 1.0})
    return frecuencias, pd.DataFrame(pares)


def _clonar_con_semilla(estimador):
    """Clona y fija la semilla también en los pasos de un Pipeline."""
    nuevo = clone(estimador)
    nuevo.set_params(**{k: SEMILLA for k in nuevo.get_params(deep=True)
                        if k.split("__")[-1] == "random_state"})
    return nuevo


def _buscar_secuencial(tabla, cfg, componentes, estimador, particion):
    """Búsqueda controlada: candidatos y detención siempre explícitos.

    Cada subconjunto se evalúa desde datos previos a limpieza. Si las reglas
    internas no generan una columna solicitada, falla sin prefiltrar. El presupuesto
    cuenta ajustes del estimador auxiliar; no incluye los del evaluador externo.
    Los empates respetan el orden declarado de candidatas. Una ronda incompleta
    por presupuesto no cambia el conjunto vigente. MAE menor es mejor.
    """
    inicio = time.perf_counter()
    if tabla is None or estimador is None or not isinstance(particion, TimeSeriesSplit):
        raise ValueError("Wrapper requiere tabla_bruta, estimador y TimeSeriesSplit interno.")
    campos = {"candidatas", "cantidad_objetivo", "max_pasos", "max_ajustes", "mejora_minima"}
    if not campos <= cfg.keys():
        raise ValueError(f"Falta configuración wrapper: {sorted(campos - cfg.keys())}")
    candidatas = list(cfg["candidatas"])
    if not candidatas or len(set(candidatas)) != len(candidatas):
        raise ValueError("Candidatas vacías o repetidas.")
    if any(c in COLUMNAS_DE_CONTROL for c in candidatas):
        raise ValueError("Una candidata es objetivo o control.")
    bases = {p for padres in componentes.values() for p in padres}
    for c in candidatas:
        if " x " in c and c not in componentes and c not in bases:
            raise ValueError(f"Falta mapa de componentes: {c}")
    objetivo = cfg["cantidad_objetivo"]
    if not isinstance(objetivo, int) or not 1 <= objetivo <= len(candidatas):
        raise ValueError("Cantidad objetivo inválida.")
    if any(not isinstance(cfg[k], int) or cfg[k] < 1 for k in ("max_pasos", "max_ajustes")):
        raise ValueError("Presupuestos deben ser enteros positivos.")
    if cfg["mejora_minima"] is not None and cfg["mejora_minima"] < 0:
        raise ValueError("Mejora mínima debe ser no negativa o None.")
    forward = cfg["metodo"] == "forward"
    directas = list(cfg["iniciales"]) if forward else candidatas.copy()
    if len(set(directas)) != len(directas) or not set(directas) <= set(candidatas):
        raise ValueError("Conjunto inicial inválido.")
    if forward and len(directas) > objetivo:
        raise ValueError("Iniciales supera cantidad objetivo.")
    disponibles = set(candidatas) | {p for v in componentes.values() for p in v}
    memoria, trayectoria = {}, []
    ajustes = 0
    def efectivas(lista):
        return cerrar_jerarquia(lista, componentes, disponibles)
    def medir(lista):
        nonlocal ajustes
        columnas = efectivas(lista)
        clave = tuple(sorted(columnas))
        if clave not in memoria:
            costo = particion.get_n_splits()
            if ajustes + costo > cfg["max_ajustes"]:
                raise StopIteration
            resultados = evaluar_temporal(tabla, particion, estimador, columnas_fijas=columnas)
            ajustes += costo
            memoria[clave] = float(np.mean([r["mae"] for r in resultados]))
        return memoria[clave]
    motivo = "max_pasos"
    try:
        actual = medir(directas) if directas else None
        for paso in range(cfg["max_pasos"]):
            if len(directas) == objetivo:
                motivo = "cantidad_objetivo"
                break
            opciones = []
            for c in candidatas:
                if (forward and c in directas) or (not forward and c not in directas):
                    continue
                propuesta = directas + [c] if forward else [v for v in directas if v != c]
                if set(efectivas(propuesta)) == set(efectivas(directas)):
                    continue
                opciones.append((medir(propuesta), propuesta))
            if not opciones:
                motivo = "sin_movimientos_efectivos"
                break
            mejor, propuesta = min(opciones, key=lambda par: par[0])
            if actual is not None and cfg["mejora_minima"] is not None:
                if actual - mejor < cfg["mejora_minima"]:
                    motivo = "mejora_minima"
                    break
            directas, actual = propuesta, mejor
            trayectoria.append({"paso": paso + 1, "mae_interno": actual,
                "directas": directas.copy(), "efectivas": efectivas(directas)})
        if len(directas) == objetivo:
            motivo = "cantidad_objetivo"
    except StopIteration:
        motivo = "max_ajustes"
    if not directas:
        raise ValueError("Presupuesto insuficiente para completar la primera ronda.")
    elegidas = efectivas(directas)
    return {"configuracion": copy.deepcopy(cfg), "candidatas": candidatas,
        "directas": directas, "seleccionadas": elegidas,
        "cantidad_directas": len(directas), "cantidad_efectiva": len(elegidas),
        "por_jerarquia": [c for c in elegidas if c not in directas],
        "componentes": componentes, "trayectoria": trayectoria,
        "ajustes": ajustes, "detencion": motivo,
        "segundos": time.perf_counter()-inicio, "semilla": SEMILLA}
