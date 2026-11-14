#Pipeline de preprocesamiento reproducible (Fase 2).
#Debe reutilizarse tal cual desde predict.py, sin reimplementarlo.

import argparse
from pathlib import Path

import pandas as pd

from auxiliar_functions import (
    MINIMO_DIAS_MEDIA_MOVIL,
    VENTANA_MEDIA_MOVIL,
    calcular_baselines,
    cierres_de_otras_hojas,
    cierres_por_bloque,
    construir_diario,
    leer_consolidado_ee,
    unir_versiones,
)
from limpieza import aprender_reglas_de_limpieza, limpiar
from variables_interacciones import (aprender_reglas_de_interacciones, construir_mapa_interacciones,
                                     crear_interacciones)
from variables_planta import NOMBRES_NUEVAS as NOMBRES_RAZONES
from variables_planta import RAZONES, aprender_reglas_de_planta, crear_variables_planta
from variables_temporales import COLUMNA_FRIO
from variables_temporales import NOMBRES_NUEVAS as NOMBRES_TEMPORALES
from variables_temporales import crear_variables_temporales

# Cortes de la división temporal (consigna 2.4): 70 % entrenamiento, 15 % validación, 15 % prueba
FRACCION_FIN_ENTRENAMIENTO = 0.70
FRACCION_FIN_VALIDACION = 0.85

#----------------------------------------------------------------------------------------------------

def construir_tabla_completa(carpeta_datos):
    """Une los tres Excel y todas sus hojas en una tabla con una fila por día."""
    archivos = sorted(Path(carpeta_datos).glob("Totalizadores Planta *.xlsx"))
    assert len(archivos) == 3, "Faltan Excel en data/ (ver README, sección 'Datos')."

    cierres = []
    for archivo in archivos:
        nombre = archivo.stem.replace("Totalizadores Planta ", "")       # p. ej. "2020-2022"
        with pd.ExcelFile(archivo) as libro:                              # se cierra solo al salir
            tabla_ee, _ = leer_consolidado_ee(libro)                      # hoja del objetivo
            de_ee = cierres_por_bloque(tabla_ee, nombre)                  # un día por fila, con el objetivo
            otras = {hoja: pd.read_excel(libro, sheet_name=hoja)          # las demás hojas, sin leer EE dos veces
                     for hoja in libro.sheet_names if hoja != "Consolidado EE"}
        de_otras = cierres_de_otras_hojas(otras)                          # una fila por bloque, ADR 0006
        assert len(de_otras) == len(de_ee), f"{nombre}: distinta cantidad de bloques"
        cierres.append(pd.concat([de_ee, de_otras], axis=1))              # pegadas por número de bloque

    # Una versión por día (ADR 0003) y marca de día válido (ADR 0004 y 0005)
    return construir_diario(unir_versiones(pd.concat(cierres, ignore_index=True)))


def cargar_tabla_completa(carpeta_datos, usar_cache=True):
    """Devuelve la tabla completa; la guarda en data/interim/ para no reconstruirla (~6 min) cada vez.

    El archivo guardado se descarta solo si algún Excel o el código que arma la tabla
    (este archivo y auxiliar_functions.py) es más nuevo que él.
    """
    carpeta_datos = Path(carpeta_datos)
    archivo_cache = carpeta_datos / "interim" / "tabla_completa.pkl"
    fuentes = list(carpeta_datos.glob("Totalizadores Planta *.xlsx")) + [Path(__file__), Path(__file__).with_name("auxiliar_functions.py")]

    if usar_cache and archivo_cache.exists():
        ultima_modificacion = max(f.stat().st_mtime for f in fuentes)
        if archivo_cache.stat().st_mtime > ultima_modificacion:           # nada cambió desde que se guardó
            return pd.read_pickle(archivo_cache)

    tabla = construir_tabla_completa(carpeta_datos)                       # camino lento
    archivo_cache.parent.mkdir(exist_ok=True)
    tabla.to_pickle(archivo_cache)
    return tabla


def dividir_por_fecha(tabla):
    """Divide la tabla del día, sin mezclarla, en entrenamiento (70 %), validación (15 %) y prueba (15 %)."""
    assert tabla["DIA"].is_monotonic_increasing, "La tabla tiene que estar en orden cronológico."
    filas = len(tabla)
    fin_entrenamiento = int(FRACCION_FIN_ENTRENAMIENTO * filas)    # 848 de 1.212
    fin_validacion = int(FRACCION_FIN_VALIDACION * filas)          # 1.030 de 1.212
    return (tabla.iloc[:fin_entrenamiento].copy(),                 # el tramo más antiguo
            tabla.iloc[fin_entrenamiento:fin_validacion].copy(),
            tabla.iloc[fin_validacion:].copy())                    # el más reciente: se toca una sola vez, al final


#----------------------------------------------------------------------------------------------------
# Variables nuevas (consigna 2.2): se aprenden reglas con entrenamiento y se aplican a cualquier tabla.

def fitear_reglas_de_variables(entrenamiento_limpio):
    """Aprende, solo con el tramo de entrenamiento ya limpio, todo lo que hace falta para crear las variables.

    Devuelve un diccionario con dos reglas, que `transform_variables` recibe tal cual:
      - `planta`: umbral de cada denominador de las razones (ADR 0013).
      - `interacciones`: las 10 variables base elegidas y el umbral de correlación (ADR 0014).
    Las variables temporales no necesitan reglas: solo miran hoy y los días anteriores.
    """
    reglas_planta = aprender_reglas_de_planta(entrenamiento_limpio)
    # Las variables base se eligen con temporales y razones ya creadas, así que se crean primero en entrenamiento.
    # Mirar solo hacia atrás da el mismo valor en el entrenamiento solo que en la tabla completa.
    con_variables = crear_variables_planta(crear_variables_temporales(entrenamiento_limpio), reglas_planta)
    return {"planta": reglas_planta, "interacciones": aprender_reglas_de_interacciones(con_variables)}


def transform_variables(tabla_limpia, reglas):
    """Agrega a la tabla limpia las variables temporales, las razones de planta y las interacciones, en ese orden.

    Recibe la tabla limpia completa y en orden cronológico (los rezagos de validación necesitan los últimos
    días de entrenamiento), y las `reglas` que devolvió `fitear_reglas_de_variables`. No aprende nada:
    no mira el contenido de la tabla para decidir umbrales ni variables. Misma cantidad de filas y mismo
    índice; las columnas originales no cambian y las nuevas se agregan al final.
    """
    con_temporales = crear_variables_temporales(tabla_limpia)
    con_planta = crear_variables_planta(con_temporales, reglas["planta"])
    return crear_interacciones(con_planta, reglas["interacciones"])


#----------------------------------------------------------------------------------------------------
# Datos nuevos (Fase 4): un solo Excel, solo las hojas y columnas que el modelo usa, errores claros.

HOJA_OBJETIVO = "Consolidado EE"


class ErrorDeEntrada(ValueError):
    """El archivo de entrada no tiene lo que el modelo necesita; el mensaje dice qué falta."""


def resolver_dependencias(columnas_modelo, reglas):
    """Deduce qué hay que leer del Excel para construir las columnas que el modelo pide.

    `reglas` es el diccionario del linaje (`limpieza` y `variables`). No adivina por el texto del nombre:
    una columna es temporal si está en `variables_temporales.NOMBRES_NUEVAS`, una razón si está en
    `variables_planta.NOMBRES_NUEVAS`, un producto si está en el mapa de `construir_mapa_interacciones`, y
    cualquier otra es una medición del Excel. Devuelve `columnas_crudas` (mediciones a limpiar), `hojas`
    (las que hay que leer, siempre con la del objetivo), `bases` (variables base de las interacciones
    pedidas) y `razones` (si hay que calcular las razones de planta).
    """
    mapa = construir_mapa_interacciones(reglas["variables"])
    crudas, bases, razones = {COLUMNA_FRIO}, set(), False        # el frío siempre: de él salen los rezagos y los controles
    for columna in columnas_modelo:
        if columna in mapa:
            bases.update(mapa[columna])
        elif columna in NOMBRES_TEMPORALES:
            pass                                                  # solo necesita DIA y Frio (Kw)
        elif columna in NOMBRES_RAZONES:
            razones = True
        else:
            crudas.add(columna)
    for base in bases:                                            # las bases que no son temporales son mediciones
        if base not in NOMBRES_TEMPORALES:
            crudas.add(base)
    if razones:                                                   # las razones se calculan juntas
        crudas.update(c for _, numerador, denominador in RAZONES for c in (numerador, denominador))
    hojas = {HOJA_OBJETIVO} | {c.split(" | ")[0] for c in crudas if " | " in c}
    return {"columnas_crudas": sorted(crudas), "hojas": sorted(hojas), "bases": bases, "razones": razones}


def leer_hojas_de_prediccion(ruta, hojas_necesarias):
    """Lee de un Excel solo las hojas pedidas, por nombre. Devuelve (tabla del objetivo, {hoja: tabla} de las demás).

    Falla con `ErrorDeEntrada` si el archivo no existe, no se puede abrir, le falta alguna hoja o a alguna
    hoja le faltan `DIA`, `HORA` o (en `Consolidado EE`) `Frio (Kw)`. Los nombres de hoja se comparan sin
    espacios en los bordes. Las hojas de más se ignoran sin leerlas.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorDeEntrada(f"No se encontró el archivo {str(ruta)!r}.")
    try:
        libro = pd.ExcelFile(ruta)
    except Exception as error:                                    # formato inválido, archivo dañado o sin permisos
        raise ErrorDeEntrada(f"No se pudo abrir {ruta.name!r} como Excel: {error}") from error
    with libro:
        reales = {str(nombre).strip(): nombre for nombre in libro.sheet_names}
        faltan = [h for h in hojas_necesarias if h not in reales]
        if faltan:
            raise ErrorDeEntrada(f"Al archivo {ruta.name!r} le faltan las hojas {faltan}. Tiene: {libro.sheet_names}.")
        try:
            if reales[HOJA_OBJETIVO] == HOJA_OBJETIVO:
                tabla_objetivo, _ = leer_consolidado_ee(libro)           # el mismo lector del entrenamiento
            else:                                                       # el nombre real tiene espacios en los bordes
                tabla_objetivo = _leer_hoja_renombrada(libro, reales[HOJA_OBJETIVO])
        except KeyError as error:
            raise ErrorDeEntrada(f"A la hoja {HOJA_OBJETIVO!r} le falta la columna {error.args[0]!r}.") from error
        otras = {h: pd.read_excel(libro, sheet_name=reales[h]) for h in hojas_necesarias if h != HOJA_OBJETIVO}
    for hoja, tabla in {HOJA_OBJETIVO: tabla_objetivo, **otras}.items():
        faltan = [c for c in ("DIA", "HORA") if c not in tabla.columns]
        if faltan:
            raise ErrorDeEntrada(f"A la hoja {hoja!r} le faltan las columnas {faltan}. Tiene: {list(tabla.columns)[:12]}...")
    if COLUMNA_FRIO not in tabla_objetivo.columns:
        raise ErrorDeEntrada(f"A la hoja {HOJA_OBJETIVO!r} le falta la columna {COLUMNA_FRIO!r}.")
    return tabla_objetivo, otras


def _leer_hoja_renombrada(libro, nombre_real):
    """Igual que `leer_consolidado_ee`, para cuando la hoja se llama igual pero con espacios en los bordes."""
    tabla = pd.read_excel(libro, sheet_name=nombre_real)
    tabla = tabla.dropna(subset=["DIA"])
    tabla["DIA"] = pd.to_datetime(tabla["DIA"])
    return tabla


def construir_tabla_para_prediccion(ruta, hojas_necesarias):
    """Tabla con una fila por día de UN Excel, con las mismas columnas de control que en entrenamiento.

    Hace lo mismo que `construir_tabla_completa` (cierre de cada día, unión por bloque, marca de día válido),
    pero con un solo archivo y solo las hojas pedidas.
    """
    tabla_ee, otras = leer_hojas_de_prediccion(ruta, hojas_necesarias)
    de_ee = cierres_por_bloque(tabla_ee, Path(ruta).stem)
    if otras:
        de_otras = cierres_de_otras_hojas(otras)
        if len(de_otras) != len(de_ee):
            raise ErrorDeEntrada(f"Las hojas {sorted(otras)} y {HOJA_OBJETIVO!r} no tienen la misma cantidad de días "
                                 f"({len(de_otras)} contra {len(de_ee)}); no se pueden alinear.")
        de_ee = pd.concat([de_ee, de_otras], axis=1)
    return construir_diario(unir_versiones(de_ee))


def transform_para_prediccion(tabla_limpia, reglas, columnas_modelo, dependencias):
    """Crea, sobre la tabla limpia, solo las variables que el modelo usa y las devuelve en su orden.

    Mismas funciones que en entrenamiento (`crear_variables_temporales`, `crear_variables_planta`,
    `crear_interacciones`), con las mismas reglas aprendidas: solo cambia que las interacciones se calculan con
    las variables base necesarias (un producto A x B depende solo de A y de B, así que da el mismo valor).
    """
    tabla = crear_variables_temporales(tabla_limpia)
    if dependencias["razones"]:
        tabla = crear_variables_planta(tabla, reglas["variables"]["planta"])
    bases = [b for b in reglas["variables"]["interacciones"]["variables_base"] if b in dependencias["bases"]]
    if len(bases) >= 2:                                           # el orden de las bases define los nombres «A x B»
        tabla = crear_interacciones(tabla, {**reglas["variables"]["interacciones"], "variables_base": bases})
    faltan = [c for c in columnas_modelo if c not in tabla.columns]
    if faltan:
        raise ErrorDeEntrada(f"No se pudieron construir estas variables que el modelo necesita: {faltan}.")
    return tabla[list(columnas_modelo)]


def preparar_para_prediccion(ruta, columnas_modelo, reglas):
    """De un Excel nuevo a (X, fechas): las columnas que el modelo pide, sin imputar ni escalar.

    La imputación y el escalado los hace el `Pipeline` guardado con el modelo. `X` conserva los vacíos.
    """
    dependencias = resolver_dependencias(columnas_modelo, reglas)
    tabla = construir_tabla_para_prediccion(ruta, dependencias["hojas"])
    try:
        limpia = limpiar(tabla, reglas["limpieza"], dependencias["columnas_crudas"])
    except ValueError as error:                                   # falta una columna de medición: se dice cuál
        raise ErrorDeEntrada(f"Al archivo {Path(ruta).name!r}: {error}") from error
    X = transform_para_prediccion(limpia, reglas, columnas_modelo, dependencias)
    return X, limpia["DIA"]


#----------------------------------------------------------------------------------------------------
# Punto de entrada: python src/preprocessing_pipeline.py

CARPETA_DATOS_POR_DEFECTO = Path(__file__).resolve().parents[1] / "data"      # la carpeta data/ del repositorio


def ejecutar_preprocesamiento(carpeta_datos=CARPETA_DATOS_POR_DEFECTO, usar_cache=True):
    """Corre, en orden, todo lo que ya existe: de los Excel a la tabla con todas las variables.

    Pasos: tabla completa, división temporal, limpieza (reglas aprendidas solo con entrenamiento) y variables
    nuevas (reglas aprendidas solo con entrenamiento). Devuelve un diccionario con la tabla final, sus tres
    tramos y las reglas aprendidas, que es lo que usarán los pasos siguientes (selección, `Pipeline`, guardado).
    """
    tabla_completa = cargar_tabla_completa(carpeta_datos, usar_cache)
    X_train, _, _ = dividir_por_fecha(tabla_completa)
    reglas_limpieza = aprender_reglas_de_limpieza(X_train)                       # solo con entrenamiento
    tabla_limpia = limpiar(tabla_completa, reglas_limpieza)
    train_limpio, _, _ = dividir_por_fecha(tabla_limpia)
    reglas_variables = fitear_reglas_de_variables(train_limpio)                  # solo con entrenamiento
    tabla_con_variables = transform_variables(tabla_limpia, reglas_variables)
    entrenamiento, validacion, prueba = dividir_por_fecha(tabla_con_variables)
    return {"tabla_completa": tabla_completa, "tabla_limpia": tabla_limpia,
            "tabla_con_variables": tabla_con_variables,
            "entrenamiento": entrenamiento, "validacion": validacion, "prueba": prueba,
            "reglas": {"limpieza": reglas_limpieza, "variables": reglas_variables}}


CARPETA_SALIDA_POR_DEFECTO = CARPETA_DATOS_POR_DEFECTO / "processed"
# Se guardan en el CSV, pero NO son entradas del modelo (ADR 0017 y 0018). `consumo_dia` vale lo mismo que la
# variable `Frio (Kw)` (es el consumo de hoy) y es el baseline de persistencia. `baseline_media_movil` es el promedio
# de los últimos 7 días (mínimo 4 válidos). Se guardan aparte porque la Fase 3 las necesita para comparar los modelos
# contra los baselines, y la media móvil necesita los días anteriores, que el CSV ya no tiene (se quitaron las filas
# sin objetivo): por eso se calcula acá, con toda la tabla y mirando solo el pasado.
COLUMNA_PERSISTENCIA = "consumo_dia"
COLUMNA_MEDIA_MOVIL = "baseline_media_movil"
COLUMNAS_AUXILIARES = (COLUMNA_PERSISTENCIA, COLUMNA_MEDIA_MOVIL)


def _quitar_etiquetas_de_frontera(tabla_con_variables):
    """Devuelve una copia donde la última fila de entrenamiento y la de validación quedan sin objetivo.

    El objetivo de una fila es el consumo del día siguiente. En la última fila de cada tramo ese día ya
    pertenece al tramo que sigue, así que esa etiqueta cruzaría la frontera: se la deja vacía y
    `separar_xy` y `guardar_dataset_final` descartan esas filas. Son 2 filas de unas 1.030.
    """
    nueva = tabla_con_variables.copy()
    entrenamiento, validacion, _ = dividir_por_fecha(nueva)
    nueva.loc[[entrenamiento.index[-1], validacion.index[-1]], "objetivo"] = float("nan")
    return nueva


def completar_fase2(resultado, carpeta_salida=CARPETA_SALIDA_POR_DEFECTO, guardar=True):
    """Cierra la Fase 2: elige las variables, ajusta el Pipeline con entrenamiento y guarda el dataset final.

    `resultado` es el diccionario que devuelve `ejecutar_preprocesamiento`; `carpeta_salida`, la carpeta donde
    se escriben `dataset_final.csv` y `data_lineage.json` (el checksum va a `checksums.json`, una carpeta más
    arriba). Con `guardar=False` hace todas las comprobaciones sin escribir nada.

    El Pipeline se ajusta solo con `X_train` y se aplica a `X_val` y `X_test` solo para comprobar que corre y
    registrar sus shapes: lo transformado no se guarda ni se usa para medir nada con las etiquetas de prueba.
    """
    # Importaciones locales: pipeline_sklearn importa este módulo al cargarse, y arriba formarían un ciclo.
    from pipeline_sklearn import (ALPHA_ELEGIDO, VARIANTE_POR_DEFECTO, aprender_reglas_de_grupos, armar_pipeline,
                                  escribir_linaje, guardar_dataset_final, separar_xy)
    from seleccion_de_variables import seleccionar_variables

    tabla = _quitar_etiquetas_de_frontera(resultado["tabla_con_variables"])
    if COLUMNA_PERSISTENCIA not in tabla.columns:
        raise ValueError(f"Falta la columna auxiliar {COLUMNA_PERSISTENCIA!r} en la tabla con variables.")
    # La media móvil mira solo hacia atrás y se calcula con TODAS las filas (también las que no llegan al CSV)
    tabla[COLUMNA_MEDIA_MOVIL] = calcular_baselines(tabla)["media_movil"]
    entrenamiento, validacion, prueba = dividir_por_fecha(tabla)
    columnas = seleccionar_variables(entrenamiento, entrenamiento["objetivo"])      # lista congelada: no aprende
    X_train, _ = separar_xy(entrenamiento, columnas)
    X_val, _ = separar_xy(validacion, columnas)
    X_test, _ = separar_xy(prueba, columnas)

    grupos = aprender_reglas_de_grupos(X_train, columnas)                             # grupos: solo con entrenamiento
    preprocesador = armar_pipeline(VARIANTE_POR_DEFECTO, columnas, grupos).named_steps["preprocessor"].fit(X_train)
    shapes = {"X_train": list(X_train.shape), "X_val": list(X_val.shape), "X_test": list(X_test.shape)}
    forma_tras_el_pipeline = {nombre: list(preprocesador.transform(X).shape)
                              for nombre, X in (("X_train", X_train), ("X_val", X_val), ("X_test", X_test))}

    rutas = {}
    if guardar:
        rutas["dataset"] = guardar_dataset_final(tabla, columnas, carpeta_salida, COLUMNAS_AUXILIARES)
        rutas["linaje"] = escribir_linaje(
            carpeta_salida, {**resultado["reglas"], "grupos": grupos}, columnas, shapes,
            {"columnas_auxiliares": list(COLUMNAS_AUXILIARES), "variante_pipeline": VARIANTE_POR_DEFECTO,
             "alpha_de_referencia": ALPHA_ELEGIDO, "forma_tras_el_pipeline": forma_tras_el_pipeline,
             "filas_de_frontera_quitadas": 2,
             "baselines": {"persistencia": COLUMNA_PERSISTENCIA, "media_movil": COLUMNA_MEDIA_MOVIL,
                           "ventana_dias": VENTANA_MEDIA_MOVIL, "minimo_dias_validos": MINIMO_DIAS_MEDIA_MOVIL}})
    return {"columnas": columnas, "shapes": shapes, "forma_tras_el_pipeline": forma_tras_el_pipeline,
            "rutas": rutas}


def main(argumentos=None):
    """Punto de entrada por línea de comandos: ejecuta el preprocesamiento e imprime un resumen de cada paso."""
    analizador = argparse.ArgumentParser(description="Preprocesamiento de la Fase 2: de los Excel al dataset final.")
    analizador.add_argument("--carpeta-datos", default=str(CARPETA_DATOS_POR_DEFECTO),
                            help="carpeta con los Excel (por defecto, data/ del repositorio)")
    analizador.add_argument("--sin-cache", action="store_true",
                            help="reconstruir la tabla completa desde los Excel (tarda unos 6 minutos)")
    analizador.add_argument("--sin-guardar", action="store_true",
                            help="hacer todas las comprobaciones sin escribir dataset_final.csv ni el linaje")
    opciones = analizador.parse_args(argumentos)

    resultado = ejecutar_preprocesamiento(opciones.carpeta_datos, usar_cache=not opciones.sin_cache)
    print(f"1. Tabla completa: {len(resultado['tabla_completa'])} días x {resultado['tabla_completa'].shape[1]} columnas")
    print(f"2. Limpieza aplicada con reglas de entrenamiento: {resultado['tabla_limpia'].shape[1]} columnas")
    print(f"3. Variables nuevas: {len(resultado['tabla_con_variables'])} días x {resultado['tabla_con_variables'].shape[1]} columnas")
    print(f"   Entrenamiento {resultado['entrenamiento'].shape}, validación {resultado['validacion'].shape}, prueba {resultado['prueba'].shape}")

    cierre = completar_fase2(resultado, guardar=not opciones.sin_guardar)
    print(f"4. Selección de variables: {len(cierre['columnas'])} variables (lista congelada, ADR 0015)")
    print(f"5. Pipeline ajustado solo con entrenamiento. Shapes de entrada {cierre['shapes']}")
    print(f"   Shapes tras el Pipeline {cierre['forma_tras_el_pipeline']}")
    if opciones.sin_guardar:
        print("6. Guardado omitido (--sin-guardar): no se escribió ningún archivo.")
    else:
        print(f"6. Guardado: {cierre['rutas']['dataset']} (sin escalar, con su checksum en data/checksums.json) "
              f"y {cierre['rutas']['linaje']}")
    resultado["cierre"] = cierre
    return resultado


if __name__ == "__main__":
    main()
