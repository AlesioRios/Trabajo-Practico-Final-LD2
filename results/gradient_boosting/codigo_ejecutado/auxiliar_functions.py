"""Funciones auxiliares compartidas por todas las fases del TP.

Acá viven las piezas reutilizables: semilla, huella SHA-256 de archivos,
logger de experimentos y escritor del registro de modelos. Con la Fase 1 se
sumaron dos grupos más:

- Presentación de tablas: estilizar, mostrar_lado_a_lado, tabla_simple.
- Lectura y unificación de los Excel de la planta: leer_consolidado_ee,
  cierres_por_bloque, unir_versiones, clasificar_fallas, construir_diario.
- Análisis exploratorio: medir_efecto_por_grupo (¿cambia el consumo según el día de la
  semana, el mes, el año?).

Las funciones de datos las reutilizan preprocessing_pipeline.py y predict.py:
el Excel oculto de la cátedra tiene que procesarse igual que el de entrenamiento.
"""

import pandas as pd
import numpy as np
from scipy.stats import kruskal
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
import hashlib
import json
from pathlib import Path



SEMILLA = 42

# Una lectura horaria más una de cierre a las 23:59: el acumulado del día se toma ahí.
# Se compara como texto porque pandas lee HORA como "HH:MM:SS", que se ordena igual que el tiempo.
HORA_CIERRE = "23:59:00"

# Corte provisorio: los consumos de cierre normales van de ~11.000 a ~30.000 kW y los
# errores de la planta son del orden de millones. No es la regla final de atípicos.
CONSUMO_MAXIMO_PLAUSIBLE = 100_000

# Un día normal sube unos 1.300 a 1.500 kW por hora y el 95 % de los días no pasa de 2.212 en
# ninguna hora. Un salto de más de 5.000 entre dos lecturas seguidas es más del doble de eso
# y no es creíble para este sistema (ADR 0005).
UMBRAL_SALTO = 5_000

# El totalizador se reinicia a 0 a las 00:00; hasta 100 kW de lectura inicial es ruido de medición.
UMBRAL_LECTURA_INICIAL = 100

# Tipos de falla que dejan inservible el cierre del día (ADR 0005). "Salto corregido" no está
# acá: la lectura errónea se corrigió sola antes de las 23:59 y el cierre es confiable.
FALLAS_INVALIDANTES = ("Salto no corregido", "Reinicio tardío")

# Baseline de media móvil (consigna: últimos 7 días). Los días inválidos no entran en el promedio,
# pero se exigen al menos 4 días válidos en la ventana para no predecir con muy poca información.
VENTANA_MEDIA_MOVIL = 7
MINIMO_DIAS_MEDIA_MOVIL = 4

COLUMNA_CONSUMO = "Frio (Kw)"

# Estilo común a todas las tablas: al pasar la tabla a HTML (mostrar_lado_a_lado) se
# pierde el CSS por defecto de Jupyter, y sin esto se ve más apretada y plana.
# El color del texto de los encabezados es explícito: si no, hereda el del tema del
# editor (gris claro en los temas oscuros) y queda ilegible sobre el fondo claro.
COLOR_TEXTO = '#24292f'
# !important porque los editores aplican sus propios estilos a las tablas HTML y, sin
# él, el color del tema puede ganar y dejar los encabezados casi invisibles.
ESTILOS_TABLA = [
    {'selector': 'th', 'props': [('padding', '6px 12px'), ('text-align', 'center'), ('border', '1px solid #d0d7de'), ('background-color', '#f6f8fa !important'), ('color', f'{COLOR_TEXTO} !important')]},
    {'selector': 'td', 'props': [('padding', '6px 12px'), ('text-align', 'center'), ('border', '1px solid #d0d7de')]},
    {'selector': '', 'props': [('border-collapse', 'collapse'), ('font-size', '13px')]},
]

# Las celdas con gradiente ya traen su propio color de texto (claro sobre fondo oscuro y
# al revés), y un color en 'td' lo pisaría. Las tablas sin gradiente sí lo necesitan.
ESTILOS_TABLA_SIMPLE = ESTILOS_TABLA + [
    {'selector': 'td', 'props': [('color', f'{COLOR_TEXTO} !important'), ('background-color', '#ffffff !important')]},
]

UMBRAL_CASI_CONSTANTE = 95      # % de días con el mismo valor a partir del cual una columna se considera casi constante
UMBRAL_MUCHOS_FALTANTES = 40    # % de días sin dato a partir del cual la columna no se puede imputar (menos de 60 % completo)

MINUTO_DE_CIERRE = 23 * 60 + 59     # 23:59 expresado en minutos desde la medianoche (ignora segundos)


# ---------------------------------------------------------------------------
# Presentación de tablas
# ---------------------------------------------------------------------------

def estilizar(df, cmap='Blues', precision=2, axis=0, vmin=None, vmax=None):
    """Codifica una tabla con color en vez de imprimirla como texto plano.

    Pinta las celdas con un gradiente (estilo mapa de calor). Es la función de los
    notebooks de la cátedra: mismo nombre y mismos primeros parámetros; los tres
    últimos son opcionales y con sus valores por defecto se comporta igual que ella.

    - axis=0 (por defecto): cada columna tiene su propia escala. Sirve cuando las
      columnas miden cosas distintas.
    - axis=None: una sola escala para toda la tabla, de modo que el mismo valor tenga
      siempre el mismo color. Es lo que hace falta cuando todas las columnas miden lo
      mismo (por ejemplo una matriz de presencia con ceros y unos).
    - vmin y vmax fijan los extremos de la escala (por ejemplo 0 y 1, o -3 y 3 para
      una paleta divergente centrada en cero).

    Solo sirve si todas las columnas son numéricas; para tablas con texto o fechas
    usar tabla_simple.
    """
    return (
        df.style
        .background_gradient(cmap=cmap, axis=axis, vmin=vmin, vmax=vmax)
        .format(precision=precision)
        .set_table_styles(ESTILOS_TABLA)
    )


def tabla_simple(df, precision=2):
    """Igual que estilizar pero sin gradiente, para tablas con texto, fechas u horas.

    background_gradient falla con columnas de texto, así que las tablas mixtas usan
    este mismo estilo de bordes y tipografía sin el color.
    """
    return df.style.format(precision=precision).set_table_styles(ESTILOS_TABLA_SIMPLE)


def _mostrar_tarjetas(tablas, estilo):
    """Muestra varias tablas en una fila, cada una en una tarjeta con su título.

    Cada elemento de `tablas` es (título, df) o (título, df, opciones), donde `opciones`
    es un diccionario que se le pasa a `estilo` solo para esa tabla.
    """
    # Import acá y no arriba: los scripts que corren por consola (predict.py) no
    # necesitan IPython, solo los notebooks.
    from IPython.display import HTML, display

    bloques = []
    for titulo, df, *resto in tablas:
        opciones = resto[0] if resto else {}
        bloques.append(
            "<div style='margin:0 32px 20px 0; padding:12px 14px; "
            "border:1px solid #d0d7de; border-radius:10px;'>"
            f"<div style='font-weight:600; margin-bottom:8px;'>{titulo}</div>"
            f"{estilo(df, **opciones).to_html()}</div>"
        )
    display(HTML("<div style='display:flex; flex-wrap:wrap;'>" + ''.join(bloques) + "</div>"))


def mostrar_lado_a_lado(tablas, cmap='Blues', precision=2):
    """Muestra varias tablas numéricas en una fila horizontal, en vez de apiladas.

    `tablas` es una lista de pares (título, DataFrame). Cada una queda dentro de una
    tarjeta con borde propio para que se vea prolija y entre en pantalla. Un tercer
    elemento opcional por tabla, un diccionario como {"axis": None, "vmin": 0}, cambia
    el estilo solo de esa tabla (ver estilizar). Con {"sin_color": True} esa tabla se
    muestra sin gradiente, útil para que entre en la misma fila que otras con color.

    Criterio para usar color: solo si todas las columnas miden lo mismo (misma unidad).
    Con unidades mezcladas o diferencias mínimas entre valores, el gradiente estira la
    escala y hace que cifras casi iguales se vean muy distintas.
    """
    def con_color(df, **opciones):
        if opciones.pop("sin_color", False):
            return tabla_simple(df, precision=opciones.get("precision", precision))
        return estilizar(df, **{"cmap": cmap, "precision": precision, **opciones})

    _mostrar_tarjetas(tablas, con_color)


def mostrar_lado_a_lado_simple(tablas, precision=2):
    """Lo mismo que mostrar_lado_a_lado, pero sin gradiente (acepta texto y fechas)."""
    def sin_color(df, **opciones):
        return tabla_simple(df, **{"precision": precision, **opciones})

    _mostrar_tarjetas(tablas, sin_color)


def formatear_fechas(df):
    """Devuelve una copia con las fechas como texto AAAA-MM-DD, sin la hora 00:00:00.

    Se aplica a las columnas de fecha y también al índice si es de fechas. Es solo
    para mostrar: el DataFrame original no se modifica.
    """
    df = df.copy()
    for columna in df.select_dtypes(include="datetime").columns:
        df[columna] = df[columna].dt.strftime("%Y-%m-%d")
    if isinstance(df.index, pd.DatetimeIndex):
        nombre = df.index.name
        df.index = df.index.strftime("%Y-%m-%d")
        df.index.name = nombre
    return df


# ---------------------------------------------------------------------------
# Lectura y unificación de los Excel de la planta
# ---------------------------------------------------------------------------

def leer_consolidado_ee(fuente):
    """Lee la hoja Consolidado EE de un Excel, que contiene el objetivo `Frio (Kw)`.

    La hoja se busca por nombre y no por posición: los tres Excel no tienen la misma
    cantidad de hojas. `fuente` es una ruta o un pd.ExcelFile ya abierto (así un mismo
    libro se puede reutilizar para leer otra hoja sin abrirlo de nuevo).

    Devuelve (tabla, filas_sin_fecha): la tabla sin las filas vacías del final de la
    hoja y con DIA como fecha, y cuántas filas vacías se descartaron.
    """
    tabla = pd.read_excel(fuente, sheet_name="Consolidado EE")
    filas_sin_fecha = int(tabla["DIA"].isna().sum())
    tabla = tabla.dropna(subset=["DIA"])
    tabla["DIA"] = pd.to_datetime(tabla["DIA"])
    return tabla, filas_sin_fecha


def resumir_por_fecha(tabla):
    """Cuántas filas tiene cada fecha y a qué hora termina."""
    por_fecha = tabla.groupby("DIA")["HORA"]
    return pd.DataFrame({"filas": por_fecha.size(), "ultima_hora": por_fecha.max()})


def numerar_bloques(tabla):
    """Número de bloque diario de cada fila (0, 1, 2...), contado en todo el archivo.

    Un bloque es un día completo de lecturas (00:00 a 23:59). Empieza uno nuevo cada
    vez que la hora retrocede respecto de la fila anterior. Se usa la hora y no la
    cantidad de filas (25) porque hay días con lecturas de menos o de más. Supone que
    la tabla conserva el orden original del archivo.
    """
    hora_retrocede = tabla["HORA"] < tabla["HORA"].shift(fill_value="")
    return hora_retrocede.cumsum()


def cierres_por_bloque(tabla, archivo):
    """Una fila por bloque diario: su última lectura (el cierre) con todas las columnas.

    Agrega: `bloque` (orden dentro de su fecha rotulada; 0 si la fecha no se repite),
    `filas` (cuántas lecturas tiene el bloque), `archivo` (de dónde viene), `completo`
    (llega a las 23:59) y `plausible` (consumo menor al corte provisorio).

    También agrega tres medidas de cómo evolucionó el contador dentro del día (ADR 0005):
    `lectura_inicial` (la primera lectura, que debería ser casi 0 porque el totalizador se
    reinicia a medianoche), `mayor_incremento` y `menor_incremento` (el mayor y el menor
    cambio entre dos lecturas seguidas; este último es negativo si el contador bajó).
    """
    numero_de_bloque = numerar_bloques(tabla)
    por_bloque = tabla.groupby(numero_de_bloque)
    cierres = por_bloque.tail(1).copy()
    cierres["lectura_inicial"] = por_bloque[COLUMNA_CONSUMO].head(1).to_numpy()
    incrementos = por_bloque[COLUMNA_CONSUMO].diff().groupby(numero_de_bloque)
    cierres["mayor_incremento"] = incrementos.max().to_numpy()
    cierres["menor_incremento"] = incrementos.min().to_numpy()
    cierres["filas"] = tabla.groupby(numero_de_bloque).size().to_numpy()
    cierres["bloque"] = cierres.groupby("DIA").cumcount()
    cierres["archivo"] = archivo
    cierres["completo"] = cierres["HORA"] >= HORA_CIERRE
    cierres["plausible"] = cierres[COLUMNA_CONSUMO] <= CONSUMO_MAXIMO_PLAUSIBLE
    return cierres.reset_index(drop=True)


def unir_versiones(cierres):
    """Una sola fila por bloque diario a partir de los cierres de varios Excel.

    Un bloque puede estar en dos archivos (se solapan en el tiempo). Se elige la
    versión completa y plausible y, si hay empate, la del archivo más reciente: en
    las 5 discrepancias que había entre el primer y el segundo Excel, el posterior
    tenía el valor correcto. Devuelve los bloques en el orden original de los datos.

    `cierres` es la unión (pd.concat) de los resultados de cierres_por_bloque.
    """
    return (
        cierres.sort_values(["completo", "plausible", "archivo"], ascending=False)
        .drop_duplicates(["DIA", "bloque"], keep="first")
        .sort_values(["DIA", "bloque"])
        .reset_index(drop=True)
    )


def clasificar_fallas(diario):
    """Tipo de falla del contador de cada día, según cómo evolucionó dentro de la jornada.

    Necesita las columnas `mayor_incremento` y `menor_incremento` (ver cierres_por_bloque).
    Un "gran salto" es un incremento de más de UMBRAL_SALTO entre dos lecturas seguidas. Se
    distingue por la suma del mayor y el menor incremento (`neto`): si la lectura errónea
    se corrigió sola, la subida y la bajada se compensan y la suma queda chica.

    - "Salto corregido": gran salto y neto menor a UMBRAL_SALTO. Un pico de una lectura;
      el cierre del día es confiable.
    - "Salto no corregido": gran salto y neto de UMBRAL_SALTO o más. El contador saltó y se
      quedó arriba, así que el cierre queda inflado.
    - "Reinicio tardío": sin gran salto, pero el contador bajó más de UMBRAL_SALTO. El
      totalizador se reinició horas después de la medianoche.
    - "Sin falla grave": el resto.
    """
    neto = diario["mayor_incremento"] + diario["menor_incremento"]
    gran_salto = diario["mayor_incremento"] > UMBRAL_SALTO
    tipo = pd.Series("Sin falla grave", index=diario.index)
    tipo.loc[gran_salto & (neto < UMBRAL_SALTO)] = "Salto corregido"
    tipo.loc[gran_salto & (neto >= UMBRAL_SALTO)] = "Salto no corregido"
    tipo.loc[~gran_salto & (diario["menor_incremento"] < -UMBRAL_SALTO)] = "Reinicio tardío"
    return tipo


def construir_diario(mejores):
    """Agrega la marca de día válido y el objetivo: el consumo del día siguiente.

    - `tipo_falla`: ver clasificar_fallas.
    - `valido`: el día llega a las 23:59, su consumo es plausible, el contador arranca en
      0 y no tiene una falla que invalide el cierre (ADR 0004 y 0005).
    - `consumo_dia`: el consumo de cierre, vacío en los días inválidos (no se imputa).
    - `objetivo`: el consumo del bloque siguiente (`shift(-1)`). Queda vacío si el día
      siguiente es inválido o si no hay día siguiente.

    "Día siguiente" significa el bloque que sigue en el orden del archivo, no la fecha
    siguiente del calendario, porque las etiquetas de fecha tienen desfasajes.
    """
    diario = mejores.copy()
    diario["tipo_falla"] = clasificar_fallas(diario)
    falla_grave = diario["tipo_falla"].isin(FALLAS_INVALIDANTES) | (diario["lectura_inicial"] > UMBRAL_LECTURA_INICIAL)
    diario["valido"] = diario["completo"] & diario["plausible"] & ~falla_grave
    diario["consumo_dia"] = diario[COLUMNA_CONSUMO].where(diario["valido"])
    diario["objetivo"] = diario["consumo_dia"].shift(-1)
    return diario


# ---------------------------------------------------------------------------
# Análisis exploratorio
# ---------------------------------------------------------------------------

def medir_efecto_por_grupo(tabla, columna_grupo, columna_valor, orden=None):
    """Mide cuánto cambia `columna_valor` entre los grupos de `columna_grupo`.

    Responde preguntas como "¿importa el día de la semana (o el mes, o el año)?". Los
    valores vacíos de `columna_valor` se ignoran. `orden` es la lista de grupos en el
    orden en que se quieren las filas del resumen (por defecto, el alfabético).

    Devuelve dos tablas:
    - resumen: una fila por grupo con `Días` (cuántos valores), `Promedio` y `Mediana`.
    - medidas: cuatro números en la columna `Valor`: la diferencia entre el grupo de
      promedio más alto y el de más bajo (en unidades de la columna y como % del promedio
      general), el % de la variación total que explican los grupos (eta cuadrado) y el
      p-valor de la prueba de Kruskal-Wallis (hipótesis nula: todos los grupos tienen la
      misma distribución).

    Esa prueba supone observaciones independientes; en una serie temporal los días
    consecutivos se parecen, así que el p-valor puede salir más chico de lo que
    correspondería. Hay que leerlo junto con el % explicado, no solo.
    """
    tabla = tabla.dropna(subset=[columna_valor])
    por_grupo = tabla.groupby(columna_grupo)[columna_valor]

    resumen = por_grupo.agg(["count", "mean", "median"])
    if orden is not None:
        resumen = resumen.reindex(orden)
    resumen.columns = ["Días", "Promedio", "Mediana"]
    resumen.index.name = columna_grupo

    promedio_general = tabla[columna_valor].mean()
    promedio_del_grupo = por_grupo.transform("mean")
    variacion_entre_grupos = ((promedio_del_grupo - promedio_general) ** 2).sum()
    variacion_total = ((tabla[columna_valor] - promedio_general) ** 2).sum()

    diferencia = resumen["Promedio"].max() - resumen["Promedio"].min()
    _, p_valor = kruskal(*[valores.to_numpy() for _, valores in por_grupo])

    medidas = pd.DataFrame({"Valor": {
        "Diferencia entre el grupo más alto y el más bajo": diferencia,
        "Esa diferencia, como % del promedio general": 100 * diferencia / promedio_general,
        "% de la variación total explicado por los grupos": 100 * variacion_entre_grupos / variacion_total,
        "p-valor de la prueba de Kruskal-Wallis": p_valor,
    }})
    return resumen, medidas


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def calcular_baselines(diario, ventana=VENTANA_MEDIA_MOVIL, minimo_dias=MINIMO_DIAS_MEDIA_MOVIL):
    """Predicción de los dos baselines para el consumo del día siguiente, una por fila de `diario`.

    La fila de hoy predice el consumo de mañana, que es su `objetivo`:
    - `persistencia`: el consumo de hoy (`consumo_dia`).
    - `media_movil`: el promedio del consumo de los últimos `ventana` días, hoy incluido. Los
      días inválidos (consumo vacío) no entran en el promedio, pero hacen falta al menos
      `minimo_dias` días válidos en la ventana.

    La ventana se cuenta en filas y no en fechas del calendario (ADR 0002). Donde no se puede
    predecir queda vacío: no se imputa (ADR 0004).
    """
    consumo = diario["consumo_dia"]
    return pd.DataFrame({
        "persistencia": consumo,
        "media_movil": consumo.rolling(ventana, min_periods=minimo_dias).mean(),
    })


def evaluar_baselines(diario, ventana=VENTANA_MEDIA_MOVIL, minimo_dias=MINIMO_DIAS_MEDIA_MOVIL):
    """MAE (error absoluto medio) de cada baseline, calculado sobre exactamente las mismas filas.

    Se evalúan solo las filas que tienen objetivo (hay un "mañana" válido con qué comparar) y
    las dos predicciones, para que los baselines se comparen entre sí sobre los mismos días.
    Las demás filas no se evalúan ni se completan.

    Devuelve (resumen, evaluadas). `resumen` tiene una fila por baseline con su MAE y la cantidad
    de filas evaluadas. `evaluadas` es la tabla de esas filas con el objetivo, las predicciones y
    el error absoluto de cada una.
    """
    baselines = calcular_baselines(diario, ventana, minimo_dias)
    tabla = baselines.assign(objetivo=diario["objetivo"])
    evaluadas = tabla.loc[tabla.notna().all(axis=1)].copy()
    for nombre in ("persistencia", "media_movil"):
        evaluadas[f"error_{nombre}"] = (evaluadas["objetivo"] - evaluadas[nombre]).abs()

    resumen = pd.DataFrame({
        "MAE (kW)": {
            "Consumo de hoy": evaluadas["error_persistencia"].mean(),
            f"Media móvil de {ventana} días": evaluadas["error_media_movil"].mean(),
        },
        "Filas evaluadas": len(evaluadas),
    })
    return resumen, evaluadas


def _claves_dia_hora(tabla):
    """Las columnas DIA y HORA de una hoja, sin las filas vacías del final y con HORA como texto."""
    claves = tabla.loc[tabla["DIA"].notna() & tabla["HORA"].notna(), ["DIA", "HORA"]].reset_index(drop=True)
    claves["HORA"] = claves["HORA"].astype(str)
    return claves


def medir_alineacion(hoja, referencia):
    """Compara fila por fila la fecha y la hora de una hoja con las de una hoja de referencia.

    Sirve para saber si dos hojas del mismo Excel se pueden pegar por posición. Una fila
    "coincide" si, en el mismo número de fila, las dos hojas tienen la misma DIA y la misma HORA.
    Si a una hoja le falta una fila, todas las siguientes quedan corridas y dejan de coincidir.

    Devuelve un diccionario con las filas de cada hoja, cuántas coinciden, el porcentaje sobre la
    referencia y el número de la primera fila que no coincide (None si coinciden todas).
    """
    claves_hoja = _claves_dia_hora(hoja)
    claves_ref = _claves_dia_hora(referencia)
    comunes = min(len(claves_hoja), len(claves_ref))
    coincide = (claves_hoja.iloc[:comunes] == claves_ref.iloc[:comunes]).all(axis=1)

    if not coincide.all():
        primera_distinta = int((~coincide).idxmax())
    elif len(claves_hoja) != len(claves_ref):
        primera_distinta = comunes          # coinciden hasta donde llega la más corta
    else:
        primera_distinta = None

    return {
        "Filas de la hoja": len(claves_hoja),
        "Filas de la referencia": len(claves_ref),
        "Filas que coinciden": int(coincide.sum()),
        "% que coincide": 100 * coincide.sum() / len(claves_ref),
        "Primera fila distinta": primera_distinta,
    }

def contexto_desfase(hojas, referencia, fila, ancho=2):
    """Muestra la fecha y hora de las filas que rodean a `fila`, con una columna por hoja.

    `hojas` es un diccionario {nombre de la hoja: tabla}; `referencia` es la hoja del objetivo.
    Así se ve si las hojas que fallan en la misma fila fallan de la misma manera.
    """
    claves_ref = _claves_dia_hora(referencia)
    claves = {nombre: _claves_dia_hora(tabla) for nombre, tabla in hojas.items()}
    posiciones = range(max(fila - ancho, 0), fila + ancho + 1)   # `ancho` filas antes y después

    def texto(tabla_claves, i):
        if i >= len(tabla_claves):            # una hoja puede terminar antes que la otra
            return "—"
        return f"{tabla_claves.loc[i, 'DIA']:%d/%m/%y} {tabla_claves.loc[i, 'HORA']}"

    columnas = {"Fila": list(posiciones),
                "Consolidado EE": [texto(claves_ref, i) for i in posiciones]}
    for nombre, tabla_claves in claves.items():   # una columna más por cada hoja del grupo
        columnas[nombre] = [texto(tabla_claves, i) for i in posiciones]
    return pd.DataFrame(columnas)

def numerar_bloques_por_caida(tabla, caida_minima_horas=12):
    """Número de bloque diario de cada fila, para hojas con horas mal escritas.

    Es una variante de `numerar_bloques`: un bloque nuevo empieza cuando la hora cae
    `caida_minima_horas` o más respecto de la fila anterior. Un cierre (23:59) seguido de
    00:00 cae casi 24 horas; un tipeo (por ejemplo 23:00 en lugar de 11:00) cae menos.
    Acepta la hora como texto ("23:59:00") o como fecha y hora ("1900-01-01 00:00:00").
    Supone que la tabla conserva el orden del archivo y no tiene filas vacías.
    """
    # Saco horas y minutos del final del texto; así sirve el formato "1900-01-01 00:00:00" también
    partes = tabla["HORA"].astype(str).str.extract(r"(\d{1,2}):(\d{2})(?::\d{2})?$").astype(float)
    hora = partes[0] + partes[1] / 60             # la hora como número: 23:30 -> 23.5
    cae = (hora.shift() - hora) >= caida_minima_horas   # cuánto bajó respecto de la fila anterior
    return cae.cumsum()                           # cada caída grande suma un bloque


def fecha_de_cada_bloque(tabla):
    """La fecha de cada bloque diario de una hoja, en el orden de los bloques.

    Usa `numerar_bloques_por_caida` y, dentro de cada bloque, la fecha que más se repite, así una
    fila con la fecha mal puesta (como la lectura de las 23:59 con el día siguiente) no cambia el resultado.
    Supone una tabla sin filas vacías y con DIA como fecha.
    """
    bloques = numerar_bloques_por_caida(tabla)    # a qué bloque pertenece cada fila
    # mode() devuelve la fecha más frecuente del grupo; iloc[0] toma la primera si hay empate
    return tabla.groupby(bloques)["DIA"].agg(lambda fechas: fechas.mode().iloc[0])

def cierres_de_hoja(tabla):
    """La última lectura de cada día (bloque) de una hoja, sin las columnas de fecha y hora.

    Para un totalizador que se reinicia a medianoche, esa lectura es el consumo del día.
    """
    tabla = tabla.dropna(subset=["DIA", "HORA"]).reset_index(drop=True)   # fuera las filas vacías del final
    bloques = numerar_bloques_por_caida(tabla)                             # a qué día pertenece cada fila
    # tail(1) deja la última fila de cada bloque; la fecha y la hora ya no hacen falta
    return tabla.groupby(bloques).tail(1).drop(columns=["DIA", "HORA"])


def perfilar_columnas(cierres):
    """Una fila por columna: faltantes, texto, valor más repetido y variabilidad relativa a su nivel."""
    perfil = []
    for columna in cierres.columns:
        valores = pd.to_numeric(cierres[columna], errors="coerce")          # lo que no es número queda vacío
        con_texto = int((cierres[columna].notna() & valores.isna()).sum())   # había algo, pero no era un número
        media, mediana = valores.mean(), valores.median()
        rango_central = valores.quantile(0.75) - valores.quantile(0.25)      # distancia entre los percentiles 25 y 75
        perfil.append({
            "Columna": str(columna).strip(),
            "% faltante": 100 * valores.isna().mean(),                       # el texto cuenta como sin dato
            "Celdas con texto": con_texto,
            "% del valor más repetido": (100 * valores.value_counts(normalize=True).iloc[0]
                                         if valores.notna().any() else None),     # columna vacía: no aplica
            # Desvío dividido por el nivel: no tiene unidades, así se compara entre columnas (si el nivel es 0, no aplica)
            "Coef. de variación": valores.std() / abs(media) if media != 0 else None,
            # Lo mismo con medidas robustas a picos: rango central dividido por la mediana
            "Dispersión robusta": rango_central / abs(mediana) if mediana != 0 else None,
        })
    return pd.DataFrame(perfil)



def resumir_perfil(perfil):
    """Cuenta, para una hoja, cuántas columnas son casi constantes, tienen muchos faltantes o tienen texto."""
    return pd.Series({
        "Columnas": len(perfil),
        "Casi constantes": int((perfil["% del valor más repetido"] >= UMBRAL_CASI_CONSTANTE).sum()),
        "Con muchos faltantes": int((perfil["% faltante"] > UMBRAL_MUCHOS_FALTANTES).sum()),
        "Con texto": int((perfil["Celdas con texto"] > 0).sum()),
    })


def convertir_texto_a_numero(valor):
    """Convierte una celda a número; devuelve None si es vacía o no se puede interpretar."""
    if isinstance(valor, (int, float)):            # ya es número (o NaN): se deja como está
        return valor
    texto = str(valor).strip()                     # se sacan espacios de los bordes
    if texto == "":                                # vacío o solo espacios -> faltante
        return None
    if "," in texto:                               # '1.234,5': puntos de miles, coma decimal
        texto = texto.replace(".", "").replace(",", ".")
    elif texto.count(".") >= 2:                    # '1.016.475': el último punto es el decimal
        entero, _, decimales = texto.rpartition(".")
        texto = entero.replace(".", "") + "." + decimales
    try:
        return float(texto)
    except ValueError:                             # '#N/A' y similares -> faltante
        return None

def convertir_textos_a_numero(tabla, columnas_a_excluir=("DIA", "HORA")):
    """Devuelve una copia de la tabla con las columnas de texto numérico convertidas a número.

    Una columna que sea en realidad de etiquetas (casi nada se puede leer como número) se deja igual.
    """
    limpia = tabla.copy()                                    # no se modifica la tabla original
    for columna in limpia.columns:
        if columna in columnas_a_excluir or pd.api.types.is_numeric_dtype(limpia[columna]):
            continue        # solo las que no son numéricas: texto mezclado con números (object) o texto puro (str en pandas 3)
        convertida = pd.to_numeric(limpia[columna].map(convertir_texto_a_numero))
        # Celdas con contenido real: las vacías y las de solo espacios no cuentan
        con_contenido = limpia[columna].map(lambda v: pd.notna(v) and str(v).strip() != "").sum()
        # Si casi todo lo que tenía contenido se pudo leer como número, la columna era numérica
        if convertida.notna().sum() >= 0.9 * con_contenido:
            limpia[columna] = convertida
    return limpia


def cierres_de_otras_hojas(hojas, hoja_objetivo="Consolidado EE"):
    """La lectura de cierre de cada día de todas las hojas menos la del objetivo, pegadas por bloque.

    `hojas` es un diccionario {nombre de la hoja: tabla} de un mismo Excel. Devuelve una fila por
    día (bloque, en el orden del archivo) y una columna por cada columna de cada hoja, con el
    nombre "Hoja | Columna". Aplica el ADR 0006: textos convertidos a número, `Id` fuera, y
    si una hoja no cierra a las 23:59 o después ese día queda vacío (no se imputa).
    """
    cierres_por_hoja = []
    for nombre, tabla in hojas.items():
        if nombre == hoja_objetivo:                       # el objetivo se trata aparte
            continue
        tabla = tabla.dropna(subset=["DIA", "HORA"]).reset_index(drop=True)
        bloques = numerar_bloques_por_caida(tabla)        # a qué día pertenece cada fila
        cierres = convertir_textos_a_numero(tabla).groupby(bloques).tail(1)   # última lectura del día
        # Hora y minuto del cierre; sirve "23:59:00" y "1900-01-01 23:59:00"
        hora = cierres["HORA"].astype(str).str.extract(r"(\d{1,2}):(\d{2})(?::\d{2})?$").astype(float)
        cierra_a_tiempo = (hora[0] * 60 + hora[1]) >= MINUTO_DE_CIERRE
        valores = cierres.drop(columns=["DIA", "HORA"])
        valores = valores.loc[:, [c for c in valores.columns if str(c).strip() != "Id"]]   # Id no es medición
        valores = valores.where(cierra_a_tiempo, axis=0)  # si no cerró a tiempo, toda la fila queda vacía
        valores.index = range(len(valores))               # el número de bloque es la clave de unión
        valores.columns = [f"{nombre} | {str(c).strip()}" for c in valores.columns]
        cierres_por_hoja.append(valores)
    return pd.concat(cierres_por_hoja, axis=1)

def marcar_columnas_descartables(perfil):
    """Agrega al perfil de columnas las marcas del ADR 0006 y si la columna se conserva.

    `perfil` es el resultado de `perfilar_columnas` calculado solo sobre los datos de
    entrenamiento. Una columna se descarta si tiene más de UMBRAL_MUCHOS_FALTANTES % de
    faltantes o si repite un mismo valor en UMBRAL_CASI_CONSTANTE % o más de los días.
    """
    marcado = perfil.copy()
    marcado["Muchos faltantes"] = marcado["% faltante"] > UMBRAL_MUCHOS_FALTANTES
    # Una columna totalmente vacía no tiene valor más repetido: se cuenta como casi constante
    marcado["Casi constante"] = marcado["% del valor más repetido"].fillna(100) >= UMBRAL_CASI_CONSTANTE
    marcado["Se conserva"] = ~marcado["Muchos faltantes"] & ~marcado["Casi constante"]
    return marcado

def correlaciones_con_objetivo(tabla, columnas, columna_objetivo="objetivo"):
    """Correlación de cada columna con el objetivo, con Spearman y con Pearson.

    Spearman usa el orden de los valores, así que casi no lo afectan los valores extremos;
    Pearson mide cuánto se parece la relación a una recta y sí lo afectan. Devuelve una fila
    por columna, ordenada de mayor a menor Spearman en valor absoluto.
    """
    numeros = tabla[columnas].apply(pd.to_numeric, errors="coerce")   # cualquier texto queda vacío
    resultado = pd.DataFrame({
        "Spearman": numeros.corrwith(tabla[columna_objetivo], method="spearman"),
        "Pearson": numeros.corrwith(tabla[columna_objetivo], method="pearson"),
    })
    resultado["Spearman absoluto"] = resultado["Spearman"].abs()      # para ordenar sin importar el signo
    return resultado.sort_values("Spearman absoluto", ascending=False)

def parejas_muy_correlacionadas(tabla, columnas, umbral=0.9):
    """Parejas de columnas cuya correlación de Spearman, en valor absoluto, llega a `umbral`.

    Cada pareja aparece una sola vez (no se repite como A-B y B-A ni se compara una columna
    consigo misma). Las parejas que no se pueden calcular (columnas sin días en común) se omiten.
    Devuelve `Columna A`, `Columna B`, `Spearman` y `Pearson` (para contrastar), de mayor a menor Spearman.
    """
    numeros = tabla[columnas].apply(pd.to_numeric, errors="coerce")   # cualquier texto queda vacío
    spearman = numeros.corr(method="spearman")
    pearson = numeros.corr(method="pearson")
    # Me quedo con el triángulo de arriba de la diagonal: ahí está cada pareja una sola vez
    solo_arriba = spearman.where(np.triu(np.ones(spearman.shape, dtype=bool), k=1))
    parejas = (solo_arriba.stack().dropna()                      # una fila por pareja calculable
               .rename_axis(["Columna A", "Columna B"]).rename("Spearman").reset_index())
    parejas = parejas[parejas["Spearman"].abs() >= umbral].copy()
    # Para cada pareja que quedó, busco su Pearson en la otra matriz
    parejas["Pearson"] = [pearson.loc[a, b] for a, b in zip(parejas["Columna A"], parejas["Columna B"])]
    return parejas.reindex(parejas["Spearman"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def ordenar_por_similitud(matriz):
    """Reordena una matriz de correlación para que las columnas parecidas queden juntas.

    Agrupa las columnas por jerarquía usando como distancia 1 menos el valor absoluto de la
    correlación (dos columnas muy correlacionadas, con signo positivo o negativo, quedan cerca).
    Una correlación que no se pudo calcular se trata como 0. Devuelve la misma matriz, con filas
    y columnas en el nuevo orden.
    """
    distancia = 1 - matriz.abs().fillna(0).to_numpy()          # 0 = idénticas, 1 = sin relación
    distancia = (distancia + distancia.T) / 2                  # simétrica, por si hay diferencias mínimas
    np.fill_diagonal(distancia, 0)                             # una columna está a distancia 0 de sí misma
    orden = leaves_list(linkage(squareform(distancia, checks=False), method="average", optimal_ordering=True))
    return matriz.iloc[orden, orden]


def etiqueta_corta(columna):
    """Nombre corto de una columna para ejes de gráficos: 'Hoja abreviada: columna' (EE si no tiene hoja)."""
    hoja, _, base = str(columna).rpartition(" | ")
    hoja = hoja.replace("Consolidado ", "C. ").replace("Totalizadores ", "T. ")
    return f"{hoja or 'EE'}: {base}"[:34]

def elegir_columnas_distintas(ranking, matriz, cantidad=5, umbral=0.98):
    """Las primeras `cantidad` columnas del ranking, saltando las que son copia de una ya elegida.

    `ranking` es la lista de columnas de mejor a peor; `matriz` es la matriz de correlación entre
    columnas (con las mismas columnas como filas y columnas). Una columna se descarta si su
    correlación, en valor absoluto, con alguna ya elegida llega a `umbral`. Si una pareja no se
    pudo calcular (falta de días en común) no se considera copia.
    """
    elegidas = []
    for columna in ranking:
        # `not abs(x) >= umbral` en lugar de `abs(x) < umbral`: si x no se pudo calcular (vacío), cuenta como distinta
        if all(not abs(matriz.loc[columna, elegida]) >= umbral for elegida in elegidas):
            elegidas.append(columna)
        if len(elegidas) == cantidad:
            break
    return elegidas


def huella(ruta, algoritmo="sha256", bloque=1 << 20):
    """Resumen hexadecimal de un archivo, leyéndolo de a bloques (por defecto 1 MiB).

    Es la función de la cátedra (Unidad 5, capítulo 10). Leer por bloques evita cargar el
    archivo entero en memoria. Si el archivo cambia aunque sea en un byte, el resumen cambia.
    """
    resumen = hashlib.new(algoritmo)
    with open(ruta, "rb") as archivo:                            # "rb": leer bytes, sin interpretar el texto
        for trozo in iter(lambda: archivo.read(bloque), b""):    # hasta que no quede nada por leer
            resumen.update(trozo)
    return resumen.hexdigest()


def guardar_huellas(archivos, destino, base):
    """Calcula la huella de cada archivo y la guarda en un JSON {ruta relativa a `base`: huella}.

    Si `destino` ya existe, conserva las huellas que tenía (así la Fase 2 puede sumar la del
    dataset final sin borrar las de los Excel). Escribe siempre con saltos de línea "\\n" para
    que el JSON sea idéntico en Windows y Linux. Devuelve el diccionario completo.
    """
    huellas = {Path(a).relative_to(base).as_posix(): huella(a) for a in archivos}
    destino = Path(destino)
    guardadas = json.loads(destino.read_text(encoding="utf-8")) if destino.exists() else {}
    guardadas.update(huellas)                                    # las nuevas pisan a las viejas del mismo archivo
    with open(destino, "w", encoding="utf-8", newline="\n") as salida:
        json.dump(guardadas, salida, indent=2, sort_keys=True, ensure_ascii=False)
        salida.write("\n")
    return guardadas


def verificar_huellas(ruta_json, carpeta):
    """Compara las huellas guardadas con las de los archivos que hay hoy en `carpeta`.

    Devuelve {archivo: "coincide" | "distinto" | "falta"}. Si algún valor no es "coincide",
    los resultados no son comparables con los del equipo.
    """
    esperadas = json.loads(Path(ruta_json).read_text(encoding="utf-8"))
    estado = {}
    for nombre, esperada in esperadas.items():
        archivo = Path(carpeta) / nombre
        if not archivo.exists():
            estado[nombre] = "falta"
        else:
            estado[nombre] = "coincide" if huella(archivo) == esperada else "distinto"
    return estado

def correlaciones_sin_estacionalidad(tabla, columnas, columna_objetivo="objetivo", columna_fecha="DIA"):
    """Correlación de Spearman de cada columna con el objetivo, después de quitarle a ambos el efecto de la época del año.

    El efecto estacional se ajusta con el seno y el coseno de la posición del día en el año (una curva
    que sube y baja una vez por año). Se trabaja con los rangos de los valores (eso es Spearman) y se
    usan solo las filas donde la columna y el objetivo tienen dato. Devuelve una serie con una
    correlación por columna: cuánto se parecen lo que sobra de la columna y lo que sobra del objetivo
    una vez descontada la estación.
    """
    angulo = 2 * np.pi * pd.to_datetime(tabla[columna_fecha]).dt.dayofyear.to_numpy() / 365.25   # de 0 a 2π en el año
    estacion = np.column_stack([np.ones(len(tabla)), np.sin(angulo), np.cos(angulo)])
    resultado = {}
    for columna in columnas:
        valores = np.column_stack([pd.to_numeric(tabla[columna], errors="coerce").to_numpy(),
                                   tabla[columna_objetivo].to_numpy()])
        completas = ~np.isnan(valores).any(axis=1)               # filas donde hay dato en las dos
        base = estacion[completas]
        rangos = pd.DataFrame(valores[completas]).rank().to_numpy()
        # Lo que sobra de cada uno una vez ajustada la curva estacional
        residuos = rangos - base @ np.linalg.lstsq(base, rangos, rcond=None)[0]
        resultado[columna] = np.corrcoef(residuos[:, 0], residuos[:, 1])[0, 1]
    return pd.Series(resultado, name="Spearman sin estacionalidad")
