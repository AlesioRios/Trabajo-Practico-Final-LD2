"""Funciones auxiliares compartidas por todas las fases del TP.

Acá viven las piezas reutilizables: semilla, huella SHA-256 de archivos,
logger de experimentos y escritor del registro de modelos. Con la Fase 1 se
sumaron dos grupos más:

- Presentación de tablas: estilizar, mostrar_lado_a_lado, tabla_simple.
- Lectura y unificación de los Excel de la planta: leer_consolidado_ee,
  cierres_por_bloque, unir_versiones, construir_diario.

Las funciones de datos las reutilizan preprocessing_pipeline.py y predict.py:
el Excel oculto de la cátedra tiene que procesarse igual que el de entrenamiento.
"""

import pandas as pd

SEMILLA = 42

# Una lectura horaria más una de cierre a las 23:59: el acumulado del día se toma ahí.
# Se compara como texto porque pandas lee HORA como "HH:MM:SS", que se ordena igual que el tiempo.
HORA_CIERRE = "23:59:00"

# Corte provisorio: los consumos de cierre normales van de ~11.000 a ~30.000 kW y los
# errores de la planta son del orden de millones. No es la regla final de atípicos.
CONSUMO_MAXIMO_PLAUSIBLE = 100_000

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
    """
    numero_de_bloque = numerar_bloques(tabla)
    cierres = tabla.groupby(numero_de_bloque).tail(1).copy()
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


def construir_diario(mejores):
    """Agrega la marca de día válido y el objetivo: el consumo del día siguiente.

    - `valido`: el día llega a las 23:59 y su consumo es plausible.
    - `consumo_dia`: el consumo de cierre, vacío en los días inválidos (no se imputa).
    - `objetivo`: el consumo del bloque siguiente (`shift(-1)`). Queda vacío si el día
      siguiente es inválido o si no hay día siguiente.

    "Día siguiente" significa el bloque que sigue en el orden del archivo, no la fecha
    siguiente del calendario, porque las etiquetas de fecha tienen desfasajes.
    """
    diario = mejores.copy()
    diario["valido"] = diario["completo"] & diario["plausible"]
    diario["consumo_dia"] = diario[COLUMNA_CONSUMO].where(diario["valido"])
    diario["objetivo"] = diario["consumo_dia"].shift(-1)
    return diario
