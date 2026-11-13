from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.impute import KNNImputer

def separar_xy(tabla, columnas_elegidas):
    """Devuelve las entradas y el objetivo de las filas con etiqueta."""
    filas_con_objetivo = tabla["objetivo"].notna()
    X = tabla.loc[filas_con_objetivo, columnas_elegidas].copy()
    y = tabla.loc[filas_con_objetivo, "objetivo"].copy()
    return X, y


def armar_pipeline(numericas, binarias):
    """Construye el preprocesador sin ajustar para la prueba con Ridge."""
    tratamiento_numerico = Pipeline([
    # Pone las columnas en escalas comparables para calcular distancias.
    ("escalar_para_distancias", StandardScaler()),
    ("imputar", KNNImputer(n_neighbors=5)),
    ("transformar", PowerTransformer(method="yeo-johnson", standardize=False)),
    ("escalar_para_ridge", StandardScaler()),
])

    return ColumnTransformer([
        ("numericas", tratamiento_numerico, numericas),
        ("binarias", "passthrough", binarias),
    ])