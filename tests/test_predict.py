"""Pruebas de src/predict.py con un Excel sintético chico, un modelo ajustado en la prueba y un registro temporal."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import predict
from preprocessing_pipeline import ErrorDeEntrada, resolver_dependencias

AGUA = "Consolidado Agua | Agua Planta (Hl)"
TRAFO = "Totalizadores Energia | KW Trafo 9"
DIAS = 40

# Las 4 variables base de las interacciones, y una quinta que NO está en el Excel de prueba (y no debe hacer falta).
BASES = ["Frio (Kw)", "dia_anio_coseno", "frio_hace_2_dias", AGUA, "Consolidado KPI | ET Bodega/Hl"]
COLUMNAS_MODELO = ["Frio (Kw)", "frio_hace_2_dias", "Frio (Kw) x " + AGUA, TRAFO, "Servicios (Kw)"]
MEDICIONES = ["Frio (Kw)", "Servicios (Kw)", AGUA, TRAFO, "Consolidado KPI | ET Bodega/Hl"]


def reglas_sinteticas():
    return {"limpieza": {"columnas_conservadas": MEDICIONES, "columnas_de_energia": MEDICIONES,
                         "columnas_de_temperatura": [], "limite_superior": {c: 1e9 for c in MEDICIONES}},
            "variables": {"planta": {"umbral_denominador": {}},
                          "interacciones": {"variables_base": BASES, "umbral_correlacion": 0.8}}}


def hoja(columnas, totales, dias=DIAS):
    """Una hoja con 24 lecturas por día (00:00 a 23:00 y cierre 23:59) de un contador que arranca el día en 0."""
    horas = [f"{h:02d}:00:00" for h in range(23)] + ["23:59:00"]
    filas = []
    for d in range(dias):
        for h, hora in enumerate(horas):
            fila = {"DIA": pd.Timestamp("2023-01-01") + pd.Timedelta(days=d), "HORA": hora}
            fila.update({c: totales[c][d] * h / 23 for c in columnas})
            filas.append(fila)
    return pd.DataFrame(filas)


def escribir_excel(ruta, sin_hojas=(), sin_columnas=(), con_extras=False, dias=DIAS):
    azar = np.random.default_rng(0)
    totales = {c: 20000 + 3000 * azar.random(dias) for c in MEDICIONES}
    hojas = {"Consolidado EE": ["Frio (Kw)", "Servicios (Kw)"], "Consolidado Agua": ["Agua Planta (Hl)"],
             "Totalizadores Energia": ["KW Trafo 9"], "Consolidado KPI": ["ET Bodega/Hl"]}
    nombre_largo = {"Agua Planta (Hl)": AGUA, "KW Trafo 9": TRAFO, "ET Bodega/Hl": "Consolidado KPI | ET Bodega/Hl"}
    with pd.ExcelWriter(ruta) as libro:
        for nombre, columnas in hojas.items():
            if nombre in sin_hojas or nombre == "Consolidado KPI" and not con_extras:
                continue
            tabla = hoja(columnas, {c: totales[nombre_largo.get(c, c)] for c in columnas}, dias)
            tabla = tabla.drop(columns=[c for c in sin_columnas if c in tabla.columns])
            if con_extras:
                tabla["Columna de más"] = 1.0
                tabla = tabla[["Columna de más", *[c for c in tabla.columns if c != "Columna de más"]]]   # otro orden
            tabla.to_excel(libro, sheet_name=nombre, index=False)
        if con_extras:
            pd.DataFrame({"x": [1]}).to_excel(libro, sheet_name="Hoja de más", index=False)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Modelo v1.9.0 y v1.10.0 ajustados con datos al azar, su registro y su linaje, todo en una carpeta temporal."""
    azar = np.random.default_rng(1)
    X = pd.DataFrame(azar.normal(size=(60, len(COLUMNAS_MODELO))), columns=COLUMNAS_MODELO)
    y = X.sum(axis=1)
    preprocesador = ColumnTransformer([("numericas", Pipeline([("imputar", SimpleImputer(strategy="median")),
                                                              ("escalar", StandardScaler())]), COLUMNAS_MODELO)])
    modelo = Pipeline([("preprocessor", preprocesador), ("regressor", Ridge())]).fit(X, y)
    (tmp_path / "models").mkdir()
    (tmp_path / "data" / "processed").mkdir(parents=True)
    registro = {}
    for version in ("1.9.0", "1.10.0"):
        joblib.dump(modelo, tmp_path / "models" / f"modelo_v{version}.pkl")
        registro[version] = {"path": f"models/modelo_v{version}.pkl"}
    (tmp_path / "models" / "model_registry.json").write_text(json.dumps(registro), encoding="utf-8")
    (tmp_path / "data" / "processed" / "data_lineage.json").write_text(
        json.dumps({"reglas": reglas_sinteticas()}), encoding="utf-8")
    monkeypatch.setattr(predict, "RAIZ", tmp_path)
    monkeypatch.setattr(predict, "RUTA_REGISTRO", tmp_path / "models" / "model_registry.json")
    monkeypatch.setattr(predict, "RUTA_LINAJE", tmp_path / "data" / "processed" / "data_lineage.json")
    return tmp_path


def test_predice_una_fila_por_dia_con_las_dos_columnas_pedidas(entorno):
    escribir_excel(entorno / "nuevo.xlsx")
    resultado = predict.predict_consumption(entorno / "nuevo.xlsx")
    assert list(resultado.columns) == ["fecha", "prediccion_frio_kw"]
    assert len(resultado) == DIAS
    assert resultado["fecha"].iloc[0] == pd.Timestamp("2023-01-01") and resultado["fecha"].iloc[-1] == pd.Timestamp("2023-02-09")
    assert resultado["prediccion_frio_kw"].notna().all()                      # el último día también se predice


def test_las_hojas_y_columnas_de_mas_y_en_otro_orden_se_ignoran(entorno):
    escribir_excel(entorno / "minimo.xlsx")
    escribir_excel(entorno / "con_extras.xlsx", con_extras=True)
    minimo = predict.predict_consumption(entorno / "minimo.xlsx")
    con_extras = predict.predict_consumption(entorno / "con_extras.xlsx")
    pd.testing.assert_frame_equal(minimo, con_extras, check_exact=False)


def test_no_pide_las_hojas_que_las_variables_no_usan(entorno):
    """El modelo de prueba no usa ninguna medición de `Consolidado KPI`: aunque la variable base exista, no hace falta."""
    dependencias = resolver_dependencias(COLUMNAS_MODELO, reglas_sinteticas())
    assert dependencias["hojas"] == ["Consolidado Agua", "Consolidado EE", "Totalizadores Energia"]
    assert "Consolidado KPI | ET Bodega/Hl" not in dependencias["columnas_crudas"]


def test_los_primeros_dias_sin_historia_se_predicen_y_se_avisan(entorno):
    escribir_excel(entorno / "nuevo.xlsx")
    resultado = predict.predict_consumption(entorno / "nuevo.xlsx")
    incompletos = resultado.attrs["dias_incompletos"]
    assert list(incompletos["fecha"]) == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")]   # sin rezago de 2 días
    assert "frio_hace_2_dias" in incompletos["variables_vacias"].iloc[0]
    assert resultado["prediccion_frio_kw"].notna().all()


@pytest.mark.parametrize("hoja_que_falta", ["Consolidado Agua", "Totalizadores Energia", "Consolidado EE"])
def test_falta_una_hoja_y_el_mensaje_dice_cual(entorno, hoja_que_falta):
    escribir_excel(entorno / "nuevo.xlsx", sin_hojas=(hoja_que_falta,))
    with pytest.raises(ErrorDeEntrada, match=hoja_que_falta):
        predict.predict_consumption(entorno / "nuevo.xlsx")


def test_falta_una_columna_y_el_mensaje_dice_cual(entorno):
    escribir_excel(entorno / "nuevo.xlsx", sin_columnas=("KW Trafo 9",))
    with pytest.raises(ErrorDeEntrada, match="KW Trafo 9"):
        predict.predict_consumption(entorno / "nuevo.xlsx")


def test_falta_la_columna_de_fecha_y_no_es_un_keyerror(entorno):
    escribir_excel(entorno / "nuevo.xlsx", sin_columnas=("DIA",))
    with pytest.raises(ErrorDeEntrada, match="DIA"):
        predict.predict_consumption(entorno / "nuevo.xlsx")


def test_por_consola_solo_se_listan_los_dias_incompletos_y_se_guarda_el_csv(entorno, capsys):
    escribir_excel(entorno / "nuevo.xlsx")
    salida = entorno / "results" / "predicciones.csv"
    assert predict.main([str(entorno / "nuevo.xlsx"), "--salida", str(salida)]) == 0
    texto = capsys.readouterr().out
    assert "2 de 40 días tienen datos incompletos" in texto
    assert "2023-01-01: 1 variable vacía (frio_hace_2_dias)" in texto and "2023-01-10:" not in texto
    guardado = pd.read_csv(salida)
    assert list(guardado.columns) == ["fecha", "prediccion_frio_kw"] and len(guardado) == DIAS


def test_con_un_error_sale_con_codigo_1_sin_csv_y_con_el_mensaje_en_stderr(entorno, capsys):
    escribir_excel(entorno / "nuevo.xlsx", sin_hojas=("Consolidado Agua",))
    salida = entorno / "results" / "predicciones.csv"
    assert predict.main([str(entorno / "nuevo.xlsx"), "--salida", str(salida)]) == 1
    assert "Consolidado Agua" in capsys.readouterr().err
    assert not salida.exists()


def test_un_archivo_que_no_existe_o_no_es_excel_da_un_mensaje_claro(entorno, capsys):
    (entorno / "texto.xlsx").write_text("no soy un excel", encoding="utf-8")
    assert predict.main([str(entorno / "no_existe.xlsx")]) == 1
    assert "No se encontró" in capsys.readouterr().err
    assert predict.main([str(entorno / "texto.xlsx")]) == 1
    assert "No se pudo abrir" in capsys.readouterr().err


def test_latest_compara_las_versiones_como_numeros(entorno):
    registro = json.loads(predict.RUTA_REGISTRO.read_text(encoding="utf-8"))
    assert predict.elegir_version(registro, "latest") == "1.10.0"          # como texto ganaría 1.9.0
    assert predict.elegir_version(registro, "v1.9.0") == "1.9.0"


def test_una_version_que_no_existe_lista_las_que_hay(entorno):
    with pytest.raises(predict.ErrorDeModelo, match="1.9.0"):
        predict.load_model("2.0.0")


def test_sin_registro_o_con_el_archivo_del_modelo_borrado_el_mensaje_es_claro(entorno):
    (entorno / "models" / "modelo_v1.10.0.pkl").unlink()
    with pytest.raises(predict.ErrorDeModelo, match="no existe"):
        predict.load_model()
    predict.RUTA_REGISTRO.unlink()
    with pytest.raises(predict.ErrorDeModelo, match="registro de modelos"):
        predict.load_model()


def test_predict_no_usa_la_red_ni_la_cache_de_data_interim():
    texto = Path(predict.__file__).read_text(encoding="utf-8")
    assert "requests" not in texto and "urllib" not in texto and "interim" not in texto


# ----------------------------------------------------------------------------------------------------
# Con los datos reales (los Excel no se versionan: se saltea si faltan)

RAIZ = Path(__file__).resolve().parents[1]
EXCEL_REAL = RAIZ / "data" / "Totalizadores Planta 2022-2023.xlsx"


@pytest.mark.skipif(not EXCEL_REAL.exists() or not (RAIZ / "data" / "processed" / "data_lineage.json").exists(),
                    reason="Faltan los Excel en data/ (no se versionan; ver README, sección Datos)")
def test_con_el_excel_real_reproduce_las_variables_de_dataset_final():
    """Prueba clave: las columnas que arma predict.py sobre un Excel crudo son las de dataset_final.csv (tarda ~1 min)."""
    linaje = json.loads((RAIZ / "data" / "processed" / "data_lineage.json").read_text(encoding="utf-8"))
    columnas = linaje["columnas"]
    X, fechas = predict.preparar_para_prediccion(EXCEL_REAL, columnas, predict.cargar_reglas())
    X, fechas = X.reset_index(drop=True), fechas.reset_index(drop=True)
    nuevo = X.assign(DIA=fechas, k=fechas.groupby(fechas).cumcount())           # DIA se repite (ADR 0002): se agrega el orden
    final = pd.read_csv(RAIZ / "data" / "processed" / "dataset_final.csv", parse_dates=["DIA"])
    final["k"] = final.groupby("DIA").cumcount()
    junto = nuevo.merge(final, on=["DIA", "k"], suffixes=("_p", "_f"))
    assert len(junto) > 400
    distintas = pd.Series(False, index=junto.index)
    for c in columnas:
        igual = (junto[c + "_p"].isna() & junto[c + "_f"].isna()) | np.isclose(junto[c + "_p"], junto[c + "_f"], rtol=1e-9, atol=1e-6)
        distintas |= ~igual
    # Solo pueden diferir los primeros días del archivo: ahí el Excel solo no tiene la historia que sí vio el entrenamiento
    assert junto.index[distintas].max() < 7
