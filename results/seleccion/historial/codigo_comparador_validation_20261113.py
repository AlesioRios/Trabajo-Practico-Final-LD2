"""Comparación congelada en validation; se ejecuta solo bajo solicitud explícita."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
import experimentos_seleccion as e
from limpieza import limpiar, anular_dias_invalidos, _es_columna_de_energia, FACTOR_SOBRE_P99

ROOT=Path(__file__).resolve().parents[1]
IND=ROOT/'results/seleccion/ejecucion-20261113T184058Z-969d4545'
FWD=ROOT/'results/seleccion/forward_max60_artefactos'

def comparar_matrices(X,y,V,z,conjuntos):
    """Mismo preprocesamiento ajustado en train; z debe estar alineado y sin faltantes."""
    if not V.index.equals(z.index) or z.isna().any(): raise ValueError('Validation desalineada o con objetivo ausente')
    if X.index.intersection(V.index).size: raise ValueError('Train y validation se solapan')
    resultados=[]
    for nombre,columnas in conjuntos.items():
        if not columnas or len(set(columnas))!=len(columnas): raise ValueError(f'Conjunto inválido: {nombre}')
        faltan=set(columnas)-set(X.columns) | (set(columnas)-set(V.columns))
        if faltan: raise ValueError(f'{nombre}: variables no reconstruibles: {sorted(faltan)}')
        prep,A=e.aprender_preparacion(X[columnas],'standard','ninguna');B=e.aplicar_preparacion(V[columnas],prep)
        pred=Ridge(alpha=100.,solver='svd',random_state=42).fit(A,y).predict(B)
        resultados.append(dict(metodo=nombre,mae=float(mean_absolute_error(z,pred)),variables=len(columnas),columnas_efectivas=A.shape[1],filas=len(z),preparacion=prep,predicciones=pred.tolist()))
    return resultados

def main():
    from preprocessing_pipeline import cargar_tabla_completa
    from auxiliar_functions import calcular_baselines
    congelado=ROOT/'results/seleccion/candidatos_congelados_validation/candidatos.json'
    esperado=(congelado.parent/'sha256.txt').read_text(encoding='utf-8').strip()
    if hashlib.sha256(congelado.read_bytes()).hexdigest()!=esperado:raise ValueError('Candidatos congelados alterados')
    conjuntos=json.loads(congelado.read_text(encoding='utf-8'))['conjuntos']
    checks=e.verificar_huellas(ROOT/'data/checksums.json',ROOT/'data')
    if any(v!='coincide' for v in checks.values()):raise ValueError('Fuentes incompatibles')
    data=cargar_tabla_completa(usar_cache=True); n=len(data);a=int(.70*n);b=int(.85*n);data=data.iloc[:b].copy()
    invalidas=data.filas.lt(24);data.loc[invalidas,'valido']=False;data.loc[invalidas,'consumo_dia']=np.nan
    data['objetivo']=np.nan;data.iloc[:a,data.columns.get_loc('objetivo')]=data.iloc[:a].consumo_dia.shift(-1).to_numpy();data.iloc[a:b,data.columns.get_loc('objetivo')]=data.iloc[a:b].consumo_dia.shift(-1).to_numpy()
    audit=json.loads((IND/'preparacion/final.json').read_text(encoding='utf-8'))
    # El conjunto Forward puede contener mediciones que la limpieza histórica descartó.
    # Se restablecen exclusivamente columnas crudas congeladas, con reglas aprendidas en Train.
    fwd=conjuntos['forward']; reglas=audit['limpieza']
    extras=sorted(set(fwd).intersection(data.columns)-set(reglas['columnas_conservadas'])-set(e.COLUMNAS_DE_CONTROL))
    for col in extras:
        reglas['columnas_conservadas'].append(col)
        if _es_columna_de_energia(col): reglas['columnas_de_energia'].append(col)
        if 'Temp' in col: reglas['columnas_de_temperatura'].append(col)
        bruto=data.iloc[:a].copy()
        if col in reglas['columnas_de_energia']:bruto=anular_dias_invalidos(bruto,[col])
        q=pd.to_numeric(bruto[col],errors='coerce').quantile(.99)*FACTOR_SOBRE_P99
        reglas['limite_superior'][col]=float(q) if pd.notna(q) and np.isfinite(q) and q>0 else None
    ingeniada=e.aplicar_ingenieria(limpiar(data,reglas),audit['ingenieria'])
    train=ingeniada.iloc[:a];val=ingeniada.iloc[a:b];train=train.loc[train.objetivo.notna()];val=val.loc[val.objetivo.notna()]
    baselines=calcular_baselines(data).loc[val.index];comunes=baselines.notna().all(axis=1); baselines_resultados={}
    for nombre in ['persistencia','media_movil']:
        baselines_resultados[nombre]=dict(mae=float(mean_absolute_error(val.objetivo.loc[comunes],baselines.loc[comunes,nombre])),filas=int(comunes.sum()),indices=val.index[comunes].tolist())
    with threadpool_limits(limits=1):resultado=comparar_matrices(train.drop(columns=e.COLUMNAS_DE_CONTROL,errors='ignore'),train.objetivo,val.drop(columns=e.COLUMNAS_DE_CONTROL,errors='ignore'),val.objetivo,conjuntos)
    indices_comunes=np.flatnonzero(comunes.to_numpy());
    for r in resultado:r['mae_en_filas_baseline']=float(mean_absolute_error(val.objetivo.iloc[indices_comunes],np.asarray(r['predicciones'])[indices_comunes]))
    out=ROOT/'results/seleccion/comparacion_validation_congelada';out.mkdir(exist_ok=False,parents=True)
    e.guardar(out/'resultados.json',dict(resultados=resultado,indices_validation=val.index.tolist(),fechas_validation=val.DIA.astype(str).tolist(),reales=val.objetivo.tolist(),baselines=baselines_resultados,candidatos_sha256=esperado,fuentes=checks,reglas_limpieza=reglas,columnas_restablecidas=extras,ridge={'alpha':100.,'solver':'svd'},nota='No se usa test; ningún ajuste o selección con Validation.'))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('preparacion','predicciones')} for r in resultado]).to_csv(out/'resumen.csv',index=False)
    print(out)
if __name__=='__main__':main()
