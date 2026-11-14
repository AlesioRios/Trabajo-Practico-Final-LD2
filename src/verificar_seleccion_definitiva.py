"""Verifica reglas y reconstrucción oficial solo en Train, sin entrenar modelos."""
from pathlib import Path
import json,sys,hashlib


def main():
    """Comprueba fuentes y preparación de Train; guarda evidencia de la ejecución."""
    R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
    import pandas as pd,numpy as np
    from seleccion_definitiva import cargar_seleccion_definitiva,reconstruir_conjunto_definitivo,preprocesar_conjunto_definitivo
    from experimentos_seleccion import aprender_preparacion
    from preprocessing_pipeline import cargar_tabla_completa
    from auxiliar_functions import verificar_huellas
    D=R/'results/seleccion/cierre_fase23'
    r=cargar_seleccion_definitiva(); huellas={}
    for sub,archivo in [('comparacion_validation_congelada','sha256.json'),('forward_max60_artefactos','sha256.json')]:
     base=R/'results/seleccion'/sub
     h=json.loads((base/archivo).read_text(encoding='utf-8'))
     for p,v in h.items():
      q=Path(p)
      if not q.is_absolute(): q=base/q
      if q.is_file(): huellas[f'{sub}/{q.name}']=hashlib.sha256(q.read_bytes()).hexdigest()==v
    assert all(huellas.values()),huellas
    b=R/'results/seleccion/candidatos_congelados_validation'; assert hashlib.sha256((b/'candidatos.json').read_bytes()).hexdigest()==(b/'sha256.txt').read_text().strip()
    checks=verificar_huellas(R/'data/checksums.json',R/'data'); assert all(v=='coincide' for v in checks.values()),checks
    # El cargador existente reconstruye la tabla; se consume únicamente el prefijo Train.
    train=cargar_tabla_completa(R / 'data').iloc[:848].copy()
    incompletos=train.filas.lt(24); train.loc[incompletos,'valido']=False; train.loc[incompletos,'consumo_dia']=np.nan
    train['objetivo']=train.consumo_dia.shift(-1)
    X=reconstruir_conjunto_definitivo(train,r); elegibles=train.objetivo.notna(); X=X.loc[elegibles]
    A=preprocesar_conjunto_definitivo(X,r)
    prep,A2=aprender_preparacion(X,'standard','ninguna')
    assert prep['columnas']==r['preprocesamiento']['columnas']
    assert prep['indicadores']==r['preprocesamiento']['indicadores']
    for clave in ['centro','escala']:
     np.testing.assert_allclose(prep[clave],r['preprocesamiento'][clave],rtol=1e-12,atol=1e-10)
    for c,v in prep['medianas'].items(): assert np.isclose(v,r['preprocesamiento']['medianas'][c],rtol=1e-12)
    np.testing.assert_allclose(A,A2,rtol=1e-12,atol=1e-10)
    assert A.shape==(829,27) and np.isfinite(A).all()
    # Comprobar causalidad de toda la reconstrucción con prefijo real sin objetivos futuros.
    X0=reconstruir_conjunto_definitivo(train.iloc[:100],r)
    X1=reconstruir_conjunto_definitivo(train,r).iloc[:100]
    pd.testing.assert_frame_equal(X0,X1)
    res={'entorno':sys.version,'fuentes':checks,'huellas':huellas,'train_bloques':len(train),'train_utilizables':len(X),'matriz':list(A.shape),'nombres_y_orden':True,'preparacion_reproducida_solo_train':True,'causalidad_prefijo_real':True,'test_evaluado':False,'validation_reevaluada':False,'modelos_entrenados':False,'disponibilidad_fisica_certificada':False}
    (D/'verificacion_train.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(res,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
