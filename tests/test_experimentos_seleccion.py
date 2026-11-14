"""Pruebas de aislamiento, selección, reglas JSON y reanudación sin datos reservados."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import experimentos_seleccion as e
from test_preprocessing_pipeline import tabla_sintetica

@pytest.mark.parametrize('escala',['standard','robust'])
@pytest.mark.parametrize('trans',['ninguna','yeo-johnson'])
def test_preparacion_serializable_y_aplicacion(escala,trans):
    X=pd.DataFrame({'a':[1.,np.nan,3.,100.],'razon':[2.,4.,np.nan,8.],'binaria':[0.,1.,0.,1.]})
    r,A=e.aprender_preparacion(X,escala,trans)
    r=json.loads(json.dumps(r,allow_nan=False))
    np.testing.assert_allclose(A,e.aplicar_preparacion(X,r),rtol=1e-12,atol=1e-12)
    assert r['medianas']['a']==3.
    assert A.shape[1]==5
    assert e.aplicar_preparacion(X.iloc[:1],r).shape==(1,5)

def test_futuro_no_cambia_aprendizaje_y_gap():
    t=tabla_sintetica(100).assign(valido=True)
    i=np.arange(60);j=np.arange(61,80)
    p=e.preparar_pliegue(t,i,j)
    futuro=t.copy();futuro.loc[61:,'Frio (Kw)']*=100
    futuro.loc[61:,'objetivo']=-1000
    q=e.preparar_pliegue(futuro,i,j)
    assert p['audit']['limpieza']==q['audit']['limpieza']
    assert p['audit']['ingenieria']==q['audit']['ingenieria']
    pd.testing.assert_frame_equal(p['X'],q['X'])
    with pytest.raises(ValueError,match='gap'):
        e.preparar_pliegue(t,i,np.arange(60,80))

def test_regla_equivalencia_y_parsimonia():
    r=[dict(id='a',estado='completo',mae=100.,cantidad=8,std=2.),
       dict(id='b',estado='completo',mae=100.9,cantidad=2,std=4.),
       dict(id='c',estado='completo',mae=101.1,cantidad=1,std=1.)]
    assert e.elegir(r)['id']=='b'

def test_configuracion_reanudable_no_repite_ajustes(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'RAIZ',tmp_path)
    monkeypatch.setattr(e,'firma_codigo',lambda: {'sintetico':'sin-datos-reales'})
    monkeypatch.setattr(e,'medir',lambda p,cfg:dict(mae=10.,cantidad=2))
    cfg=dict(metodo='spearman',k=2)
    carpeta=tmp_path/'run';carpeta.mkdir()
    r=e.evaluar_config([{}],cfg,carpeta,'interno')
    monkeypatch.setattr(e,'medir',lambda *args: (_ for _ in ()).throw(AssertionError('No repetir')))
    assert e.evaluar_config([{}],cfg,carpeta,'interno')==r


def test_escritura_reintenta_bloqueo_temporal(tmp_path,monkeypatch):
    original=Path.replace
    intentos=[]
    def reemplazar(p,destino):
        intentos.append(1)
        if len(intentos)<3: raise PermissionError('bloqueo temporal simulado')
        return original(p,destino)
    monkeypatch.setattr(Path,'replace',reemplazar)
    monkeypatch.setattr(e.time,'sleep',lambda _:None)
    e.guardar(tmp_path/'registro.json',{'completo':True})
    assert len(intentos)==3
    assert json.loads((tmp_path/'registro.json').read_text())=={'completo':True}
