from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from comparar_selecciones_validation import comparar_matrices

def datos():
 X=pd.DataFrame({'a':[1.,2.,3.,4.],'b':[None,1.,2.,3.]},index=[0,1,2,3]);y=pd.Series([2.,3.,4.,5.],index=X.index);V=pd.DataFrame({'a':[5.,6.],'b':[4.,None]},index=[4,5]);z=pd.Series([6.,7.],index=V.index);return X,y,V,z

def test_mismas_filas_indicadores_y_prediccion_sin_etiquetas_de_validation():
 X,y,V,z=datos();conj={'uno':['a'],'dos':['a','b']};r=comparar_matrices(X,y,V,z,conj);c=comparar_matrices(X,y,V,z+100,conj)
 assert [v['filas'] for v in r]==[2,2] and r[1]['columnas_efectivas']==3
 assert [v['predicciones'] for v in r]==[v['predicciones'] for v in c]

def test_rechaza_columnas_ausentes_y_solapamiento():
 X,y,V,z=datos()
 with pytest.raises(ValueError,match='no reconstruibles'):comparar_matrices(X,y,V,z,{'x':['ausente']})
 V.index=[3,4];z.index=V.index
 with pytest.raises(ValueError,match='solapan'):comparar_matrices(X,y,V,z,{'x':['a']})


def test_frontera_no_importa_objetivo_de_validation():
 from comparar_selecciones_validation import construir_objetivo_por_tramos
 tabla=pd.DataFrame({'consumo_dia':[10.,20.,30.,100.,200.,300.]})
 y=construir_objetivo_por_tramos(tabla,[3])
 assert y.iloc[:2].tolist()==[20.,30.]
 assert np.isnan(y.iloc[2]) and np.isnan(y.iloc[-1])
 futura=tabla.copy();futura.loc[3:,'consumo_dia']=-9999.
 pd.testing.assert_series_equal(y.iloc[:3],construir_objetivo_por_tramos(futura,[3]).iloc[:3])


def test_etiquetas_evaluacion_no_cambian_preparacion_ni_ajuste(monkeypatch):
 import comparar_selecciones_validation as c
 original=c.Ridge;ajustes=[]
 class Espia:
  def __init__(self,**kwargs): self.modelo=original(**kwargs)
  def fit(self,A,y):
   ajustes.append((A.copy(),y.copy()));self.modelo.fit(A,y);return self
  def predict(self,V):return self.modelo.predict(V)
 monkeypatch.setattr(c,'Ridge',Espia)
 X,y,V,z=datos()
 a=c.comparar_matrices(X,y,V,z,{'uno':['a','b']})[0]
 b=c.comparar_matrices(X,y,V,z*1000,{'uno':['a','b']})[0]
 assert a['preparacion']==b['preparacion']
 assert a['predicciones']==b['predicciones']
 for A,etiquetas in ajustes:
  assert A.shape[0]==len(X)
  pd.testing.assert_series_equal(etiquetas,y)
