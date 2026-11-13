from pathlib import Path
import sys,os,time,json,uuid,hashlib,platform,traceback,argparse
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from limpieza import aprender_reglas_de_limpieza, limpiar, COLUMNAS_DE_CONTROL
from variables_temporales import crear_variables_temporales
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experimentos_seleccion as e

RAIZ=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--max-features',type=int,default=40);args=parser.parse_args()
if args.max_features<1: parser.error('--max-features debe ser positivo')
LIMITE=args.max_features
OUT=RAIZ/f'results/seleccion/forward-tradicional-corregido-max{LIMITE}-20261113-01'

def main():
 OUT.mkdir(parents=True,exist_ok=False); start=time.time(); estado=dict(pid=os.getpid(),estado='preparando',inicio_utc=datetime.now(timezone.utc).isoformat(),ronda=0,maximo_rondas=LIMITE,combinaciones_evaluadas=0,seleccionadas=[],mejor_mae=None)
 def progreso(**kw):
  estado.update(kw);estado['segundos_transcurridos']=time.time()-start;estado['actualizado_utc']=datetime.now(timezone.utc).isoformat();e.guardar(OUT/'progreso.json',estado)
 progreso();tabla,datos=e.cargar_entrenamiento();folds=[]
 with threadpool_limits(limits=1):
  particiones=list(TimeSeriesSplit(n_splits=3,gap=1).split(tabla))
  # Catálogo fijado con limpieza no supervisada del primer entrenamiento.
  primer_i,_=particiones[0]; reglas0=aprender_reglas_de_limpieza(tabla.iloc[primer_i]); inicial=limpiar(tabla.iloc[primer_i],reglas0)
  inicial=crear_variables_temporales(inicial)
  cols=sorted(c for c in inicial.columns if c not in COLUMNAS_DE_CONTROL and pd.api.types.is_numeric_dtype(inicial[c]))
  assert cols and 'objetivo' not in cols
  for k,(i,j) in enumerate(particiones):
   assert i[-1]+1<j[0]
   reglas=aprender_reglas_de_limpieza(tabla.iloc[i]);historia=limpiar(tabla.iloc[:int(j[-1])+1],reglas);historia=crear_variables_temporales(historia)
   tr=historia.iloc[i];va=historia.iloc[j];tr=tr.loc[tr.objetivo.notna()];va=va.loc[va.objetivo.notna()]
   X=tr.reindex(columns=cols).astype(float);V=va.reindex(columns=cols).astype(float)
   prep,A=e.aprender_preparacion(X,'standard','ninguna');VV=e.aplicar_preparacion(V,prep)
   p=dict(X=X,V=V,y=tr.objetivo,z=va.objetivo,mapa={})
   folds.append((p,prep,A,VV));e.guardar(OUT/f'preparacion_pliegue_{k+1}.json',dict(train_indices=tr.index.tolist(),eval_indices=va.index.tolist(),limpieza=reglas,preparacion=prep));progreso(pliegue_preparado=k+1)
  cfg={'metodo':'forward_tradicional','candidatas':cols,'semilla':42,'ridge_alpha':100.,'solver':'svd','gap':1,'pliegues':3,'maximo_efectivas':LIMITE,'imputacion':'mediana','escalador':'standard','datos':datos,'codigo':e.firma_codigo(),'python':sys.version,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'estimacion_maxima_ajustes':3*sum(len(cols)-i for i in range(min(LIMITE,len(cols))))}
  e.guardar(OUT/'configuracion.json',cfg);progreso(estado='ejecutando',candidatas_iniciales=len(cols),estimacion_maxima_ajustes=cfg['estimacion_maxima_ajustes'])
  seleccion=[];mejor=float('inf');mejor_sel=[];acumuladas=0
  for ronda in range(1,min(LIMITE,len(cols))+1):
   restantes=[c for c in cols if c not in seleccion];resultados=[];progreso(ronda=ronda,combinaciones_evaluadas=0,combinaciones_pendientes=len(restantes))
   for num,c in enumerate(restantes,1):
    try:
     cierres=[e.cerrar_jerarquia(seleccion+[c],p['mapa'],p['X'].columns) for p,_,_,_ in folds]
     tamanos=[len(nombres)+sum(x in prep['indicadores'] for x in nombres) for nombres,(_,prep,_,_) in zip(cierres,folds)]
     if any(n>LIMITE for n in tamanos):
      reg=dict(ronda=ronda,candidata=c,estado=f'excluida_limite_{LIMITE}',cantidades=tamanos);
      with (OUT/'evaluaciones.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(reg,ensure_ascii=False)+'\n');f.flush()
      progreso(combinaciones_revisadas=num,combinaciones_pendientes=len(restantes)-num,ultima_candidata=c);continue
     fs=[]; efectivos=[]
     for p,prep,A,V in folds:
      nombres=e.cerrar_jerarquia(seleccion+[c],p['mapa'],p['X'].columns); ids=[prep['columnas'].index(x) for x in nombres];ind=[len(prep['columnas'])+prep['indicadores'].index(x) for x in prep['indicadores'] if x in nombres];model=Ridge(alpha=100.,solver='svd',random_state=42).fit(A[:,ids+ind],p['y']);fs.append(float(mean_absolute_error(p['z'],model.predict(V[:,ids+ind]))));efectivos.append(nombres);assert A[:,ids+ind].shape[1]<=LIMITE
     reg=dict(ronda=ronda,candidata=c,directas=seleccion+[c],variables_efectivas=efectivos,columnas_modelo=tamanos,mae_pliegues=fs,mae=float(np.mean(fs)),std=float(np.std(fs)),estado='completo')
     resultados.append(reg);acumuladas+=1
    except Exception:reg=dict(ronda=ronda,candidata=c,estado='error',error=traceback.format_exc())
    with (OUT/'evaluaciones.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(reg,ensure_ascii=False)+'\n');f.flush()
    progreso(combinaciones_evaluadas=len(resultados),combinaciones_revisadas=num,combinaciones_acumuladas=acumuladas,combinaciones_pendientes=len(restantes)-num,ultima_candidata=c,ultimo_mae=reg.get('mae'))
   if not resultados:
    progreso(estado='completo',motivo_parada='sin candidatas compatibles con el límite o sin ajustes válidos');break
   ganador=min(resultados,key=lambda r:r['mae']);anterior=mejor;seleccion.append(ganador['candidata']);
   if ganador['mae']<mejor:mejor=ganador['mae'];mejor_sel=seleccion.copy()
   checkpoint=dict(ronda=ronda,ganador=ganador,mejor_mae=mejor,mejor_subconjunto=mejor_sel);e.guardar(OUT/f'checkpoint_ronda_{ronda:02}.json',checkpoint)
   progreso(seleccionadas=seleccion,mejor_conjunto=mejor_sel,mejor_mae=mejor,variable_incorporada=ganador['candidata'],mae_ronda=ganador['mae'],cantidades_efectivas=ganador['columnas_modelo'],mae_pliegues=ganador['mae_pliegues'],ultimo_checkpoint=f'checkpoint_ronda_{ronda:02}.json')
   if ganador['mae']>=anterior:
    progreso(estado='completo',motivo_parada='sin mejora estricta frente al mejor MAE previo');break
  progreso(estado='completo',motivo_parada=estado.get('motivo_parada','máximo de rondas o candidatas'))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(parents=True,exist_ok=True);(OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
