"""Experimentos auditables de Fase 2.3; exclusivamente entrenamiento temporal.

No entrena modelos finales. La preparación experimental es independiente del
Pipeline final; reutiliza limpieza e ingeniería versionadas, por pliegue.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import shutil
import sys
import time
import traceback
import uuid
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from threadpoolctl import threadpool_limits
from auxiliar_functions import SEMILLA, huella, verificar_huellas
from limpieza import COLUMNAS_DE_CONTROL, aprender_reglas_de_limpieza, limpiar
from variables_temporales import crear_variables_temporales
from variables_planta import aprender_reglas_de_planta, crear_variables_planta
from variables_interacciones import (aprender_reglas_de_interacciones, crear_interacciones,
                                    construir_mapa_interacciones, columnas_candidatas)
from seleccion_de_variables import aprender_seleccion, cerrar_jerarquia, resumir_estabilidad

RAIZ = Path(__file__).resolve().parents[1]
FORWARD_LIMITE = float('inf')
METODOS = ('spearman', 'pearson', 'mutua', 'lasso')
POLITICA = dict(semilla=SEMILLA, externos=4, internos=3, gap=1, ridge_alpha=100.,
    min_observaciones=20, min_cobertura=.60, equivalencia=.01,
    cantidades=[1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 80, 120],
    alphas=[.1, 1., 10., 100., 1000., 10000.],
    escaladores=['standard', 'robust'], transformaciones=['ninguna', 'yeo-johnson'],
    criterio='MAE dentro del 1% del minimo; menor cantidad efectiva; menor desviacion; ID estable')


def normalizar(obj):
    """Convierte a JSON estricto; ausencias no son números inventados."""
    if isinstance(obj, dict): return {str(k): normalizar(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray, pd.Index)): return [normalizar(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (float,np.floating)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (Path,pd.Timestamp)): return str(obj)
    return obj


def guardar(ruta, obj):
    """Escritura atómica de un artefacto dentro de una ejecución identificada."""
    ruta=Path(ruta);ruta.parent.mkdir(parents=True,exist_ok=True)
    tmp=ruta.with_suffix(ruta.suffix+'.tmp')
    tmp.write_text(json.dumps(normalizar(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Reintentos por bloqueos temporales de archivos sincronizados en Windows.
    for intento in range(12):
        try:
            tmp.replace(ruta)
            break
        except PermissionError:
            if intento==11: raise
            time.sleep(min(.05 * 2**intento, 2.))


def clave(obj):
    return hashlib.sha256(json.dumps(normalizar(obj),sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]


def evento(carpeta, tipo, **datos):
    with (carpeta/'eventos.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(normalizar(dict(fecha=datetime.now(timezone.utc).isoformat(),tipo=tipo,**datos)),ensure_ascii=False)+'\n')


def progreso(carpeta, metodo, **estado):
    """Snapshot vivo; el historial de cambios queda en eventos de la ejecución."""
    ruta=RAIZ/'results/seleccion/progreso.json'
    previo=json.loads(ruta.read_text(encoding='utf-8')) if ruta.exists() else {}
    if previo.get('ejecucion')!=carpeta.name:
        previo=dict(ejecucion=carpeta.name,ruta=carpeta.relative_to(RAIZ).as_posix(),metodos={})
    actual=previo['metodos'].setdefault(metodo,{})
    actual.update(estado)
    if estado.get('estado') in ('iniciado','ejecutando','completo'): actual.pop('error',None)
    actual['actualizado_utc']=datetime.now(timezone.utc).isoformat()
    previo['actualizado_utc']=actual['actualizado_utc']
    guardar(ruta,previo);guardar(carpeta/'progreso.json',previo)
    evento(carpeta,'progreso',metodo=metodo,**estado)


def firma_codigo():
    paths=sorted((RAIZ/'src').glob('*.py'))+[RAIZ/'requirements.txt']
    return {p.relative_to(RAIZ).as_posix():huella(p) for p in paths}


def cargar_entrenamiento(sin_cache=False):
    """Verifica fuentes y reconstruye; no consulta valores de validation/test para decidir."""
    from preprocessing_pipeline import cargar_tabla_completa
    checks=verificar_huellas(RAIZ/'data/checksums.json',RAIZ/'data')
    if any(v!='coincide' for v in checks.values()):
        raise ValueError(f'Fuentes o artefactos con SHA-256 incompatible: {checks}')
    tabla=cargar_tabla_completa(RAIZ / 'data', usar_cache=not sin_cache)
    n=len(tabla);a=int(.70*n);b=int(.85*n)
    limites=[]
    for nombre,i,j in [('train',0,a),('validation',a,b),('test',b,n)]:
        limites.append(dict(tramo=nombre,filas=j-i,inicio_pos=i,fin_pos=j-1,
            etiqueta_inicio=str(tabla.DIA.iloc[i]),etiqueta_fin=str(tabla.DIA.iloc[j-1])))
    train=tabla.iloc[:a].copy()
    # Consigna oficial prima sobre la excepción histórica de ADR 0004/0005.
    incompletos=train.filas.lt(24)
    train.loc[incompletos,'valido']=False
    train.loc[incompletos,'consumo_dia']=np.nan
    train['objetivo']=train.consumo_dia.shift(-1)
    return train,dict(divisiones=limites,fuentes=checks,
        filas_estrictas_menos_24=train.index[incompletos].tolist(),
        etiquetas_utilizables=int(train.objetivo.notna().sum()),
        ultima_etiqueta_excluida=True,fechas='etiquetas originales; horizonte por bloque, no calendario certificado')


def aprender_ingenieria(t):
    """Reglas JSON, aprendidas solo con el prefijo autorizado."""
    planta=aprender_reglas_de_planta(t)
    base=crear_variables_planta(crear_variables_temporales(t),planta)
    return dict(planta=planta,interacciones=aprender_reglas_de_interacciones(base))


def aplicar_ingenieria(t, reglas):
    return crear_interacciones(crear_variables_planta(crear_variables_temporales(t),reglas['planta']),reglas['interacciones'])


def preparar_pliegue(tabla, i, j, carpeta=None, nombre='final'):
    """Aprende limpieza e ingeniería por prefijo; historia causal antes de quitar NaN."""
    if len(j) and i[-1]+1 >= j[0]: raise ValueError('Se exige gap >= 1 para etiqueta D+1.')
    bruto=tabla.iloc[i].copy()
    rl=aprender_reglas_de_limpieza(bruto)
    rv=aprender_ingenieria(limpiar(bruto,rl))
    historia=tabla.iloc[:int(j[-1])+1] if len(j) else tabla.iloc[:int(i[-1])+1]
    creada=aplicar_ingenieria(limpiar(historia,rl),rv)
    tr=creada.iloc[i];va=creada.iloc[j]
    tr=tr.loc[tr.objetivo.notna()];va=va.loc[va.objetivo.notna()]
    X=tr.drop(columns=COLUMNAS_DE_CONTROL,errors='ignore').astype(float)
    V=va.reindex(columns=X.columns).astype(float)
    if not len(X): raise ValueError('Sin etiquetas utilizables en entrenamiento.')
    mapa=construir_mapa_interacciones(rv)
    candidatas=columnas_candidatas(X)
    perfil=pd.DataFrame({'faltantes_pct':X.isna().mean()*100,'n':X.count(),
        'media':X.mean(),'std':X.std(),'min':X.min(),'q01':X.quantile(.01),
        'q25':X.quantile(.25),'mediana':X.median(),'q75':X.quantile(.75),
        'q99':X.quantile(.99),'max':X.max(),'asimetria':X.skew()})
    perfil['iqr']=perfil.q75-perfil.q25
    perfil['extremos_iqr']=((X<perfil.q25-1.5*perfil.iqr)|(X>perfil.q75+1.5*perfil.iqr)).sum()
    perfil['pearson']=X.corrwith(tr.objetivo,method='pearson')
    perfil['spearman']=X.corrwith(tr.objetivo,method='spearman')
    # Diagnóstico de influencia únicamente; no winsoriza los datos del experimento.
    perfil['pearson_sin_extremos_1_99']=X.where((X>=perfil.q01)&(X<=perfil.q99)).corrwith(tr.objetivo)
    audit=dict(nombre=nombre,train_indices=tr.index.tolist(),eval_indices=va.index.tolist(),
        fin_train=int(i[-1]),inicio_eval=int(j[0]) if len(j) else None,
        limpieza=rl,ingenieria=rv,candidatas=candidatas,
        excluidas_limpieza=[c for c in bruto if c not in COLUMNAS_DE_CONTROL and c not in X],
        excluidas_constantes=[c for c in X if c not in candidatas],
        celdas_anuladas=int((bruto[rl['columnas_conservadas']].notna() & limpiar(bruto,rl)[rl['columnas_conservadas']].isna()).sum().sum()))
    if carpeta:
        guardar(carpeta/'preparacion'/f'{nombre}.json',audit)
        perfil.to_csv(carpeta/'preparacion'/f'{nombre}_perfil.csv',index_label='variable')
    return dict(X=X,y=tr.objetivo,V=V,z=va.objetivo,mapa=mapa,candidatas=candidatas,
                audit=audit,perfil=perfil,rankings={},preparaciones={})


def aprender_preparacion(X, escalador, transformacion):
    """Mediana + indicadores de ausencia; Yeo-Johnson solo continuas, reglas JSON."""
    med=X.median().fillna(0.)
    A=X.fillna(med).to_numpy(dtype=float)
    cols=X.columns.tolist();indicadores=X.columns[X.isna().any()].tolist()
    reglas=dict(columnas=cols,medianas=med.to_dict(),indicadores=indicadores,
        transformacion=transformacion,escalador=escalador,yeo={})
    if transformacion=='yeo-johnson':
        for k,c in enumerate(cols):
            if X[c].nunique()>2:
                # Preacondicionamiento positivo de escala: evita overflow de productos industriales.
                factor=float(max(np.std(A[:,k]),1.))
                pt=PowerTransformer(method='yeo-johnson',standardize=False)
                A[:,k]=pt.fit_transform((A[:,k]/factor).reshape(-1,1)).ravel()
                reglas['yeo'][c]=dict(factor=factor,lamb=float(pt.lambdas_[0]))
    sc=StandardScaler() if escalador=='standard' else RobustScaler()
    A=sc.fit_transform(A)
    reglas['centro']=(sc.mean_ if escalador=='standard' else sc.center_).tolist()
    reglas['escala']=sc.scale_.tolist()
    if indicadores: A=np.column_stack([A,X[indicadores].isna().astype(float)])
    return reglas,A


def aplicar_preparacion(X, reglas):
    """Aplica sin ajustar; indicadores distinguen ausencia estructural de un día normal."""
    X=X[reglas['columnas']];A=X.fillna(reglas['medianas']).to_numpy(dtype=float)
    for c,r in reglas['yeo'].items():
        k=reglas['columnas'].index(c);A[:,k]=yeojohnson(A[:,k]/r['factor'],r['lamb'])
    A=(A-np.array(reglas['centro']))/np.array(reglas['escala'])
    if reglas['indicadores']: A=np.column_stack([A,X[reglas['indicadores']].isna().astype(float)])
    return A


def ranking(p, metodo):
    if metodo not in p['rankings']:
        X=p['X'];y=p['y'];cols=p['candidatas']
        cfg=dict(metodo=metodo,k=1,vecinos=3,discretas=[c for c in cols if X[c].nunique()<=2])
        # Reutiliza el selector existente; el ranking completo no fija una cantidad final.
        r=aprender_seleccion(X,y,cfg,p['mapa'])
        scores=pd.Series(r['puntajes'],dtype=float)
        cobertura=X[cols].notna().mean();n=X[cols].count()
        aptas=(cobertura>=POLITICA['min_cobertura'])&(n>=POLITICA['min_observaciones'])
        if metodo in ('spearman','pearson'): scores=scores.where(aptas)
        orden=scores.dropna().sort_values(ascending=False,kind='stable').index.tolist()
        p['rankings'][metodo]=dict(orden=orden,puntajes=r['puntajes'],
            cobertura=cobertura.to_dict(),observaciones=n.to_dict(),
            excluidas_pares=scores.index[scores.isna()].tolist())
    return p['rankings'][metodo]


def seleccionar(p,cfg):
    m=cfg['metodo'];extra={}
    if m=='lasso':
        cols=p['candidatas'];X=p['X'][cols]
        cache=('lasso',cfg['escalador'],cfg['transformacion'])
        if cache not in p['preparaciones']:
            p['preparaciones'][cache]=aprender_preparacion(X,cfg['escalador'],cfg['transformacion'])
        prep,A=p['preparaciones'][cache]
        modelo=Lasso(alpha=cfg['alpha'],max_iter=20000,tol=.0001,random_state=SEMILLA)
        modelo.fit(A,p['y'])
        # Un indicador no nulo conserva también la variable de la que informa ausencia.
        directas=[c for c,v in zip(cols,modelo.coef_[:len(cols)]) if v!=0]
        directas+= [c for c,v in zip(prep['indicadores'],modelo.coef_[len(cols):]) if v!=0 and c not in directas]
        extra=dict(coeficientes=dict(zip(cols+['ausente::'+c for c in prep['indicadores']],modelo.coef_)),
            iteraciones=int(modelo.n_iter_),dual_gap=float(modelo.dual_gap_),preparacion_selector=prep)
    else:
        r=ranking(p,'spearman' if m=='forward' else m);orden=r['orden']
        if m=='forward':
            directas=[orden[pos] for pos in cfg['posiciones'] if pos<len(orden)]
        else: directas=orden[:min(cfg['k'],len(orden))]
        extra=dict(ranking=r)
    if not directas: raise ValueError('Configuración sin variables seleccionadas.')
    elegidas=cerrar_jerarquia(directas,p['mapa'],p['X'].columns)
    return dict(directas=directas,seleccionadas=elegidas,
        por_jerarquia=[c for c in elegidas if c not in directas],**extra)


def medir(p,cfg):
    inicio=time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        sel=seleccionar(p,cfg);cols=sel['seleccionadas']
        cache=('ridge',cfg['escalador'],cfg['transformacion'])
        if cache not in p['preparaciones']:
            prep_total,A_total=aprender_preparacion(p['X'],cfg['escalador'],cfg['transformacion'])
            V_total=aplicar_preparacion(p['V'],prep_total)
            p['preparaciones'][cache]=(prep_total,A_total,V_total)
        total,A_total,V_total=p['preparaciones'][cache]
        ids=[total['columnas'].index(c) for c in cols]
        indicadores=[c for c in total['indicadores'] if c in cols]
        indices=ids+[len(total['columnas'])+total['indicadores'].index(c) for c in indicadores]
        prep=dict(total,columnas=cols,medianas={c:total['medianas'][c] for c in cols},
            indicadores=indicadores,yeo={c:total['yeo'][c] for c in cols if c in total['yeo']},
            centro=[total['centro'][k] for k in ids],escala=[total['escala'][k] for k in ids])
        A=A_total[:,indices]
        modelo=Ridge(alpha=100.,solver='svd',random_state=SEMILLA).fit(A,p['y'])
        pred=modelo.predict(V_total[:,indices])
    return dict(mae=float(mean_absolute_error(p['z'],pred)),cantidad=len(cols),
        cantidad_modelo=A.shape[1],seleccionadas=cols,seleccion=sel,
        preparacion=prep,indices_evaluados=p['V'].index.tolist(),
        predicciones=pred.tolist(),reales=p['z'].tolist(),
        n_candidatas=len(p['candidatas']),segundos=time.perf_counter()-inicio,
        avisos=[str(w.message) for w in avisos],
        fin_entrenamiento=p['audit']['fin_train'],inicio_evaluacion=p['audit']['inicio_eval'])


def elegir(registros):
    validos=[r for r in registros if r['estado']=='completo']
    if not validos: raise ValueError('Ninguna configuración terminó; consultar eventos y errores.')
    minimo=min(r['mae'] for r in validos)
    equivalentes=[r for r in validos if r['mae']<=minimo*1.01]
    return min(equivalentes,key=lambda r:(r['cantidad'],r['std'],r['id']))


def evaluar_config(pliegues,cfg,carpeta,contexto):
    ident=clave(dict(contexto=contexto,cfg=cfg));ruta=carpeta/'experimentos'/f'{ident}.json'
    pendientes_previos=[]
    if ruta.exists():
        previo=json.loads(ruta.read_text(encoding='utf-8'))
        if previo['estado']=='completo': return previo
        if previo['estado']=='fallido' and 'PermissionError' not in previo.get('error',''): return previo
        guardar(carpeta/'intentos_previos'/f'{ident}-{uuid.uuid4().hex[:8]}.json',previo)
        pendientes_previos=previo.get('pliegues',[])
    inicio=time.perf_counter();r=dict(id=ident,contexto=contexto,configuracion=cfg,
        fecha=datetime.now(timezone.utc).isoformat(),estado='iniciado',pliegues=pendientes_previos,
        codigo_id=clave(firma_codigo()))
    guardar(ruta,r);evento(carpeta,'inicio_configuracion',id=ident,contexto=contexto,configuracion=cfg)
    try:
        for k,p in list(enumerate(pliegues))[len(pendientes_previos):]:
            progreso(carpeta,cfg['metodo'],contexto=contexto,pliegue_interno=k+1,pliegues_internos=len(pliegues),configuracion_actual=ident)
            r['pliegues'].append(dict(pliegue=k,**medir(p,cfg)));guardar(ruta,r)
        vals=[f['mae'] for f in r['pliegues']]
        r.update(estado='completo',mae=float(np.mean(vals)),std=float(np.std(vals)),
            cantidad=float(np.mean([f['cantidad'] for f in r['pliegues']])))
    except Exception:
        r.update(estado='fallido',error=traceback.format_exc())
    r['segundos']=time.perf_counter()-inicio;guardar(ruta,r)
    evento(carpeta,'fin_configuracion',id=ident,estado=r['estado'],segundos=r['segundos'])
    return r


def buscar(pliegues,metodo,carpeta,contexto):
    """Búsqueda amplia y refinada exclusivamente interna; todos los intentos persisten."""
    registros=[]
    inicio_busqueda=time.perf_counter()
    previstas=0
    def probar(cfg):
        r=evaluar_config(pliegues,cfg,carpeta,contexto);registros.append(r)
        progreso(carpeta,metodo,estado='ejecutando',contexto=contexto,
            configuraciones_previstas=previstas,configuraciones_completadas=sum(x['estado']=='completo' for x in registros),
            fallidas=sum(x['estado']=='fallido' for x in registros),
            segundos_busqueda=time.perf_counter()-inicio_busqueda,
            mejor_mae_interno=min((x['mae'] for x in registros if x['estado']=='completo'),default=None),
            resultados=f'experimentos/{r["id"]}.json')
        return r
    if metodo not in METODOS: raise ValueError('Método no disponible en este motor de selección individual.')
    cantidades=sorted(set(POLITICA['cantidades']+[len(p['candidatas']) for p in pliegues]))
    # Agrega el total realmente disponible: 120 nunca es un límite superior fijo.
    previstas=4*(len(POLITICA['alphas']) if metodo=='lasso' else len(cantidades))
    for escala in POLITICA['escaladores']:
        for transformacion in POLITICA['transformaciones']:
            for valor in (POLITICA['alphas'] if metodo=='lasso' else cantidades):
                probar(dict(metodo=metodo,escalador=escala,transformacion=transformacion,
                    **({'alpha':valor} if metodo=='lasso' else {'k':valor})))
    mejor=elegir(registros);cfg=mejor['configuracion']
    if metodo=='lasso':
        refinados=np.geomspace(max(cfg['alpha']/10,1e-5),cfg['alpha']*10,9)
    else:
        k=cfg['k'];radio=max(3,int(np.ceil(k*.25)))
        refinados=range(max(1,k-radio),min(max(cantidades),k+radio)+1)
    vistos={clave(r['configuracion']) for r in registros}
    previstas=len(registros)+sum(clave(dict(cfg,**({'alpha':float(v)} if metodo=='lasso' else {'k':int(v)}))) not in vistos for v in refinados)
    for valor in refinados:
        nueva=dict(cfg,**({'alpha':float(valor)} if metodo=='lasso' else {'k':int(valor)}))
        if clave(nueva) not in vistos: probar(nueva)
    return elegir(registros)


def exportar(carpeta):
    experimentos=[];pliegues=[]
    for ruta in sorted((carpeta/'experimentos').glob('*.json')):
        r=json.loads(ruta.read_text(encoding='utf-8'));cfg=r['configuracion']
        experimentos.append(dict(id=r['id'],contexto=r['contexto'],metodo=cfg['metodo'],
            configuracion=json.dumps(cfg,ensure_ascii=False),fecha=r['fecha'],estado=r['estado'],
            mae=r.get('mae'),std=r.get('std'),cantidad=r.get('cantidad'),segundos=r.get('segundos'),error=r.get('error')))
        for f in r['pliegues']:
            pliegues.append(dict(id=r['id'],contexto=r['contexto'],metodo=cfg['metodo'],
                **{k:f[k] for k in ['pliegue','mae','cantidad','cantidad_modelo','n_candidatas','segundos']},
                directas=len(f['seleccion']['directas']),jerarquia=len(f['seleccion']['por_jerarquia']),
                seleccionadas=json.dumps(f['seleccionadas'],ensure_ascii=False)))
    pd.DataFrame(experimentos).to_csv(carpeta/'experimentos.csv',index=False)
    pd.DataFrame(pliegues).to_csv(carpeta/'resultados_pliegues.csv',index=False)


def ejecutar(carpeta=None,sin_cache=False):
    """Cuatro pliegues externos, tres internos y ajuste final solo en el 70% autorizado."""
    if sys.version_info[:2]!=(3,12): raise RuntimeError('La consigna exige Python 3.12; activar cervecera_env.')
    global FORWARD_LIMITE
    nueva=carpeta is None
    carpeta=Path(carpeta).resolve() if carpeta else RAIZ/'results/seleccion'/('ejecucion-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    carpeta.mkdir(parents=True,exist_ok=not nueva)
    firma=firma_codigo()
    if not nueva:
        previa=json.loads((carpeta/'configuraciones.json').read_text(encoding='utf-8'))
        revision=carpeta/'revision_codigo.json'
        codigo_esperado=json.loads(revision.read_text(encoding='utf-8'))['codigo_corregido'] if revision.exists() else previa['codigo']
        if codigo_esperado!=firma or previa['politica']!=POLITICA: raise ValueError('Codigo/configuracion distintos: iniciar otra ejecucion.')
    else:
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=RAIZ,capture_output=True,text=True).stdout.strip()
        guardar(carpeta/'configuraciones.json',dict(politica=POLITICA,codigo=firma,commit=commit,
            codigo_sin_commit_sha256=hashlib.sha256(json.dumps(firma,sort_keys=True).encode()).hexdigest(),python=sys.version,plataforma=platform.platform(),
            dependencias={p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','openpyxl','threadpoolctl']},
            fuentes=json.loads((RAIZ/'data/checksums.json').read_text()),
            restauracion='fase23-20261113-152329-a7172f93'))
    if nueva:
        for nombre in firma:
            destino=carpeta/'codigo'/nombre;destino.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(RAIZ/nombre,destino)
    print(f'EJECUCION {carpeta}',flush=True);evento(carpeta,'inicio' if nueva else 'reanudacion')
    try:
        with threadpool_limits(limits=1):
            progreso(carpeta,'preparacion',estado='cargando_fuentes')
            train,datos=cargar_entrenamiento(sin_cache);guardar(carpeta/'datos.json',datos)
            externos=list(TimeSeriesSplit(4,gap=1).split(train));resultados={m:[] for m in METODOS}
            # Cada método termina sus cuatro evaluaciones antes del siguiente.
            contextos={}
            for o,(i,j) in enumerate(externos):
                prefijo=train.iloc[i].copy();prefijo.iloc[-1,prefijo.columns.get_loc('objetivo')]=np.nan
                inner=[preparar_pliegue(prefijo,a,b,carpeta,f'externo{o}_interno{k}')
                    for k,(a,b) in enumerate(TimeSeriesSplit(3,gap=1).split(prefijo))]
                outer=preparar_pliegue(train,i,j,carpeta,f'externo{o}')
                contextos[o]=(inner,outer)
            final_inner=[preparar_pliegue(train,a,b,carpeta,f'final_interno{k}')
                for k,(a,b) in enumerate(TimeSeriesSplit(3,gap=1).split(train))]
            final_p=preparar_pliegue(train,np.arange(len(train)),np.array([],dtype=int),carpeta,'final')
            progreso(carpeta,'preparacion',estado='completo',pliegues_principales_preparados=16)
            finales={};resumen=[]
            for m in METODOS:
                inicio=time.perf_counter();print(f'METODO {m}',flush=True)
                progreso(carpeta,m,estado='iniciado',inicio_utc=datetime.now(timezone.utc).isoformat(),siguiente=METODOS[METODOS.index(m)+1] if METODOS.index(m)+1<len(METODOS) else 'union')
                try:
                    usados=contextos
                    inner_final=final_inner
                    for o,(inner,outer) in usados.items():
                        mejor=buscar(inner,m,carpeta,f'{m}_externo{o}')
                        r=evaluar_config([outer],mejor['configuracion'],carpeta,f'{m}_evaluacion_externa{o}')
                        if r['estado']!='completo': raise RuntimeError(r.get('error'))
                        f=r['pliegues'][0];resultados[m].append(f)
                        print(f'{m} externo {o+1}/{len(usados)} MAE={f["mae"]:.3f}, variables={f["cantidad"]}',flush=True)
                    mejor=buscar(inner_final,m,carpeta,f'{m}_ajuste_final')
                    sel=seleccionar(final_p,mejor['configuracion'])
                    finales[m]=dict(configuracion=mejor['configuracion'],seleccion=sel,
                        seleccionadas=sel['seleccionadas'],id_busqueda=mejor['id'])
                    frec,pares=resumir_estabilidad(resultados[m])
                    frec.to_csv(carpeta/f'estabilidad_{m}.csv',index=False)
                    vals=[r['mae'] for r in resultados[m]]
                    resumen.append(dict(metodo=m,mae=float(np.mean(vals)),std=float(np.std(vals)),
                        cantidad=len(sel['seleccionadas']),cantidad_cv=float(np.mean([r['cantidad'] for r in resultados[m]])),
                        jaccard=float(pares.Jaccard.mean()),segundos=time.perf_counter()-inicio,
                        configuracion=json.dumps(mejor['configuracion']),estado='completo',
                        protocolo='complementario_3x2' if m=='forward' else 'principal_4x3'))
                    progreso(carpeta,m,estado='completo',segundos_etapa=time.perf_counter()-inicio,resultado=resumen[-1])
                    guardar(carpeta/'selecciones_por_metodo.json',finales)
                    pd.DataFrame(resumen).to_csv(carpeta/'resumen_resultados.csv',index=False)
                except Exception:
                    progreso(carpeta,m,estado='fallido',error=traceback.format_exc())
                    evento(carpeta,'metodo_fallido',metodo=m,error=traceback.format_exc())
                    if m!='forward': raise
                finally: exportar(carpeta)
            principales=[r for r in resumen if r['metodo']!='forward']
            minimo=min(r['mae'] for r in principales)
            equivalentes=[r for r in principales if r['mae']<=minimo*1.01]
            ganador=min(equivalentes,key=lambda r:(r['cantidad'], -r['jaccard'],r['std'],r['metodo']))
            seleccion=finales[ganador['metodo']]
            contrato=dict(ganador=ganador,**seleccion,reglas_limpieza=final_p['audit']['limpieza'],
                reglas_variables=final_p['audit']['ingenieria'],semilla=SEMILLA,datos=datos,
                codigo=firma,politica=POLITICA,
                integracion=dict(dataset='data/processed/dataset_final.csv',linaje='data/processed/data_lineage.json',
                    estado='pendiente modulo 2.4; no sobrescrito',sin_escalar=True,sin_transformacion=True,
                    invalidar_menos_24=True,objetivo='consumo_dia.shift(-1) por bloque',ultima_etiqueta_train_excluida=True))
            guardar(carpeta/'variables_seleccionadas.json',contrato)
            guardar(carpeta/'resultados_externos.json',resultados)
            # Dos ajustes sucesivos reales sobre un pliegue externo; no consulta test.
            determinismo=[]
            for m in finales:
                cfg=finales[m]['configuracion'];p=contextos[0][1]
                r1=medir(p,cfg);r2=medir(p,cfg)
                determinismo.append(dict(metodo=m,mae_1=r1['mae'],mae_2=r2['mae'],
                    mismas_variables=r1['seleccionadas']==r2['seleccionadas'],
                    predicciones_identicas=r1['predicciones']==r2['predicciones']))
            guardar(carpeta/'determinismo.json',determinismo)
            assert all(r['mismas_variables'] and r['predicciones_identicas'] for r in determinismo)
            columnas=seleccion['seleccionadas'];corr=final_p['X'][columnas].corr(method='spearman')
            corr.to_csv(carpeta/'correlaciones_finales.csv')
            guardar(carpeta/'estado.json',dict(estado='completo',ganador=ganador,
                fecha=datetime.now(timezone.utc).isoformat(),test_utilizado=False))
            guardar(carpeta/'checksums_artefactos.json',{p.relative_to(carpeta).as_posix():huella(p) for p in carpeta.rglob('*') if p.is_file() and p.name!='checksums_artefactos.json'})
            print('GANADOR',ganador,flush=True)
            return carpeta
    except BaseException:
        evento(carpeta,'interrupcion_o_error',error=traceback.format_exc())
        guardar(carpeta/'estado.json',dict(estado='interrumpido_o_fallido',error=traceback.format_exc()))
        exportar(carpeta);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reanudar',type=Path)
    parser.add_argument('--sin-cache',action='store_true')
    args=parser.parse_args();ejecutar(args.reanudar,args.sin_cache)
