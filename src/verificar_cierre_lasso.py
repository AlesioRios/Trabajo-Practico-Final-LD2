"""Verificación final de evidencia, sin ajustar ni predecir sobre test."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from packaging.requirements import Requirement

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--historico',action='store_true',help='verificar los archivos del commit_base contra las firmas historicas')
parser.add_argument('--sin-guardar',action='store_true',help='mostrar los controles sin escribir verificacion_final.json')
args=parser.parse_args()

r=Path.cwd()
s=r/'results/lasso_final'
def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta=leer(s/'entrenamiento.json');ev=leer(s/'evaluacion_test.json');prep=leer(s/'preparacion_cierre.json')
registry=leer(r/'models/model_registry.json');entrada=registry[meta['version']]
modelo=joblib.load(r/meta['path'])
pred=pd.read_csv(s/'predicciones_test.csv',float_precision='round_trip')
coef=pd.read_csv(s/'coeficientes.csv',float_precision='round_trip').set_index('variable').loc[meta['columnas']]
log=list(csv.DictReader((r/'results/experiment_logs.csv').open(encoding='utf-8')))
recientes=log[prep['registros_previos']:]
checks={
 'modelo_conserva_sha256':sha(r/meta['path'])==meta['modelo_sha256']==ev['modelo_sha256'],
 'predicciones_conservan_sha256':sha(s/'predicciones_test.csv')==ev['predicciones_sha256'],
 'catorce_variables_en_orden':list(modelo.feature_names_in_)==meta['columnas']==modelo.cierre_metadata_['columnas'],
 'parametros_fijados':all(modelo.named_steps['regressor'].get_params()[k]==v for k,v in meta['hiperparametros'].items()),
 'coeficientes_archivados_coinciden':np.array_equal(modelo.named_steps['regressor'].coef_,coef.coeficiente_estandarizado),
 'registry_modelo_y_validacion_coinciden':entrada['modelo_sha256']==meta['modelo_sha256'] and entrada['metricas_validacion_original']==meta['metricas_validacion_original'],
 'registry_test_coincide':entrada['metricas_test_local']==ev['metricas'] and entrada['filas_test']==len(pred),
 'log_historico_intacto':hashlib.sha256((r/'results/experiment_logs.csv').read_bytes()[:prep['log_previo_bytes']]).hexdigest()==prep['log_previo_sha256'],
 'cuatro_eventos_nuevos_distintos':len(recientes)==4 and recientes[0]['tramo']=='entrenamiento' and all(x['tramo']=='prueba' for x in recientes[1:]),
 'mismo_soporte_test_tres_modelos':all(int(x['filas'])==len(pred)==ev['filas_evaluadas'] for x in recientes[1:]),
 'indices_ajuste_y_test_separados':max(meta['indices_ajuste'])<min(ev['indices_evaluados']) and not set(meta['indices_ajuste'])&set(ev['indices_evaluados']),
 'congelado_antes_del_test':datetime.fromisoformat(meta['fin_entrenamiento'])<datetime.fromisoformat(leer(s/'test_consumido.json')['inicio']),
 'un_test_registrado':ev['evaluaciones_predict_test']==1 and leer(s/'test_consumido.json')['estado']=='completo_no_repetir',
 'python312':meta['python'].startswith('3.12.'),
 **meta['reproducibilidad'],
}
for ruta,firma in prep['archivos_protegidos_sha256'].items():
 if args.historico:
  contenido=subprocess.check_output(['git','show',prep['commit_base']+':'+ruta])
  # Las firmas del cierre incluyen archivos de Windows con CRLF; Git puede conservar LF.
  historico_crlf=contenido.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
  checks['preservado_historico:'+ruta]=firma in {hashlib.sha256(contenido).hexdigest(),hashlib.sha256(historico_crlf).hexdigest()}
 else:
  checks['preservado:'+ruta]=sha(r/ruta)==firma
code=subprocess.check_output(['git','show',meta['commit']+':src/train_model.py'])
checks['codigo_entrenado_recuperable']=hashlib.sha256(code).hexdigest()==sha(r/'src/train_model.py')
checks['commit_registry_artefacto']=meta['commit']==entrada['commit']==modelo.cierre_metadata_['commit_entrenamiento']
checks['dataset_congelado']=sha(r/'data/processed/dataset_final.csv')==meta['dataset_sha256']
metricas={}
y=pred.real.to_numpy();total=np.square(y-y.mean()).sum()
for nombre,campo in [('lasso','predicho'),('persistencia','persistencia'),('media_movil','media_movil')]:
 err=y-pred[campo].to_numpy()
 mm={'mae':float(np.abs(err).mean()),'rmse':float(np.sqrt(np.square(err).mean())),'r2':float(1-np.square(err).sum()/total)}
 metricas[nombre]=mm
 checks['aritmetica:'+nombre]=all(np.isclose(v,ev['metricas'][nombre][k],rtol=1e-12,atol=1e-9) for k,v in mm.items())
 fila=next(x for x in recientes[1:] if x['modelo']==nombre+'_v'+meta['version'])
 checks['log_metricas:'+nombre]=all(float(fila[k])==ev['metricas'][nombre][k] for k in mm)
faltantes=[]
for line in (r/'requirements.txt').read_text().splitlines():
 if not line.strip() or line.lstrip().startswith('#'):continue
 req=Requirement(line)
 if req.marker and not req.marker.evaluate():continue
 try:
  installed=importlib.metadata.version(req.name)
  if installed not in req.specifier:faltantes.append({'paquete':req.name,'version':installed,'esperado':str(req.specifier)})
 except importlib.metadata.PackageNotFoundError:faltantes.append({'paquete':req.name,'version':None})
checks['dependencias_congeladas_coinciden']=not faltantes
nb=leer(r/'notebooks/modelado.ipynb');celdas=[c for c in nb['cells'] if c['cell_type']=='code']
checks['notebook_ejecutado_de_cero']=all(c['execution_count']==i for i,c in enumerate(celdas,1)) and not any(o.get('output_type')=='error' for c in celdas for o in c['outputs'])
xml=ET.parse(s/'pruebas.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml)
tests={k:sum(int(t.attrib.get(k,0)) for t in suites) for k in ['tests','failures','errors','skipped']}
tests['aprobadas']=tests['tests']-tests['failures']-tests['errors']-tests['skipped']
tests['motivos_omitidas']=[x.attrib for x in xml.iter('skipped')]
checks['suite_completa_sin_fallos']=tests['failures']==tests['errors']==0
resultado={'modo_auditoria':'historico' if args.historico else 'estricto','controles':{k:bool(v) for k,v in checks.items()},'todos_los_controles':all(checks.values()),
 'pruebas':tests,'dependencias_discrepantes':faltantes,'modelo_sha256':meta['modelo_sha256'],'commit_entrenamiento':meta['commit'],
 'version':meta['version'],'tag_anotado_previsto':'v'+meta['version'],'metricas_auditadas':metricas,
 'limite_fuga':'Ausencia de test en ajuste y fronteras verificadas; disponibilidad operativa y fechas físicas no certificadas.',
 'fase4':'Se verifica src/predict.py del commit_base.' if args.historico else 'Se exige preservar src/predict.py del cierre en la copia de trabajo.',
 'revision_visual':'Figuras de coeficientes y residuos inspeccionadas: textos legibles y sin recortes.',
 'faltantes_consigna':'Test oculto y Fase 4 fuera de alcance; conclusiones generales y PR de integración del equipo pendientes.'}
if not args.sin_guardar:
 (s/'verificacion_final.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:resultado[k] for k in ['modo_auditoria','todos_los_controles','pruebas','dependencias_discrepantes']},ensure_ascii=True,indent=2))
print('Controles fallidos:',[k for k,v in checks.items() if not v])
assert all(checks.values())
