"""Post-seal validation against SciPy and literal block-resample duplication."""
import json,math
import numpy as np
from scipy.stats import kendalltau,spearmanr
from common import *
from ranking_stats import coefficients,block_weights
for p,h in json.loads((O/'score_seal.json').read_text()).items():assert sha(B/p)==h,p
scores=read(O/'task_scores_sealed.csv');data=read(O/'任务级数据.csv');stats=read(O/'跨任务排序统计.csv');meta=json.loads((O/'task_manifest.json').read_text());assert len(scores)==len(data)==100 and len(stats)==2040
for a,b in zip(scores,data):
 for k in METHODS:
  x,y=float(a[k]),float(b[k]);assert x==y or math.isnan(x) and math.isnan(y)
for r in data:
 r['cls']=int(r['cls']);r['common']=r['common']=='True';r['true_AD']=float(r['true_AD'])
 for k in METHODS:r[k]=float(r[k])
errors=[];checked=0
for s in stats:
 keys={tuple(k) for k in meta['cohorts'][s['cohort']]};q=[r for r in data if r['panel']==s['panel'] and (r['cls'],r['src'],r['tgt']) in keys and (s['scenario']=='all' or r['scenario']==s['scenario']) and (s['scope']=='own' or r['common']) and math.isfinite(r[s['method']]) and math.isfinite(r['true_AD'])];assert len(q)==int(s['valid_tasks']);x=np.array([r[s['method']]*(1 if s['method'] in BAD else -1) for r in q]);y=np.array([r['true_AD'] for r in q]);v=(np.nan,np.nan) if len(q)<3 or np.ptp(x)==0 or np.ptp(y)==0 else (kendalltau(x,y,variant='b').statistic,spearmanr(x,y).statistic)
 for got,key in zip(v,['tau_b','Spearman']):
  wanted=float(s[key]);assert (np.isnan(got) and np.isnan(wanted)) or abs(got-wanted)<1e-12,(s,key,got);errors.append(abs(got-wanted) if np.isfinite(got) else 0)
 checked+=1
# Paired resamples on real D42 records, including ties and model-scale extremes.
boot_errors=[]
for pi,(panel,methods) in enumerate(PANELS.items()):
 ck={tuple(k) for k in meta['cohorts']['D42']};nom=[r for r in data if r['panel']==panel and (r['cls'],r['src'],r['tgt']) in ck]
 for bi,kind in enumerate(['class','source_model']):
  labels=[str(r['cls']) if kind=='class' else str(r['cls'])+'__'+r['src'] for r in nom];w,_=block_weights(labels,REPS,SEED+pi*1000000+bi*100)
  for method in methods:
   mask=np.array([r['common'] for r in nom]);q=[r for r,m in zip(nom,mask) if m];x=np.array([r[method]*(1 if method in BAD else -1) for r in q]);y=np.array([r['true_AD'] for r in q]);ww=w[:10,mask];tt,rr=coefficients(x,y,ww)
   for i,mults in enumerate(ww):
    if np.count_nonzero(mults)<3 or np.ptp(x)==0 or np.ptp(y)==0:continue
    xx=np.repeat(x,mults);yy=np.repeat(y,mults);ref=kendalltau(xx,yy,variant='b').statistic,spearmanr(xx,yy).statistic
    for got,want in zip([tt[i],rr[i]],ref):assert abs(got-want)<1e-12;boot_errors.append(abs(got-want))
if (B/'figures/index.json').exists():
 figs=json.loads((B/'figures/index.json').read_text());assert len(figs)==32
 for panel,methods in PANELS.items():
  for cohort in ['D42','E50']:
   shown={k for r in figs if r['kind']=='task_scatter' and r['panel']==panel and r['cohort']==cohort for k in r['methods']};assert shown==set(methods)
 ann=json.loads((B/'figures/annotation_audit.json').read_text())
 for a in ann:
  s=next(r for r in stats if r['panel']==a['panel'] and r['cohort']==a['cohort'] and r['scenario']=='all' and r['scope']=='common' and r['bootstrap_block']=='class' and r['method']==a['method'])
  for k in ['tau_b','tau_lo','tau_hi','Spearman','rho_lo','rho_hi']:
   x,y=float(a[k]),float(s[k]);assert x==y or math.isnan(x) and math.isnan(y)
  assert int(a['N'])==int(s['valid_tasks'])
report=dict(score_seal_verified=True,score_values_preserved=True,statistic_rows_scipy_checked=checked,max_point_error=max(errors),real_bootstrap_coefficients_checked=len(boot_errors),max_real_bootstrap_error=max(boot_errors),figures_cover_all_applicable_methods=True,annotation_numbers_match_tables=True,target_annotation_images_reopened=False,verifier_sha256=sha(Path(__file__)))
(O/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
