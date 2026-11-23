# -*- coding: utf-8 -*-
"""Seal U8 score-only outputs, then join independent historic evaluation."""
from common import *
import numpy as np,math
from collections import defaultdict
sys.path.insert(0,str(BR/'U7R_跨迁移任务排序与置信区间/code'))
from ranking_stats import coefficients,block_weights,interval
m=json.loads((OLD/'artifacts/manifest_full.json').read_text());keys=['cls','src','tgt','kind','budget','seed','repeat'];key=lambda r:tuple(str(r[k]) for k in keys)
rows=[];audits=[];ots={};validation=[]
for mi in m['models']:
 stem=f'{mi["cls"]}_{mi["src"]}'
 for folder in ['workers','ot_workers']:
  a=json.loads((O/folder/f'{stem}_audit.json').read_text());assert a['target_reference_reads']==0 and not a['unexpected_denials'];audits.append(a)
  for p,h in a['inputs'].items():assert sha(p)==h,p
  for name,h in a['code_sha'].items():assert sha(B/'code'/name)==h,name
 for r in read(O/'ot_workers'/f'{stem}.csv'):assert key(r) not in ots;ots[key(r)]=r
 rows+=read(O/'workers'/f'{stem}.csv');validation+=read(O/'ot_workers'/f'{stem}_source_validation.json') if False else json.loads((O/'ot_workers'/f'{stem}_source_validation.json').read_text())
assert len(rows)==3300 and len(ots)==3300 and len({key(r) for r in rows})==3300
for r in rows:
 q=ots[key(r)];assert r['indices']==q['indices'];r['OTCE_PL_SrcCal']=q['OTCE_PL_SrcCal']
save(O/'predictions_sealed.csv',rows);dump(O/'score_seal.json',{'artifacts/predictions_sealed.csv':sha(O/'predictions_sealed.csv'),**{'code/'+p.name:sha(p) for p in (B/'code').glob('*.py')}})
dump(O/'score_verification.json',dict(status='passed',models=25,tasks=50,score_rows=len(rows),audits=len(audits),target_reference_reads=0,original_weighted_max_error=max(a.get('weighted_score_max_error',0) for a in audits),OTCE_numerical_scheme='Q1024 source-only quantized feasible product lift, approximate',mmd='exact unbiased block RBF',source_validation_cases=len(validation),median_relative_OT_cost_error=float(np.median([v['relative_W_error'] for v in validation])),max_relative_OT_cost_error=float(max(v['relative_W_error'] for v in validation)),median_absolute_CE_error=float(np.median([v['absolute_CE_error'] for v in validation]))))
# Evaluation only after sealing. Reference masks are never reopened.
truth={key(r):r for r in read(OLD/'artifacts/逐项评价_full.csv') if r['panel']=='full2048'}
ALL=METHODS+ABL
for r in rows:
 t=truth[key(r)];assert r['indices']==t['indices']
 for k in ALL:r[k]=float(r[k])
 for k in ['true_F1','true_AD','true_accuracy','scenario']:r[k]=t[k] if k=='scenario' else float(t[k])
 r['cls']=int(r['cls']);r['common']=bool(np.isfinite(r['true_AD']) and all(np.isfinite(r[k]) for k in METHODS))
save(O/'evaluated.csv',rows)
def cor(x,y):
 x=np.array(x,float);y=np.array(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<4 or np.ptp(x)==0 or np.ptp(y)==0:return len(x),np.nan,np.nan
 from scipy.stats import rankdata
 cc=lambda a,b:float(np.corrcoef((a-a.min())/np.ptp(a),(b-b.min())/np.ptp(b))[0,1])
 return len(x),cc(x,y),cc(rankdata(x),rankdata(y))
groups=defaultdict(list)
for r in rows:
 if r['kind']=='batch4':groups[r['cls'],r['src'],r['tgt'],r['seed']].append(r)
cr=[]
for key0,rr in groups.items():
 for scope in ['common','own']:
  sub=[r for r in rr if scope=='own' or r['common']]
  for k in ALL:
   n,pr,sr=cor([r[k]*(1 if k in BAD else -1) for r in sub],[r['true_AD'] for r in sub]);cr.append(dict(cls=key0[0],src=key0[1],tgt=key0[2],seed=key0[3],scenario=rr[0]['scenario'],scope=scope,method=k,K=n,Pearson=pr,Spearman=sr))
save(O/'batch_correlations.csv',cr);summary=[];scens=['all','region_S2','region_L8','sensor_GBA','sensor_XJ']
for cohort,kk in m['cohorts'].items():
 kk={tuple(k) for k in kk}
 for scen in scens:
  for scope in ['common','own']:
   for method in ALL:
    a=[r for r in cr if (r['cls'],r['src'],r['tgt']) in kk and r['scope']==scope and r['method']==method and (scen=='all' or r['scenario']==scen)];ff=[];sp=[]
    for seed in sorted({r['seed'] for r in a}):
     v=[r for r in a if r['seed']==seed];pp=[r['Pearson'] for r in v if np.isfinite(r['Pearson'])];rho=[r['Spearman'] for r in v if np.isfinite(r['Spearman'])];ff.append(float(np.tanh(np.mean(np.arctanh(np.clip(pp,-1+1e-7,1-1e-7))))) if pp else np.nan);sp.append(float(np.mean(rho)) if rho else np.nan)
    summary.append(dict(cohort=cohort,scenario=scen,scope=scope,method=method,Fisher_r=float(np.nanmean(ff)),rho=float(np.nanmean(sp)),seed_sd=float(np.nanstd(ff)),defined_correlations=sum(np.isfinite(r['Pearson']) for r in a),points=sum(r['K'] for r in a)))
save(O/'summary.csv',summary)
# All method ordering; same block resamples as U7R, with U8 common support.
task=[r for r in rows if r['kind']=='pool' and int(r['budget'])==40];save(O/'task_data.csv',task);out=[]
for ci,cohort in enumerate(['D42','P37','E50']):
 kk={tuple(k) for k in m['cohorts'][cohort]}
 for si,scen in enumerate(scens):
  nominal=[r for r in task if (r['cls'],r['src'],r['tgt']) in kk and (scen=='all' or r['scenario']==scen)]
  for oi,scope in enumerate(['common','own']):
   gs={kind:[str(r['cls']) if kind=='class' else str(r['cls'])+'__'+r['src'] for r in nominal] for kind in ['class','source_model']};weights={kind:block_weights(g,5000,20261123+ci*100000+si*10000+oi*1000+j*100) for j,(kind,g) in enumerate(gs.items())}
   for method in ALL:
    valid=np.array([np.isfinite(r[method]) and np.isfinite(r['true_AD']) and (scope=='own' or r['common']) for r in nominal]);q=[r for r,k in zip(nominal,valid) if k];x=np.array([r[method]*(1 if method in BAD else -1) for r in q]);y=np.array([r['true_AD'] for r in q]);tau,rho=coefficients(x,y)
    for kind,(w,blocks) in weights.items():
     usable=len({g for g,k in zip(gs[kind],valid) if k});tt,ss=coefficients(x,y,w[:,valid]);r=dict(cohort=cohort,scenario=scen,scope=scope,method=method,bootstrap_block=kind,valid_tasks=len(q),nominal_tasks=len(nominal),tau_b=float(tau[0]),Spearman=float(rho[0]))
     for name,vals in [('tau',tt),('rho',ss)]:
      lo,hi,n,st=interval(vals,usable,5000);r.update({name+'_lo':lo,name+'_hi':hi,name+'_valid_bootstrap':n,name+'_CI_status':st})
     out.append(r)
  print('RANKED',cohort,scen,flush=True)
save(O/'ranking.csv',out);dump(O/'evaluation.json',dict(status='complete',records=len(rows),batch_correlations=len(cr),ranking_records=len(out),target_reference_cache_only_postseal=True,reference_cache_sha=sha(OLD/'artifacts/逐项评价_full.csv'),common_batch_records=sum(r['common'] for r in rows if r['kind']=='batch4')))
