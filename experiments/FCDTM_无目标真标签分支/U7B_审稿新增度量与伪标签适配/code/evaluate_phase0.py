"""Separate post-seal cache-only evaluation; no predictor imports this module."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,math
import numpy as np
B=Path(__file__).resolve().parents[1];O=B/'artifacts';A=B.parent/'U7A_原论文方法无标签正式对比/artifacts'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def save(p,r):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def ident(r):return tuple(r[k] for k in ['cls','src','tgt','kind','budget','seed','repeat'])
def rank(x):
 o=np.argsort(x,kind='stable');r=np.empty(len(x));i=0
 while i<len(o):
  j=i+1
  while j<len(o) and x[o[j]]==x[o[i]]:j+=1
  r[o[i:j]]=(i+j-1)/2;i=j
 return r

def corr(x,y):
 x=np.array(x);y=np.array(y)
 if len(x)<4 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan,np.nan
 c=lambda a,b:float(np.corrcoef((a-a.min())/np.ptp(a),(b-b.min())/np.ptp(b))[0,1])
 return c(x,y),c(rank(x),rank(y))
def mean(v):
 a=[float(x) for x in v if math.isfinite(float(x))];return float(np.mean(a)) if a else float('nan')
seal=json.loads((O/'seal_phase0.json').read_text())
for p,h in seal.items():assert sha(B/p)==h
pred={ident(r):r for r in read(O/'predictions_phase0.csv')};truth=read(A/'逐项评价.csv');m=json.loads((A/'manifest.json').read_text());merged=[];groups=defaultdict(list)
for r in truth:
 q={**r,**{k:pred[ident(r)][k] for k in ['Hscore_PL','NCE_Pred_PL']}};merged.append(q)
 if r['kind']=='batch4' and r['common']=='True':groups[r['cls'],r['src'],r['tgt'],r['seed']].append(q)
cr=[]
for key,rs in groups.items():
 for method in ['Hscore_PL','NCE_Pred_PL']:
  pr,sr=corr([-float(r[method]) for r in rs],[float(r['true_AD']) for r in rs])
  cr.append(dict(cls=key[0],src=key[1],tgt=key[2],seed=key[3],scenario=rs[0]['scenario'],scope='common',method=method,K=len(rs),Pearson=pr,Spearman=sr))
summary=[]
for cohort,keys in m['cohorts'].items():
 keys={tuple(str(z) for z in k) for k in keys}
 for scenario in ['all','sensor_GBA','sensor_XJ','region_S2','region_L8']:
  for method in ['Hscore_PL','NCE_Pred_PL']:
   rr=[r for r in cr if (r['cls'],r['src'],r['tgt']) in keys and r['method']==method and (scenario=='all' or r['scenario']==scenario)];by=defaultdict(list)
   for r in rr:by[r['seed']].append(r)
   f=[];rho=[]
   for seed,values in by.items():
    v=[r['Pearson'] for r in values if np.isfinite(r['Pearson'])];f.append(float(np.tanh(mean(np.arctanh(np.clip(v,-1+1e-7,1-1e-7))))) if v else np.nan);rho.append(mean(r['Spearman'] for r in values))
   summary.append(dict(cohort=cohort,scenario=scenario,method=method,Fisher_r=mean(f),Fisher_r_seed_sd=float(np.std([v for v in f if np.isfinite(v)])) if any(np.isfinite(f)) else np.nan,rho=mean(rho),rho_seed_sd=float(np.std([v for v in rho if np.isfinite(v)])) if any(np.isfinite(rho)) else np.nan,finite_correlations=sum(np.isfinite(r['Pearson']) for r in rr),common_points=sum(r['K'] for r in rr),status='constant_self_PL' if method=='NCE_Pred_PL' else 'evaluated_phase0'))
save(O/'逐项评价_phase0.csv',merged);save(O/'逐任务相关_phase0.csv',cr);save(O/'相关汇总_phase0.csv',summary)
all_primary=['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL']
# Source original five values stay exactly unchanged; phase 0 has no new support exclusions.
choices=[];selection=[]
for cohort,keys in m['cohorts'].items():
 keys={tuple(str(z) for z in k) for k in keys};pools=defaultdict(list)
 for r in merged:
  if r['kind']=='pool' and int(r['budget'])<40 and (r['cls'],r['src'],r['tgt']) in keys:pools[r['cls'],r['tgt'],r['budget'],r['repeat']].append(r)
 for key,rs in pools.items():
  if len(rs)<2 or not all(np.isfinite(float(r['true_F1'])) and all(np.isfinite(float(r[k])) or (k=='DS_pixel' and np.isinf(float(r[k]))) for k in all_primary) for r in rs):continue
  y=np.array([float(r['true_F1']) for r in rs])
  for method in ['Hscore_PL','NCE_Pred_PL']:
   x=np.array([-float(r[method]) for r in rs]);lo=x.min();scale=max(float(np.abs(x).max()),np.finfo(float).tiny);w=np.abs(x-lo)<=1e-12*scale
   choices.append(dict(cohort=cohort,cls=key[0],tgt=key[1],budget=key[2],repeat=key[3],method=method,regret=float(y.max()-y[w].mean()),Top1=float(np.isclose(y,y.max(),atol=1e-12,rtol=0)[w].mean()),worst_tie_loss=float(y.max()-y[w].min())))
 for method in ['Hscore_PL','NCE_Pred_PL']:
  cc=[r for r in choices if r['cohort']==cohort and r['method']==method];d=dict(cohort=cohort,method=method,common_configs=len(cc))
  for col in ['regret','Top1']:
   budgets=defaultdict(list)
   for r in cc:budgets[r['cls'],r['tgt'],r['budget']].append(r[col])
   tasks=defaultdict(list)
   for k,v in budgets.items():tasks[k[:2]].append(mean(v))
   d[col]=mean(mean(v) for v in tasks.values());d['target_class_groups']=len(tasks)
  d['max_loss']=max(r['regret'] for r in cc);d['worst_tie_loss']=max(r['worst_tie_loss'] for r in cc);selection.append(d)
save(O/'模型选择_phase0.csv',choices);save(O/'模型选择宏汇总_phase0.csv',selection)
(O/'evaluation_phase0.json').write_text(json.dumps(dict(phase0_seal_verified=True,target_truth_only_postseal=True,truth_source_sha256=sha(A/'逐项评价.csv'),predictor_sha256=sha(B/'code/score_moments.py'),evaluated_records=len(merged),source_U7A_protocol_unchanged=True),indent=2))
print('D42 phase 0 summary:')
for r in summary:
 if r['cohort']=='D42' and r['scenario']=='all':print(r)
for r in selection:
 if r['cohort']=='D42':print(r)
