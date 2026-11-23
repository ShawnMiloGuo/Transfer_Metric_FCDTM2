"""Independent task-ranking evaluator; uses only post-seal U7B evaluation cache."""
import sys,json,math,os
import numpy as np
from common import *
from ranking_stats import coefficients,block_weights,interval

def main():
 assert (O/'score_seal.json').exists()
 for p,h in json.loads((O/'score_seal.json').read_text()).items():assert sha(B/p)==h,p
 # Raw source/target annotation files are forbidden even in this reuse evaluator.
 def guard(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes)) and '/annotations/' in os.fsdecode(a[0]):raise PermissionError('No annotation re-reading in U7R')
 sys.addaudithook(guard)
 meta=json.loads((O/'task_manifest.json').read_text());rr=read(O/'task_scores_sealed.csv');ev=read(OLD/'逐项评价_full.csv');truth={}
 for r in ev:
  if r['kind']=='pool' and int(r['budget'])==40:truth[r['panel'],int(r['cls']),r['src'],r['tgt']]=r
 for r in rr:
  r['cls']=int(r['cls']);v=truth[r['panel'],r['cls'],r['src'],r['tgt']];assert r['indices']==v['indices'] and float(r['source_F1'])==float(v['source_F1'])
  for k in PANELS[r['panel']]:
   a,b=float(r[k]),float(v[k]);assert a==b or math.isnan(a) and math.isnan(b),k
  r['true_F1']=float(v['true_F1']);r['true_AD']=float(v['true_AD']);assert (not math.isfinite(r['true_AD'])) or abs(float(r['source_F1'])-r['true_F1']-r['true_AD'])<1e-12
  for k in METHODS:r[k]=float(r[k])
  r['common']=r['score_common']=='True' and math.isfinite(r['true_AD'])
 save(O/'任务级数据.csv',rr);out=[]
 for pi,(panel,methods) in enumerate(PANELS.items()):
  for ci,cohort in enumerate(COHORTS):
   keys={tuple(k) for k in meta['cohorts'][cohort]}
   for si,scen in enumerate(SCENARIOS):
    nominal=[r for r in rr if r['panel']==panel and (r['cls'],r['src'],r['tgt']) in keys and (scen=='all' or r['scenario']==scen)]
    for oi,scope in enumerate(['common','own']):
     groups={kind:([str(r['cls']) if kind=='class' else str(r['cls'])+'__'+r['src'] for r in nominal]) for kind in ['class','source_model']}
     weights={kind:block_weights(g,REPS,SEED+pi*1000000+ci*100000+si*10000+oi*1000+j*100) for j,(kind,g) in enumerate(groups.items())}
     for method in methods:
      valid=np.array([math.isfinite(r[method]) and math.isfinite(r['true_AD']) and (scope=='own' or r['common']) for r in nominal]);q=[r for r,k in zip(nominal,valid) if k];x=np.array([r[method]*(1 if method in BAD else -1) for r in q]);y=np.array([r['true_AD'] for r in q]);tau,rho=coefficients(x,y);status='defined' if np.isfinite(tau[0]) else ('insufficient_tasks' if len(q)<3 else 'constant_score' if np.ptp(x)==0 else 'constant_AD')
      for kind,(w,allblocks) in weights.items():
       usable=len({g for g,k in zip(groups[kind],valid) if k});tt,ss=coefficients(x,y,w[:,valid]);row=dict(panel=panel,cohort=cohort,scenario=scen,scope=scope,method=method,bootstrap_block=kind,nominal_tasks=len(nominal),valid_tasks=len(q),score_supported_tasks=sum(math.isfinite(r[method]) for r in nominal),AD_defined_tasks=sum(math.isfinite(r['true_AD']) for r in nominal),usable_blocks=usable,nominal_blocks=len(allblocks),tau_b=float(tau[0]),Spearman=float(rho[0]),point_status=status,bootstrap_replicates=REPS)
       for name,values in [('tau',tt),('rho',ss)]:
        lo,hi,n,st=interval(values,usable,REPS);row.update({name+'_lo':lo,name+'_hi':hi,name+'_valid_bootstrap':n,name+'_CI_status':st})
       out.append(row)
    print('RANKED',panel,cohort,scen,flush=True)
 save(O/'跨任务排序统计.csv',out)
 (O/'evaluation.json').write_text(json.dumps(dict(task_score_seal_verified=True,task_rows=len(rr),statistic_rows=len(out),target_annotation_reads=0,target_reference_cache_only_postseal=True,evaluation_cache_sha256=sha(OLD/'逐项评价_full.csv'),evaluating_code_sha256=sha(Path(__file__)),primary_interval='class-block percentile, 5000, descriptive conditional on historical domains',sensitivity_interval='source-model-block percentile, 5000',no_target_based_formula_or_direction_tuning=True),indent=2));print('EVALUATION COMPLETE',len(out),'statistics',flush=True)
if __name__=='__main__':main()
