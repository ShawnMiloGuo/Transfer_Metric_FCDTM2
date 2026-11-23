# -*- coding: utf-8 -*-
from common import *
import numpy as np,math
from scipy.stats import pearsonr,spearmanr,kendalltau
from collections import defaultdict
from ranking_stats import coefficients,block_weights,interval
for p,h in json.loads((O/'score_seal.json').read_text()).items():assert sha(B/p)==h,p
# Reference labels are never reopened by this evaluator, only old post-seal AD.
def guard(e,a):
 if e=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)) and '/annotations/' in os.fsdecode(a[0]):raise PermissionError('U8 evaluator uses only post-seal cache')
sys.addaudithook(guard)
rows=read(O/'predictions_sealed.csv');old=read(OLD/'artifacts/逐项评价_full.csv');keys=['cls','src','tgt','panel','kind','budget','seed','repeat'];uid=lambda r:tuple(str(r[k]) for k in keys);lookup={uid(r):r for r in old if r['panel']=='full2048'};m=json.loads((O/'manifest.json').read_text());COHORTS=['D42','P37','E50'];SCENES=['all','region_S2','region_L8','sensor_GBA','sensor_XJ'];FAMILIES={'hard':['FCDTM_HardSigned','FCDTM_HardAbs','FCDTM_HardPL'],'soft':['FCDTM_SoftSigned','FCDTM_SoftAbs','FCDTM_SoftPL']}
def orientation(r,k):return r[k] if k in BAD else -r[k]
def stats(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<4 or np.ptp(x)==0 or np.ptp(y)==0:return len(x),np.nan,np.nan
 return len(x),float(pearsonr((x-x.min())/np.ptp(x),(y-y.min())/np.ptp(y)).statistic),float(spearmanr(x,y).statistic)
def avg(v):
 x=[float(x) for x in v if np.isfinite(float(x))];return float(np.mean(x)) if x else np.nan
for r in rows:
 q=lookup[uid(r)];assert r['indices']==q['indices'] and float(r['source_F1'])==float(q['source_F1'])
 for k in METHODS+ABL+['source_F1']:r[k]=float(r[k])
 for k in ['cls','budget','seed','repeat']:r[k]=int(r[k])
 for k in ['true_F1','true_AD','true_accuracy']:r[k]=float(q[k])
 r['scenario']=q['scenario'];r['common']=np.isfinite(r['true_AD']) and all(np.isfinite(r[k]) for k in METHODS)
 for fam,methods in FAMILIES.items():r['common_'+fam]=np.isfinite(r['true_AD']) and all(np.isfinite(r[k]) for k in methods)
save(O/'逐项评价.csv',rows);batch=defaultdict(list)
for r in rows:
 if r['kind']=='batch4':batch[r['cls'],r['src'],r['tgt'],r['seed']].append(r)
cr=[]
for key,rr in batch.items():
 for scope,methods in [('common',METHODS),('own',METHODS+ABL),('ablation_hard',FAMILIES['hard']),('ablation_soft',FAMILIES['soft'])]:
  sub=[r for r in rr if scope=='own' or r['common'] if scope in ['common','own']] if scope in ['common','own'] else [r for r in rr if r['common'] and r['common_'+scope.split('_')[1]]]
  for method in methods:
   n,pr,sr=stats([orientation(r,method) for r in sub],[r['true_AD'] for r in sub]);cr.append(dict(cls=key[0],src=key[1],tgt=key[2],seed=key[3],scenario=rr[0]['scenario'],scope=scope,method=method,K=n,Pearson=pr,Spearman=sr,status='defined' if np.isfinite(pr) else 'undefined'))
save(O/'逐任务相关.csv',cr);summ=[]
for co in COHORTS:
 ck={tuple(k) for k in m['cohorts'][co]}
 for scen in SCENES:
  for scope in ['common','own','ablation_hard','ablation_soft']:
   methods=METHODS if scope=='common' else METHODS+ABL if scope=='own' else FAMILIES[scope.split('_')[1]]
   for method in methods:
    a=[r for r in cr if r['scope']==scope and r['method']==method and (r['cls'],r['src'],r['tgt']) in ck and (scen=='all' or r['scenario']==scen)];by=defaultdict(list)
    for r in a:by[r['seed']].append(r)
    fish=[];rho=[]
    for vals in by.values():
     zz=[np.arctanh(np.clip(r['Pearson'],-1+1e-7,1-1e-7)) for r in vals if np.isfinite(r['Pearson'])];fish.append(float(np.tanh(avg(zz))) if zz else np.nan);rho.append(avg(r['Spearman'] for r in vals))
    summ.append(dict(cohort=co,scenario=scen,scope=scope,method=method,Fisher_r=avg(fish),Fisher_r_seed_sd=float(np.nanstd(fish)) if fish else np.nan,rho=avg(rho),rho_seed_sd=float(np.nanstd(rho)) if rho else np.nan,defined_correlations=sum(np.isfinite(r['Pearson']) for r in a),nominal_correlations=len(a),finite_points=sum(r['K'] for r in a)))
save(O/'相关汇总.csv',summ)
# Directional table values preserve the original five-seed arithmetic convention.
dirrows=[]
for co in COHORTS:
 for cls,src,tgt in m['cohorts'][co]:
  for scope in ['common','ablation_hard','ablation_soft']:
   methods=METHODS if scope=='common' else FAMILIES[scope.split('_')[1]]
   for method in methods:
    a=[r for r in cr if r['cls']==cls and r['src']==src and r['tgt']==tgt and r['scope']==scope and r['method']==method];dirrows.append(dict(cohort=co,cls=cls,src=src,tgt=tgt,scenario=a[0]['scenario'],scope=scope,method=method,Pearson=avg(r['Pearson'] for r in a),Spearman=avg(r['Spearman'] for r in a),defined_seeds=sum(np.isfinite(r['Pearson']) for r in a)))
save(O/'方向任务结果.csv',dirrows)
# Full-pool task-level ordering, preserving the U7R interval design.
tasks=[r for r in rows if r['kind']=='pool' and r['budget']==40];assert len(tasks)==50;ranking=[]
for ci,co in enumerate(COHORTS):
 ck={tuple(k) for k in m['cohorts'][co]}
 for si,scen in enumerate(SCENES):
  nominal=[r for r in tasks if (r['cls'],r['src'],r['tgt']) in ck and (scen=='all' or r['scenario']==scen)]
  for oi,scope in enumerate(['common','own','ablation_hard','ablation_soft']):
   methods=METHODS if scope=='common' else METHODS+ABL if scope=='own' else FAMILIES[scope.split('_')[1]];groups={kind:[str(r['cls']) if kind=='class' else str(r['cls'])+'__'+r['src'] for r in nominal] for kind in ['class','source_model']};weights={kind:block_weights(g,5000,20261123+ci*100000+si*10000+oi*1000+j*100) for j,(kind,g) in enumerate(groups.items())}
   for method in methods:
    valid=np.array([np.isfinite(r[method]) and np.isfinite(r['true_AD']) and (scope=='own' or r['common'] if scope in ['common','own'] else r['common'] and r['common_'+scope.split('_')[1]]) for r in nominal]);q=[r for r,ok in zip(nominal,valid) if ok];x=np.array([orientation(r,method) for r in q]);y=np.array([r['true_AD'] for r in q]);tt,rr=coefficients(x,y);tau=float(tt[0]);rho=float(rr[0]);assert len(x)<3 or not np.isfinite(tau) or abs(tau-kendalltau(x,y).statistic)<1e-12
    for kind,(w,blocks) in weights.items():
     usable=len({g for g,k in zip(groups[kind],valid) if k});t,ss=coefficients(x,y,w[:,valid]);row=dict(cohort=co,scenario=scen,scope=scope,method=method,bootstrap_block=kind,nominal_tasks=len(nominal),valid_tasks=len(q),usable_blocks=usable,tau_b=tau,Spearman=rho)
     for label,values in [('tau',t),('rho',ss)]:lo,hi,n,status=interval(values,usable,5000);row.update({label+'_lo':lo,label+'_hi':hi,label+'_valid_bootstrap':n,label+'_CI_status':status})
     ranking.append(row)
  print('RANKED_U8',co,scen,flush=True)
save(O/'任务级数据.csv',tasks);save(O/'跨任务排序统计.csv',ranking)
# Source-model paired descriptive intervals for ablation, after averaging the
# task/seed z differences within each model; do not claim independent regions.
paired=[]
for co in COHORTS:
 ck={tuple(k) for k in m['cohorts'][co]}
 for fam,methods in FAMILIES.items():
  weighted='FCDTM_'+('HardPL' if fam=='hard' else 'SoftPL');rr=[r for r in cr if r['scope']=='ablation_'+fam and (r['cls'],r['src'],r['tgt']) in ck];look={(r['cls'],r['src'],r['tgt'],r['seed'],r['method']):r for r in rr}
  for scen in SCENES:
   for method in methods[:-1]:
    clusters=defaultdict(list)
    for r in rr:
     if r['method']!=method or (scen!='all' and r['scenario']!=scen):continue
     w=look[r['cls'],r['src'],r['tgt'],r['seed'],weighted]
     if np.isfinite(r['Pearson']) and np.isfinite(w['Pearson']):clusters[r['cls'],r['src']].append(np.arctanh(np.clip(w['Pearson'],-1+1e-7,1-1e-7))-np.arctanh(np.clip(r['Pearson'],-1+1e-7,1-1e-7)))
    values=np.array([np.mean(v) for v in clusters.values()]);rng=np.random.default_rng(20261123);boot=rng.choice(values,size=(5000,len(values)),replace=True).mean(1) if len(values) else np.array([np.nan]);paired.append(dict(cohort=co,scenario=scen,family=fam,comparator=method,model_clusters=len(values),mean_delta_z=avg(values),lo=float(np.nanquantile(boot,.025)),hi=float(np.nanquantile(boot,.975))))
save(O/'消融配对描述性区间.csv',paired)
choices=[];selection=[]
for co in COHORTS:
 ck={tuple(k) for k in m['cohorts'][co]};groups=defaultdict(list)
 for r in rows:
  if r['kind']=='pool' and r['budget']<40 and (r['cls'],r['src'],r['tgt']) in ck:groups[r['cls'],r['tgt'],r['budget'],r['repeat']].append(r)
 nominal=0;support=0
 for key,rs in groups.items():
  if len(rs)<2:continue
  nominal+=1;y=np.array([r['true_F1'] for r in rs]);joint=np.isfinite(y).all() and all(np.isfinite(orientation(r,k)) for r in rs for k in METHODS)
  if not joint:continue
  support+=1
  for method in METHODS:
   score=np.array([orientation(r,method) for r in rs]);best=score.min();scale=max(float(abs(score).max()),np.finfo(float).tiny);win=abs(score-best)<=1e-12*scale;selected=float(y[win].mean());choices.append(dict(cohort=co,cls=key[0],tgt=key[1],budget=key[2],repeat=key[3],method=method,regret=float(y.max()-selected),Top1=float(np.isclose(y,y.max(),atol=1e-12,rtol=0)[win].mean()),selected_F1=selected,worst_tie_loss=float(y.max()-y[win].min()),tied_candidates=int(win.sum())))
 for method in METHODS:
  rr=[r for r in choices if r['cohort']==co and r['method']==method];out=dict(cohort=co,method=method,common_configs=support,total_configs=nominal)
  for col in ['regret','Top1','selected_F1']:
   by=defaultdict(list)
   for r in rr:by[r['cls'],r['tgt'],r['budget']].append(r[col])
   task=defaultdict(list)
   for k,v in by.items():task[k[:2]].append(avg(v))
   out[col]=avg(avg(v) for v in task.values())
  out['max_loss']=max((r['regret'] for r in rr),default=np.nan);selection.append(out)
save(O/'模型选择.csv',choices);save(O/'模型选择宏汇总.csv',selection)
dump(O/'evaluation.json',dict(score_seal_verified=True,records=len(rows),methods=METHODS,ablation=FAMILIES,task_rows=len(tasks),ranking_rows=len(ranking),target_reference_only_postseal=True,target_annotation_reads=0,evaluation_cache_sha256=sha(OLD/'artifacts/逐项评价_full.csv'),formal_direction_table='arithmetic five-seed mean within-task correlation',aggregate_Pearson='Fisher z across tasks within seed, then mean five seeds',OTCE_approximation='Q1024_full_input_product_lift',evaluation_code_sha256=sha(__file__)));print('U8_EVALUATION_COMPLETE',flush=True)
