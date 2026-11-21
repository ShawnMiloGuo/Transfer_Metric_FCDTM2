"""Independent evaluator. This entry point is forbidden until full score sealing."""
from pathlib import Path
from collections import defaultdict
import sys,csv,json,hashlib,math
B=Path(__file__).resolve().parents[1];O=B/'artifacts';sys.path.insert(0,str(B/'.deps'));import numpy as np
import metrics as met
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def save(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def f1(c):
 t,p,n=np.asarray(c).sum(0) if np.asarray(c).ndim>1 else c;den=2*t+p+n
 return float(2*t/den) if den else float('nan')
def mean(v):
 a=[float(x) for x in v if np.isfinite(x)];return float(np.mean(a)) if a else float('nan')
def ranks(x):
 o=np.argsort(x,kind='stable');r=np.empty(len(x));i=0
 while i<len(o):
  j=i+1
  while j<len(o) and x[o[j]]==x[o[i]]:j+=1
  r[o[i:j]]=(i+j-1)/2;i=j
 return r

def corr(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok];n=len(x)
 if n<4:return n,np.nan,np.nan,'insufficient_points'
 if np.ptp(x)==0:return n,np.nan,np.nan,'constant_score'
 if np.ptp(y)==0:return n,np.nan,np.nan,'constant_AD'
 def cc(a,b):return float(np.corrcoef((a-a.min())/np.ptp(a),(b-b.min())/np.ptp(b))[0,1])
 return n,cc(x,y),cc(ranks(x),ranks(y)),'defined'
def orient(r,k):return float(r[k]) if k in met.BAD else -float(r[k])
def scenario(src,tgt):return (('region_S2' if 'sentinel2' in src else 'region_L8') if src.split('_')[0]!=tgt.split('_')[0] else ('sensor_GBA' if src.startswith('dwq') else 'sensor_XJ'))

def main():
 assert (O/'seal_full.json').exists(),'No target-reference access before complete score sealing.'
 import subprocess
 subprocess.check_call([sys.executable,str(B/'code/augment_doc_sourcecal.py')])
 subprocess.check_call([sys.executable,str(B/'code/finalise_singleton.py')])
 assert (O/'singleton_canonicalization.json').exists()
 seal=json.loads((O/'seal_full.json').read_text())
 for p,h in seal.items():assert sha(B/p)==h,p
 manifest=json.loads((O/'manifest_full.json').read_text());applicable={'full2048':met.METHODS[:-2]+['DoC_Acc_SrcCal'],'shared256':met.METHODS+['DoC_Acc_SrcCal']};rows=read(O/'predictions_full.csv');assert len(rows)==6600
 # Only after verifying the full seal are reference masks opened.
 from osgeo import gdal
 gdal.UseExceptions();data=Path('/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset');truth={};labelhash={}
 for domain,pool in manifest['pools'].items():
  labels=[]
  for item in pool:
   p=data/domain/'train_val/annotations'/item['split']/item['name'];labels.append(gdal.Open(str(p)).ReadAsArray().astype('uint8'));labelhash[str(p)]=sha(p)
  truth[domain]=np.array(labels).reshape(40,-1)
 counts={};accuracies={}
 for task in manifest['tasks']:
  cls,src,tgt=task['cls'],task['src'],task['tgt']
  with np.load(O/'cache'/f'{cls}_{src}__{tgt}.npz') as z:
   pred=np.unpackbits(z['bits'],axis=1,count=65536).astype(bool);valid=np.unpackbits(z['valid_bits'],axis=1,count=65536).astype(bool);y=truth[tgt]==cls
   counts[cls,src,tgt]=np.c_[(pred&y&valid).sum(1),(pred&~y&valid).sum(1),(~pred&y&valid).sum(1)];accuracies[cls,src,tgt]=np.c_[((pred==y)&valid).sum(1),valid.sum(1)]
 groups=defaultdict(list)
 for r in rows:
  for key in ['cls','budget','seed','repeat']:r[key]=int(r[key])
  for key in met.METHODS+['DoC_Acc_SrcCal','source_F1']:r[key]=float(r[key])
  ix=np.array(json.loads(r['indices']));key=r['cls'],r['src'],r['tgt'];r['true_F1']=f1(counts[key][ix]);r['true_AD']=r['source_F1']-r['true_F1'];ac=accuracies[key][ix].sum(0);r['true_accuracy']=float(ac[0]/ac[1]);r['scenario']=scenario(r['src'],r['tgt']);r['common']=bool(np.isfinite(r['true_AD']) and all(np.isfinite(r[k]) for k in applicable[r['panel']]))
  if r['kind']=='batch4':groups[(r['panel'],*key,r['seed'])].append(r)
 save(O/'逐项评价_full.csv',rows)
 cr=[]
 for key,rr in groups.items():
  for scope in ['own','common']:
   sub=[r for r in rr if scope=='own' or r['common']]
   for method in applicable[key[0]]:
    n,pr,sr,status=corr([orient(r,method) for r in sub],[r['true_AD'] for r in sub]);cr.append(dict(panel=key[0],cls=key[1],src=key[2],tgt=key[3],seed=key[4],scenario=rr[0]['scenario'],scope=scope,method=method,K=n,Pearson=pr,Spearman=sr,status=status))
 save(O/'逐任务相关_full.csv',cr);summary=[];scenarios=['all','sensor_GBA','sensor_XJ','region_S2','region_L8']
 for panel,methods in applicable.items():
  for cohort,keys in manifest['cohorts'].items():
   keys={tuple(k) for k in keys}
   for scen in scenarios:
    for scope in ['common','own']:
     for method in methods:
      a=[r for r in cr if r['panel']==panel and r['scope']==scope and r['method']==method and (r['cls'],r['src'],r['tgt']) in keys and (scen=='all' or r['scenario']==scen)];seeds=defaultdict(list)
      for r in a:seeds[r['seed']].append(r)
      ff=[];rho=[]
      for seed,v in seeds.items():
       fish=[np.arctanh(np.clip(r['Pearson'],-1+1e-7,1-1e-7)) for r in v if np.isfinite(r['Pearson'])];ff.append(float(np.tanh(mean(fish))) if fish else np.nan);rho.append(mean(r['Spearman'] for r in v))
      sd=lambda x:float(np.std([v for v in x if np.isfinite(v)])) if any(np.isfinite(x)) else np.nan
      summary.append(dict(panel=panel,cohort=cohort,scenario=scen,scope=scope,method=method,Fisher_r=mean(ff),Fisher_r_seed_sd=sd(ff),rho=mean(rho),rho_seed_sd=sd(rho),finite_correlations=sum(r['status']=='defined' for r in a),constant_score_tasks=sum(r['status']=='constant_score' for r in a),constant_AD_tasks=sum(r['status']=='constant_AD' for r in a),common_points=sum(r['K'] for r in a)))
 save(O/'相关汇总_full.csv',summary)
 # Correlation differences are paired per task/seed; constant-score baselines cannot
 # be rescued by silently discarding all their tasks from every other comparison.
 paired=[];rng=np.random.default_rng(721121)
 for panel,methods in applicable.items():
  for cohort,keys in manifest['cohorts'].items():
   keys={tuple(k) for k in keys};a=[r for r in cr if r['panel']==panel and r['scope']=='common' and (r['cls'],r['src'],r['tgt']) in keys];lookup={(r['cls'],r['src'],r['tgt'],r['seed'],r['method']):r for r in a}
   for method in methods:
    if method=='FCDTM_SoftPL':continue
    by=defaultdict(list)
    for r in a:
     if r['method']!=method:continue
     other=lookup[r['cls'],r['src'],r['tgt'],r['seed'],'FCDTM_SoftPL']
     if np.isfinite(r['Pearson']) and np.isfinite(other['Pearson']):by[r['cls'],r['src']].append(float(np.arctanh(np.clip(other['Pearson'],-1+1e-7,1-1e-7))-np.arctanh(np.clip(r['Pearson'],-1+1e-7,1-1e-7))))
    cluster=np.array([np.mean(v) for v in by.values()]);boot=np.array([rng.choice(cluster,len(cluster),replace=True).mean() for _ in range(1000)]) if len(cluster) else np.array([np.nan]);paired.append(dict(panel=panel,cohort=cohort,comparator=method,paired_task_seeds=sum(len(v) for v in by.values()),source_model_clusters=len(cluster),mean_delta_Fisher_z=float(cluster.mean()) if len(cluster) else np.nan,descriptive_95_lo=float(np.nanquantile(boot,.025)) if len(cluster) else np.nan,descriptive_95_hi=float(np.nanquantile(boot,.975)) if len(cluster) else np.nan))
 save(O/'配对源模型簇差异_full.csv',paired)
 choices=[];selection=[]
 for panel,methods in applicable.items():
  for cohort,keys in manifest['cohorts'].items():
   keys={tuple(k) for k in keys};pool=defaultdict(list)
   for r in rows:
    if r['panel']==panel and r['kind']=='pool' and r['budget']<40 and (r['cls'],r['src'],r['tgt']) in keys:pool[r['cls'],r['tgt'],r['budget'],r['repeat']].append(r)
   eligible=0;total=0
   for key,rs in pool.items():
    if len(rs)<2:continue
    total+=1;y=np.array([r['true_F1'] for r in rs]);joint=np.isfinite(y).all() and all(not any(np.isnan(orient(r,k)) for r in rs) for k in methods)
    if joint:eligible+=1
    for method in methods:
     score=np.array([orient(r,method) for r in rs]);ok=np.isfinite(y).all() and not np.isnan(score).any();regret=top=selected=worst=np.nan;ties=0
     if ok:
      best=score.min()
      if np.isinf(best):win=score==best
      else:
       finite=np.isfinite(score);scale=max(float(abs(score[finite]).max()),np.finfo(float).tiny);win=finite&(abs(score-best)<=1e-12*scale)
      ties=int(win.sum());selected=float(y[win].mean());regret=float(y.max()-selected);top=float(np.isclose(y,y.max(),atol=1e-12,rtol=0)[win].mean());worst=float(y.max()-y[win].min())
     choices.append(dict(panel=panel,cohort=cohort,cls=key[0],tgt=key[1],budget=key[2],repeat=key[3],method=method,common=joint,regret=regret,Top1=top,selected_F1=selected,worst_tie_loss=worst,tied_candidates=ties,candidates=len(rs)))
   for method in methods:
    cc=[r for r in choices if r['panel']==panel and r['cohort']==cohort and r['method']==method and r['common']];out=dict(panel=panel,cohort=cohort,method=method,common_configs=eligible,total_configs=total)
    for column in ['regret','Top1','selected_F1']:
     budgets=defaultdict(list)
     for r in cc:budgets[r['cls'],r['tgt'],r['budget']].append(r[column])
     tasks=defaultdict(list)
     for k,v in budgets.items():tasks[k[:2]].append(mean(v))
     out[column]=mean(mean(v) for v in tasks.values());out['target_class_groups']=len(tasks)
    out['max_loss']=max((r['regret'] for r in cc),default=np.nan);out['worst_possible_tie_loss']=max((r['worst_tie_loss'] for r in cc),default=np.nan);selection.append(out)
 save(O/'模型选择_full.csv',choices);save(O/'模型选择宏汇总_full.csv',selection)
 accuracy=[]
 for panel in applicable:
  for method in ['ATC_MC','ATC_NE','DoC_Acc_SrcCal']:
   q=[r for r in rows if r['panel']==panel and r['kind']=='batch4' and np.isfinite(r[method])];error=np.array([r[method]-r['true_accuracy'] for r in q]);accuracy.append(dict(panel=panel,method=method,unit='pixel_accuracy_not_foreground_F1',records=len(q),MAE=float(abs(error).mean()),bias=float(error.mean())))
 save(O/'ATC原生accuracy误差.csv',accuracy)
 (O/'evaluation_full.json').write_text(json.dumps(dict(seal_verified=True,records=len(rows),target_reference_only_postseal=True,evaluator_sha256=sha(Path(__file__)),target_reference_sha256=labelhash,methods=applicable,constant_NCE_is_not_failure=True),indent=1));print('EVALUATION COMPLETE',len(rows),'records; both sampling panels; all 18 comparator entries (including raw and source-calibrated DoC)',flush=True)
if __name__=='__main__':main()
