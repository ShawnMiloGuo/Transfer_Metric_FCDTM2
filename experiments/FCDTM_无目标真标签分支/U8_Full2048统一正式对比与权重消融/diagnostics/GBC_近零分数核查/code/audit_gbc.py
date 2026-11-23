# -*- coding: utf-8 -*-
"""Post-seal GBC numerical and grouping diagnostic; no new predictor fitting."""
from pathlib import Path
import sys,csv,json,hashlib,time
sys.path.insert(0,'/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/experiments/FCDTM_无目标真标签分支/U7B_审稿新增度量与伪标签适配/.deps')
import numpy as np
from osgeo import gdal
from threadpoolctl import threadpool_limits
threadpool_limits(1);gdal.UseExceptions()
BR=Path('/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/experiments/FCDTM_无目标真标签分支');B=next(BR.glob('U8*'));OLD=BR/'U7B_审稿新增度量与伪标签适配';A=B/'diagnostics/GBC_近零分数核查/artifacts';A.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:list(csv.DictReader(p.open()));m=json.loads((OLD/'artifacts/manifest_full.json').read_text());start=time.time()
for p,h in json.loads((B/'artifacts/score_seal.json').read_text()).items():assert sha(B/p)==h,p
# Existing scores were sealed before these reference labels are opened.
DATA=Path('/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset');truth={};labels_sha={}
for dom,pool in m['pools'].items():
 values=[]
 for r in pool:
  p=DATA/dom/'train_val/annotations'/r['split']/r['name'];values.append(gdal.Open(str(p)).ReadAsArray().ravel());labels_sha[str(p)]=sha(p)
 truth[dom]=np.array(values)
domains=['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8'];rows=read(B/'artifacts/evaluated.csv');q=[r for r in rows if (r['kind']=='batch4' and r['seed']=='290117') or (r['kind']=='pool' and r['budget']=='40')];out=[];error=0.
def gbc(z,y):
 n=np.bincount(y.astype(int),minlength=2)
 if n.min()<2:return dict(score=np.nan,log_magnitude=np.nan,distance=np.nan,n0=int(n[0]),n1=int(n[1]),status='insufficient_predicted_or_true_class_support')
 zz=z.astype(float);mu=np.array([zz[y==j].mean(0) for j in [0,1]]);vv=np.array([zz[y==j].var(0) for j in [0,1]]);v=np.maximum(vv,1e-4).mean(1);avg=v.mean();dm=.125*sum((mu[0]-mu[1])**2/avg);dv=.5*zz.shape[1]*np.log(avg/np.sqrt(v.prod()));d=max(float(dm+dv),0.);logmag=float(np.log(2)-d);score=float(-np.exp(logmag));return dict(score=score,log_magnitude=logmag,distance=d,mean_term=float(dm),variance_term=float(dv),variance0=float(v[0]),variance1=float(v[1]),variance_floor_fraction=float((vv<1e-4).mean()),n0=int(n[0]),n1=int(n[1]),status='defined')
for task in m['tasks']:
 cls,src,tgt=task['cls'],task['src'],task['tgt'];stem=f'{cls}_{src}__{tgt}';cf=OLD/'artifacts/features'/(stem+'.npz');z=dict(np.load(cf));bit=dict(np.load(OLD/'artifacts/cache'/(stem+'.npz')));valid=np.unpackbits(bit['valid_bits'],axis=1,count=65536).astype(bool);sample=[]
 for i in range(40):
  ids=np.random.default_rng(270117+cls*1000+domains.index(tgt)*100+i).choice(np.flatnonzero(valid[i]),min(2048,valid[i].sum()),False);sample.append(truth[tgt][i,ids]==cls)
 y=np.concatenate(sample)
 for r in q:
  if int(r['cls'])!=cls or r['src']!=src or r['tgt']!=tgt:continue
  ids=json.loads(r['indices']);ix=np.concatenate([np.arange(z['offsets'][i],z['offsets'][i+1]) for i in ids]);pl=gbc(z['z'][ix],z['pred'][ix]);gt=gbc(z['z'][ix],y[ix]);old=float(r['GBC_spherical_PL']);a=pl['score'];assert np.isclose(a,old,rtol=1e-7,atol=1e-250,equal_nan=True),(stem,a,old)
  if np.isfinite(a):error=max(error,abs(a-old))
  out.append({**{k:r[k] for k in ['cls','src','tgt','kind','budget','seed','repeat','true_AD']},**{'PL_'+k:v for k,v in pl.items()},**{'GT_'+k:v for k,v in gt.items()},'sampled_pseudo_precision':float(y[ix][z['pred'][ix]].mean()) if z['pred'][ix].any() else np.nan})
 print('CHECKED',stem,flush=True)
with (A/'grouping_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
v=np.array([float(r['GBC_spherical_PL']) for r in rows if r['kind']=='batch4']);v=v[np.isfinite(v)];batch=[r for r in out if r['kind']=='batch4'];report=dict(status='passed',all_finite_batch_scores=len(v),minimum_score=float(v.min()),maximum_score=float(v.max()),exact_zeros=int((v==0).sum()),abs_below_1e_3=int((abs(v)<1e-3).sum()),median_abs_score=float(np.median(abs(v))),diagnostic_records=len(out),independent_score_max_absolute_error=error,target_reference_use='post-seal diagnostic only; not used to modify scores or tuning',seconds=time.time()-start)
for mode in ['PL','GT']:
 vals=np.array([r[mode+'_score'] for r in batch]);vals=vals[np.isfinite(vals)];report[mode+'_diagnostic']=dict(finite=len(vals),exact_zeros=int((vals==0).sum()),median_abs=float(np.median(abs(vals))),abs_below_1e_3=int((abs(vals)<1e-3).sum()))
(A/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(A/'label_checksums.json').write_text(json.dumps(labels_sha,indent=2)+'\n');print('COMPLETE',json.dumps(report),flush=True)
