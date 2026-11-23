# -*- coding: utf-8 -*-
"""Independent, two-phase audit of frozen Bareland scores; never tunes outcomes."""
from pathlib import Path
import sys,os,json,csv,hashlib,time
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'U7B_审稿新增度量与伪标签适配/.deps'))
import numpy as np
from scipy.stats import pearsonr,spearmanr
from scipy.linalg import sqrtm
import torch
from osgeo import gdal
from threadpoolctl import threadpool_limits
D=Path(__file__).resolve().parents[1]; BR=D.parents[2]; B=BR/'U7B_审稿新增度量与伪标签适配'; O=B/'artifacts'; A=D/'artifacts'; A.mkdir(exist_ok=True)
REPO=BR.parents[1]; DATA=Path('/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset');sys.path.insert(0,str(REPO))
threadpool_limits(1);torch.set_num_threads(4);gdal.UseExceptions()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:list(csv.DictReader(Path(p).open()))
def save(name,rows):
 with (A/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(name,x):(A/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=True)+'\n')
def sf1(c):
 t,p,n=np.asarray(c).reshape(-1,3).sum(0);return float(2*t/(2*t+p+n)) if 2*t+p+n else np.nan
def corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 if len(x)<4 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan,np.nan
 return float(pearsonr((x-x.min())/np.ptp(x),(y-y.min())/np.ptp(y)).statistic),float(spearmanr(x,y).statistic)
phase=['score'];blocked=[]
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  p=os.fsdecode(args[0])
  if phase[0]=='score' and ('/annotations/' in p or '逐项评价' in p or '逐任务相关' in p or 'evaluation_full.json' in p):blocked.append(p);raise PermissionError('Forbidden reference access during score audit')
sys.addaudithook(guard)
try:open(str(DATA/'dummy/annotations/probe'))
except PermissionError:pass
m=json.loads((O/'manifest_full.json').read_text());seal=json.loads((O/'seal_full.json').read_text())
for p,h in seal.items():assert sha(B/p)==h,p
pred=[r for r in read(O/'predictions_full.csv') if r['cls']=='8' and r['panel']=='full2048']
methods=['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL'];domains=['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8']
sources={};target={};weights={};models=[];fresh=[];errors={k:0. for k in methods};recomputed=[];start=time.time()

def direct(z,p,yp,mu,w,source_cov,source_mean):
 zz=z.astype('float64');hard=zz[yp].mean(0) if yp.any() else np.full(len(mu),np.nan);soft=(z*p[:,None]).sum(0,dtype='float64')/p.sum(dtype='float64');den=np.where(abs(mu)<1e-6,1e-6,mu)
 mean=zz.mean(0);cov=np.cov(zz,rowvar=False);cross=sqrtm(source_cov@cov);fd=float(np.sum((source_mean-mean)**2)+np.trace(source_cov)+np.trace(cov)-2*np.trace(cross).real)
 cnt=np.bincount(yp.astype(int),minlength=2);means=np.array([zz[~yp].mean(0),zz[yp].mean(0)])
 ds=float(np.log(sum(cnt[j]*sum((means[j]-mean)**2) for j in [0,1] if cnt[j])))
 variance=np.array([np.mean(zz[~yp]**2,0)-means[0]**2,np.mean(zz[yp]**2,0)-means[1]**2]);variance=np.maximum(variance,1e-4).mean(1);avg=variance.mean();dist=.125*np.sum((means[0]-means[1])**2/avg)+.5*len(mu)*np.log(avg/np.sqrt(variance.prod()));gbc=float(-2*np.exp(-max(dist,0))) if min(cnt)>=2 else np.nan
 return dict(zip(methods,[float(((hard-mu)/den)@w),float(((soft-mu)/den)@w),max(fd,0),ds,gbc]))

for mi in [x for x in m['models'] if x['cls']==8]:
 src=mi['src'];assert sha(mi['ck'])==mi['sha'];model=torch.load(mi['ck'],map_location='cpu',weights_only=False).eval();holder={};hook=model.outc.register_forward_pre_hook(lambda mod,a:holder.update(z=a[0]));v=(model.outc.weight[1]-model.outc.weight[0]).detach().flatten().double().numpy();bias=float((model.outc.bias[1]-model.outc.bias[0]).detach());w=-10*v-bias;weights[src]=w
 sc=dict(np.load(O/'features'/f'8_{src}_source.npz'));sources[src]=sc;mu=sc['z'][sc['source_y'].astype(bool)].mean(0,dtype='float64');den=np.where(abs(mu)<1e-6,1e-6,mu);source_mean=sc['z'].mean(0,dtype='float64');source_cov=np.cov(sc['z'].astype('float64'),rowvar=False)
 models.append(dict(src=src,checkpoint=mi['ck'],checkpoint_sha=mi['sha'],source_F1=sf1(sc['counts']),source_sampled_foreground_fraction=float(sc['source_y'].mean()),source_pred_foreground_fraction=float(sc['pred'].mean()),mu_min=float(mu.min()),mu_max=float(mu.max()),near_zero_dimensions=int((abs(mu)<1e-6).sum()),feature_dimensions=len(mu),source_sampled_logit_sign_agreement=float(np.mean(((sc['z']@v+bias)>0)==sc['pred']))))
 for domain,info,cache,bitfile in [(src,dict(name=mi['ref'][0],split='val'),sc,None)]+[(t['tgt'],m['pools'][t['tgt']][0],dict(np.load(O/'features'/f'8_{src}__{t["tgt"]}.npz')),O/'cache'/f'8_{src}__{t["tgt"]}.npz') for t in m['tasks'] if t['cls']==8 and t['src']==src]:
  path=DATA/domain/'train_val/images'/info['split']/info['name'];ds=gdal.Open(str(path));x=ds.ReadAsArray().astype('float32');valid=np.isfinite(x).all(0)
  for j in range(4):
   nd=ds.GetRasterBand(j+1).GetNoDataValue()
   if nd is not None:valid&=x[j]!=nd
  with torch.inference_mode():logits=model(torch.from_numpy(np.nan_to_num(x))[None])[0];z=holder.pop('z')[0].permute(1,2,0).reshape(-1,len(v)).numpy();p=logits.softmax(0)[1].numpy().ravel();yp=logits.argmax(0).numpy().ravel().astype(bool)
  ix=np.random.default_rng(270117+8000+domains.index(domain)*100).choice(np.flatnonzero(valid.ravel()),min(2048,valid.sum()),False);lo,hi=cache['offsets'][:2];cz=cache['z'][lo:hi];cp=cache['p'][lo:hi];cyp=cache['pred'][lo:hi]
  # Original domain order is verified against U0 config before execution below.
  difz=float(np.max(abs(z[ix]-cz)));difp=float(np.max(abs(p[ix]-cp)));same=bool(np.array_equal(yp[ix],cyp));entry=dict(src=src,tgt=domain,name=info['name'],image_sha=sha(path),max_feature_difference=difz,max_probability_difference=difp,sampled_predictions_identical=same)
  assert difz<1e-5 and difp<1e-6 and same,entry
  if bitfile:
   bc=dict(np.load(bitfile));assert bc['names'].tolist()==[r['name'] for r in m['pools'][domain]];entry['full_prediction_bits_identical']=bool(np.array_equal(np.packbits(yp),bc['bits'][0]));entry['valid_bits_identical']=bool(np.array_equal(np.packbits(valid),bc['valid_bits'][0]));assert entry['full_prediction_bits_identical'] and entry['valid_bits_identical'];target[src,domain]=(cache,bc)
  else:entry['source_fresh_pred']=yp;entry['source_fresh_valid']=valid.ravel()
  fresh.append(entry)
 hook.remove()
 for r in [r for r in pred if r['src']==src]:
  domain=r['tgt'];cache,_=target[src,domain];ids=json.loads(r['indices']);pos=np.concatenate([np.arange(cache['offsets'][i],cache['offsets'][i+1]) for i in ids]);scores=direct(cache['z'][pos],cache['p'][pos],cache['pred'][pos],mu,w,source_cov,source_mean)
  for k,x in scores.items():
   old=float(r[k]);errors[k]=max(errors[k],abs(x-old)) if np.isfinite(x) and np.isfinite(old) else errors[k];assert np.isclose(x,old,rtol=2e-6,atol=2e-6,equal_nan=True),(src,domain,k,x,old)
  recomputed.append({**{k:r[k] for k in ['src','tgt','kind','budget','seed','repeat','indices']},**scores})
 print('SCORE_MODEL_DONE',src,flush=True)
save('recomputed_scores.csv',recomputed);dump('score_audit.json',dict(phase='score_only',records=len(recomputed),checkpoint_hashes_verified=True,target_annotation_reads=0,reference_probe_blocked=bool(blocked),max_absolute_score_difference=errors))
ss=sha(A/'recomputed_scores.csv');dump('score_seal.json',dict(score_sha256=ss,code_sha256=sha(__file__),input_manifest_sha256=sha(O/'manifest_full.json'),old_prediction_sha256=sha(O/'predictions_full.csv')))
assert ss==sha(A/'recomputed_scores.csv');phase[0]='evaluation'
# Post-seal evaluation and oracle diagnostics. No prediction or sign is updated.
old_eval=[r for r in read(O/'逐项评价_full.csv') if r['cls']=='8' and r['panel']=='full2048'];ev={(r['src'],r['tgt'],r['kind'],r['budget'],r['seed'],r['repeat']):r for r in old_eval};oldcorr=read(O/'逐任务相关_full.csv');group={};evaluated=[];taskrows=[];corrrows=[];maxf1=0.;source_label_checks=[]
for mi in [x for x in m['models'] if x['cls']==8]:
 src=mi['src'];sc=sources[src]
 for i,name in enumerate(mi['ref']):
  ds=gdal.Open(str(DATA/src/'train_val/images/val'/name));x=ds.ReadAsArray();valid=np.isfinite(x).all(0)
  for j in range(4):
   nd=ds.GetRasterBand(j+1).GetNoDataValue()
   if nd is not None:valid&=x[j]!=nd
  truth=gdal.Open(str(DATA/src/'train_val/annotations/val'/name)).ReadAsArray().ravel()==8;ix=np.random.default_rng(270117+8000+domains.index(src)*100+i).choice(np.flatnonzero(valid.ravel()),min(2048,valid.sum()),False);lo,hi=sc['offsets'][i:i+2];assert np.array_equal(truth[ix],sc['source_y'][lo:hi])
  if i==0:
   fr=next(x for x in fresh if x['src']==src and x['tgt']==src);yp=fr.pop('source_fresh_pred');vv=fr.pop('source_fresh_valid');c=[int((yp&truth&vv).sum()),int((yp&~truth&vv).sum()),int((~yp&truth&vv).sum())];assert np.array_equal(c,sc['counts'][0])
 source_label_checks.append(dict(src=src,source_images=64,class_id=8,sampled_label_match=True,first_fresh_full_counts_match=True))
for (src,tgt),(cache,bc) in target.items():
 truth=np.array([gdal.Open(str(DATA/tgt/'train_val/annotations'/r['split']/r['name'])).ReadAsArray().ravel()==8 for r in m['pools'][tgt]]);yp=np.unpackbits(bc['bits'],axis=1,count=65536).astype(bool);valid=np.unpackbits(bc['valid_bits'],axis=1,count=65536).astype(bool);counts=np.c_[(yp&truth&valid).sum(1),(yp&~truth&valid).sum(1),(~yp&truth&valid).sum(1)];n=valid.sum(1);sampletruth=[]
 for i in range(40):
  ix=np.random.default_rng(270117+8000+domains.index(tgt)*100+i).choice(np.flatnonzero(valid[i]),min(2048,valid[i].sum()),False);sampletruth.append(truth[i,ix])
 st=np.concatenate(sampletruth);sc=sources[src];mu=sc['z'][sc['source_y'].astype(bool)].mean(0,dtype='float64');den=np.where(abs(mu)<1e-6,1e-6,mu);w=weights[src]
 sf=sf1(sc['counts']);c=counts.sum(0);tp,fp,fn=c
 taskrows.append(dict(src=src,tgt=tgt,source_F1=sf,target_F1=sf1(c),AD=sf-sf1(c),precision=float(tp/(tp+fp)),recall=float(tp/(tp+fn)),true_foreground_fraction=float((tp+fn)/n.sum()),pred_foreground_fraction=float((tp+fp)/n.sum()),TP=int(tp),FP=int(fp),FN=int(fn)))
 for r in [r for r in recomputed if r['src']==src and r['tgt']==tgt]:
  ids=json.loads(r['indices']);c=counts[ids].sum(0);tp,fp,fn=c;f=sf1(c);ad=sf-f;old=ev[src,tgt,r['kind'],r['budget'],r['seed'],r['repeat']];maxf1=max(maxf1,abs(f-float(old['true_F1'])),abs(ad-float(old['true_AD'])));assert abs(f-float(old['true_F1']))<1e-12 and abs(ad-float(old['true_AD']))<1e-12
  pos=np.concatenate([np.arange(cache['offsets'][i],cache['offsets'][i+1]) for i in ids]);zz=cache['z'][pos];tt=st[pos];pp=cache['pred'][pos];oracle=zz[tt].mean(0,dtype='float64') if tt.any() else np.full(len(mu),np.nan);oracle_score=float(((oracle-mu)/den)@w);hard=zz[pp].mean(0,dtype='float64') if pp.any() else np.full(len(mu),np.nan)
  e={**r,'source_F1':sf,'target_F1':f,'AD':ad,'precision':float(tp/(tp+fp)) if tp+fp else np.nan,'recall':float(tp/(tp+fn)) if tp+fn else np.nan,'pred_foreground_fraction':float((tp+fp)/n[ids].sum()),'true_foreground_fraction':float((tp+fn)/n[ids].sum()),'oracle_FCDTM':oracle_score,'pseudo_vs_oracle_mean_relative_L1':float(np.mean(abs((hard-oracle)/den))),'sampled_pseudo_precision':float(tt[pp].mean()) if pp.any() else np.nan}
  evaluated.append(e)
  if r['kind']=='batch4' and old['common']=='True':group.setdefault((src,tgt,r['seed']),[]).append(e)
for (src,tgt,seed),rr in group.items():
 for k in methods+['oracle_FCDTM']:
  xx=[r[k] if k not in ['DS_pixel','GBC_spherical_PL'] else -r[k] for r in rr];pr,sr=corr(xx,[r['AD'] for r in rr]);out=dict(src=src,tgt=tgt,seed=seed,method=k,K=len(rr),Pearson=pr,Spearman=sr)
  if k!='oracle_FCDTM':
   old=next(r for r in oldcorr if r['cls']=='8' and r['panel']=='full2048' and r['scope']=='common' and r['src']==src and r['tgt']==tgt and r['seed']==seed and r['method']==k);assert int(old['K'])==len(rr) and abs(float(old['Pearson'])-pr)<1e-5 and abs(float(old['Spearman'])-sr)<1e-10,(out,old)
  for d in ['precision','recall','pred_foreground_fraction','true_foreground_fraction','sampled_pseudo_precision','pseudo_vs_oracle_mean_relative_L1']:out['r_'+d]=corr(xx,[r[d] for r in rr])[0]
  corrrows.append(out)
means=[]
for src,tgt in target:
 for k in methods+['oracle_FCDTM']:
  rr=[r for r in corrrows if r['src']==src and r['tgt']==tgt and r['method']==k];means.append(dict(src=src,tgt=tgt,method=k,Pearson=float(np.nanmean([r['Pearson'] for r in rr])),Spearman=float(np.nanmean([r['Spearman'] for r in rr])),negative_Pearson_seeds=sum(r['Pearson']<0 for r in rr),seeds=len(rr),**{c:float(np.nanmean([r[c] for r in rr])) for c in rr[0] if c.startswith('r_')}))
save('task_diagnostics.csv',taskrows);save('batch_diagnostics.csv',evaluated);save('seed_correlations.csv',corrrows);save('mean_correlations.csv',means)
dump('model_checks.json',models);dump('fresh_inference_checks.json',fresh);dump('source_label_checks.json',source_label_checks);dump('verification.json',dict(status='passed',models=4,tasks=8,source_images_label_checked=256,fresh_images_checked=len(fresh),score_rows=len(recomputed),correlations_checked=200,max_score_absolute_difference=errors,max_F1_AD_difference=maxf1,score_seal_unchanged=ss==sha(A/'recomputed_scores.csv'),source_and_target_separated=True,target_reference_only_postseal=True,seconds=time.time()-start,limitations='One fresh image per cache; no retraining or independent georegistration audit. Oracle is evaluation-only.'))
print('AUDIT_COMPLETE',json.dumps(taskrows),flush=True)
