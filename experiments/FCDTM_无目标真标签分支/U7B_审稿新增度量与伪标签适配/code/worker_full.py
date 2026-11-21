from pathlib import Path
import sys,os,json,time,re,hashlib,csv
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'.deps'));A=B.parent/'U7A_原论文方法无标签正式对比';sys.path.insert(0,str(A/'code'));import common as old
import numpy as np
import metrics as met
O=B/'artifacts';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(cls,source,device):
 import torch
 from osgeo import gdal
 from threadpoolctl import threadpool_limits
 threadpool_limits(limits=1);torch.set_num_threads(1);gdal.UseExceptions()
 manifest=json.loads((O/'manifest_full.json').read_text());modelinfo=next(x for x in manifest['models'] if x['cls']==cls and x['src']==source)
 for rel,h in manifest['code_sha256'].items():assert sha(B/rel)==h,rel
 allowed={str((old.DATA/source/'train_val/annotations/val'/name).resolve()) for name in modelinfo['ref']};denied=[]
 def guard(path):
  if not isinstance(path,(str,bytes,Path)):return
  p=str(Path(os.fsdecode(path)).absolute())
  if ('/annotations/' in p and p not in allowed) or any(k in p for k in ['逐项评价.csv','评价真值.csv','评价真值.npz','true_F1']):denied.append(p);raise PermissionError('U7B target reference/evaluation access forbidden')
 sys.addaudithook(lambda e,a:guard(a[0]) if e=='open' else None);orig=gdal.Open
 def safe(path,*a,**kw):guard(path);return orig(path,*a,**kw)
 gdal.Open=safe
 target=next(t['tgt'] for t in manifest['tasks'] if t['cls']==cls and t['src']==source);test=manifest['pools'][target][0]
 try:gdal.Open(str(old.DATA/target/'train_val/annotations'/test['split']/test['name']));raise AssertionError('Target reference guard failed')
 except PermissionError:pass
 assert sha(modelinfo['ck'])==modelinfo['sha'];sys.path.insert(0,str(old.REPO));dev=torch.device('cuda:'+str(device));model=torch.load(modelinfo['ck'],map_location='cpu',weights_only=False).eval().to(dev);holder={};hook=model.outc.register_forward_pre_hook(lambda mod,a:holder.update(z=a[0]));v=(model.outc.weight[1]-model.outc.weight[0]).detach().flatten().cpu().double().numpy();bias=float((model.outc.bias[1]-model.outc.bias[0]).detach().cpu());weight=-10*v-bias
 def infer(domain,info,i,source_label=False):
  ds=gdal.Open(str(old.DATA/domain/'train_val/images'/info['split']/info['name']));x=ds.ReadAsArray().astype('float32');valid=np.isfinite(x).all(0)
  for j in range(4):
   nd=ds.GetRasterBand(j+1).GetNoDataValue()
   if nd is not None:valid&=x[j]!=nd
  assert x.shape==(4,256,256) and valid.any()
  with torch.inference_mode():
   out=model(torch.from_numpy(np.nan_to_num(x))[None].to(dev))[0];z=holder.pop('z')[0].permute(1,2,0).reshape(-1,len(v)).cpu().numpy();margin=(out[1]-out[0]).cpu().numpy().ravel();p=out.softmax(0)[1].cpu().numpy().ravel()
  assert np.max(abs(z@v+bias-margin))<.01;pred=margin>0;ids=np.flatnonzero(valid.ravel());ix=np.random.default_rng(270117+cls*1000+old.DOM.index(domain)*100+i).choice(ids,min(2048,len(ids)),replace=False)
  result=dict(z=z[ix],p=p[ix],pred=pred[ix],bits=np.packbits(pred),valid_bits=np.packbits(valid))
  if source_label:
   y=gdal.Open(str(old.DATA/domain/'train_val/annotations/val'/info['name'])).ReadAsArray().ravel()==cls;result['y']=y[ix];result['counts']=np.array([(pred&y&valid.ravel()).sum(),(pred&~y&valid.ravel()).sum(),(~pred&y&valid.ravel()).sum()])
  return result
 src=[infer(source,dict(name=n,split='val'),i,True) for i,n in enumerate(modelinfo['ref'])];srcz=np.concatenate([x['z'] for x in src]);srcp=np.concatenate([x['p'] for x in src]);srcpred=np.concatenate([x['pred'] for x in src]);srcy=np.concatenate([x['y'] for x in src]).astype(int);source_counts=np.array([x['counts'] for x in src]);sf=old.f1(source_counts)
 cix=np.random.default_rng(301121+cls*1000+old.DOM.index(source)*100).choice(len(srcz),256,replace=False)
 contexts={'full2048':met.context(srcz,srcp,srcy,weight,srcpred),'shared256':met.context(srcz[cix],srcp[cix],srcy[cix],weight,srcpred[cix])}
 # Index-block source-only pseudo-transfer calibration, no target reference data.
 grids=[tuple(map(int,re.search(r'_i(\d+)j(\d+)',n).groups())) for n in modelinfo['ref']];axis=0 if len({g[0] for g in grids})>1 else 1;values=sorted({g[axis] for g in grids});left=set(values[:len(values)//2]);fit=np.array([i for i,g in enumerate(grids) if g[axis] in left]);check=np.array([i for i,g in enumerate(grids) if g[axis] not in left]);cal=None;calmeta=dict(axis=axis,fit_images=fit.tolist(),check_images=check.tolist(),status='unsupported_source_partition')
 if len(fit)>=4 and len(check)>=4:
  ctx=contexts['shared256'];calrows=[]
  for role,pool,count in [('fit',fit,16),('check',check,8)]:
   for j in range(count):
    rng=np.random.default_rng(399121+cls*1000+old.DOM.index(source)*100+j+(0 if role=='fit' else 10000));ix=rng.choice(pool,4,False);zz=np.concatenate([src[i]['z'] for i in ix]);pp=np.concatenate([src[i]['pred'] for i in ix]);cap=rng.choice(len(zz),256,False);comp,_=met.transport(ctx['z'],zz[cap],ctx['y'],pp[cap].astype(int));calrows.append(dict(role=role,indices=ix.tolist(),features=comp.tolist(),AD=sf-old.f1(source_counts[ix])))
  train=[r for r in calrows if r['role']=='fit' and np.isfinite(r['AD'])];chk=[r for r in calrows if r['role']=='check' and np.isfinite(r['AD'])];calmeta.update(rows=calrows,defined_fit_tasks=len(train),defined_check_tasks=len(chk))
  if len(train)>=3:
   x=np.array([r['features'] for r in train]);y=np.array([r['AD'] for r in train]);center=x.mean(0);scale=np.maximum(x.std(0),1e-8);xx=(x-center)/scale;inter=y.mean();coef=np.linalg.solve(xx.T@xx+1e-3*np.eye(2),xx.T@(y-inter));cal=dict(mean=center,scale=scale,intercept=inter,coef=coef)
   err=[abs(cal['intercept']+((np.array(r['features'])-center)/scale)@coef-r['AD']) for r in chk];calmeta.update(status='source_only_fit',mean=center.tolist(),scale=scale.tolist(),coef=coef.tolist(),intercept=float(inter),source_check_MAE=float(np.mean(err)) if err else None,ridge_lambda=.001)
  else:calmeta.update(status='insufficient_defined_source_AD')

 stem=f'{cls}_{source}';(O/'source').mkdir(exist_ok=True);(O/'cache').mkdir(exist_ok=True);(O/'features').mkdir(exist_ok=True)
 np.savez_compressed(O/'features'/f'{stem}_source.npz',z=srcz,p=srcp,pred=srcpred,source_y=srcy,counts=source_counts,offsets=np.r_[0,np.cumsum([len(x['z']) for x in src])])
 (O/'source'/f'{stem}.json').write_text(json.dumps(dict(source_F1=sf,calibration=calmeta,ATC_thresholds={k:{n:float(v[n]) for n in ['ATC_MC','ATC_NE','bandwidth']} for k,v in contexts.items()}),indent=1));rows=[];start=time.time();trace=[]
 for domain in [t['tgt'] for t in manifest['tasks'] if t['cls']==cls and t['src']==source]:
  targetdata=[infer(domain,r,i) for i,r in enumerate(manifest['pools'][domain])];ss=[old.summary(x['z'],x['pred'],x['p']) for x in targetdata];summaries={k:np.array([r[k] for r in ss]) for k in ss[0]};bits=np.array([r['bits'] for r in targetdata]);masks=np.array([r['valid_bits'] for r in targetdata]);np.savez_compressed(O/'cache'/f'{stem}__{domain}.npz',names=np.array([r['name'] for r in manifest['pools'][domain]]),bits=bits,valid_bits=masks)
  np.savez_compressed(O/'features'/f'{stem}__{domain}.npz',z=np.concatenate([r['z'] for r in targetdata]),p=np.concatenate([r['p'] for r in targetdata]),pred=np.concatenate([r['pred'] for r in targetdata]),offsets=np.r_[0,np.cumsum([len(r['z']) for r in targetdata])])
  with np.load(A/'artifacts/cache'/f'{stem}__{domain}.npz') as oldcache:trace.append(dict(tgt=domain,bits_identical=bool(np.array_equal(bits,oldcache['bits'])),valid_identical=bool(np.array_equal(masks,oldcache['valid_bits']))))
  from concurrent.futures import ThreadPoolExecutor
  def one_record(spec):
   kind,budget,seed,repeat,ix=spec
   zz=np.concatenate([targetdata[i]['z'] for i in ix]);pp=np.concatenate([targetdata[i]['p'] for i in ix]);yp=np.concatenate([targetdata[i]['pred'] for i in ix]);seedcode=420121+cls*100000+old.DOM.index(source)*10000+old.DOM.index(domain)*1000+int(seed)+budget*7+repeat;rng=np.random.default_rng(seedcode);cap=rng.choice(len(zz),256,False);answer=[]
   for panel in ['full2048','shared256']:
    cut=slice(None) if panel=='full2048' else cap;values,detail=met.score(zz[cut],pp[cut],contexts[panel],old,full_stats=old.aggregate(summaries,ix) if panel=='full2048' else None,otcal=cal,shared=panel=='shared256',yp=yp[cut]);answer.append(dict(cls=cls,src=source,tgt=domain,panel=panel,kind=kind,budget=budget,seed=seed,repeat=repeat,indices=json.dumps(ix.tolist()),source_F1=sf,**values,diagnostics=json.dumps(detail)))
   return answer
  with ThreadPoolExecutor(max_workers=4) as cpu_pool:
   for record,answer in enumerate(cpu_pool.map(one_record,list(old.subsets(manifest,domain)))):
    rows.extend(answer)
    if record%10==0:print('PROGRESS',stem,domain,record,'seconds',round(time.time()-start),flush=True)

 hook.remove()
 with (O/f'worker_{stem}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (O/f'guard_{stem}.json').write_text(json.dumps(dict(target_truth_reads=0,denied=denied,source_whitelist=sorted(allowed),prediction_bit_comparison=trace,seconds=time.time()-start),indent=1));print('DONE',stem,len(rows),flush=True)
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
