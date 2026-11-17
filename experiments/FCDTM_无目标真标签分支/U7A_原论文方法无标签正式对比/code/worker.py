import os,sys,json
from pathlib import Path
import numpy as np
import common as c

def main(cls,source,device):
 m=c.check();t=next(t for t in m['models'] if (t['cls'],t['src'])==(cls,source))
 import torch
 from osgeo import gdal
 torch.set_num_threads(1);gdal.UseExceptions();sys.path.insert(0,str(c.REPO));allowed={str(c.DATA/source/'train_val/annotations/val'/n) for n in t['ref']};denied=[]
 def guard(path):
  if not isinstance(path,(str,bytes,Path)):return
  p=str(Path(os.fsdecode(path)).absolute())
  if ('/annotations/' in p and p not in allowed) or ('/experiments/' in p and not p.startswith(str(c.BASE)) and not p.endswith(('.py','.pyc'))):denied.append(p);raise PermissionError('U7A target truth/evaluation access forbidden')
 sys.addaudithook(lambda e,a:guard(a[0]) if e=='open' else None);orig=gdal.Open
 def safe(p,*a,**kw):guard(p);return orig(p,*a,**kw)
 gdal.Open=safe
 other=next(x['tgt'] for x in m['tasks'] if (x['cls'],x['src'])==(cls,source));r=m['pools'][other][0]
 try:gdal.Open(str(c.DATA/other/'train_val/annotations'/r['split']/r['name']));raise AssertionError('Guard did not block')
 except PermissionError:pass
 assert c.sha(t['ck'])==t['sha'];dev=torch.device('cuda:'+str(device) if torch.cuda.is_available() else 'cpu');model=torch.load(t['ck'],map_location='cpu',weights_only=False).eval().to(dev);holder={};hook=model.outc.register_forward_pre_hook(lambda mod,a:holder.update(z=a[0]));v=(model.outc.weight[1]-model.outc.weight[0]).detach().flatten().cpu().double().numpy();bias=float((model.outc.bias[1]-model.outc.bias[0]).detach().cpu());w=-10*v-bias
 sample_features=None
 def infer(domain,r,i):
  nonlocal sample_features
  ds=gdal.Open(str(c.DATA/domain/'train_val/images'/r['split']/r['name']));x=ds.ReadAsArray().astype('float32');valid=np.isfinite(x).all(0)
  for band in range(4):
   nd=ds.GetRasterBand(band+1).GetNoDataValue()
   if nd is not None:valid&=x[band]!=nd
  assert x.shape==(4,256,256) and valid.any();x=np.nan_to_num(x)
  with torch.inference_mode():out=model(torch.from_numpy(x)[None].to(dev))[0];z=holder.pop('z')[0].permute(1,2,0).reshape(-1,len(v)).cpu().numpy();margin=(out[1]-out[0]).cpu().numpy().ravel();prob=out.softmax(0)[1].cpu().numpy().ravel()
  assert np.max(abs(z@v+bias-margin))<.01
  pred=margin>0;pool=np.flatnonzero(valid.ravel());ix=np.random.default_rng(270117+cls*1000+c.DOM.index(domain)*100+i).choice(pool,min(2048,len(pool)),replace=False);sample_features=z[ix];s=c.summary(sample_features,pred[ix],prob[ix]);return s,np.packbits(pred),np.packbits(valid),ix
 arrays=[];mus=np.zeros(len(v));mn=0;counts=np.zeros(3)
 for i,name in enumerate(t['ref']):
  s,bit,mask,ix=infer(source,dict(name=name,split='val'),i);truth=gdal.Open(str(c.DATA/source/'train_val/annotations/val'/name)).ReadAsArray().ravel()==cls;p=np.unpackbits(bit,count=65536).astype(bool);valid=np.unpackbits(mask,count=65536).astype(bool);counts+=np.array([(p&truth&valid).sum(),(p&~truth&valid).sum(),(~p&truth&valid).sum()]);keep=truth[ix]
  # Source true foreground is selected only after uniform sampling, no label-balanced sampling.
  # Reconstruct sampled feature sum from a dedicated forward hook-free second array capture.
  # Cached sample features are returned in infer through its shared holder.
  if keep.any():mus+=sample_features[keep].sum(0,dtype='float64');mn+=int(keep.sum())
  arrays.append(s)
 source_arrays={k:np.array([r[k] for r in arrays]) for k in arrays[0]};src=c.aggregate(source_arrays,np.arange(64));mu=mus/mn if mn else np.full(len(v),np.nan);sf=c.f1(counts);out=c.OUT/'source';out.mkdir(exist_ok=True);np.savez_compressed(out/f'{cls}_{source}.npz',**source_arrays);(out/f'{cls}_{source}.json').write_text(json.dumps(dict(source_F1=sf,mu=mu.tolist(),weight=w.tolist(),source_foreground_samples=mn,protected_dimensions=int((abs(mu)<1e-6).sum()),device=str(dev)),indent=1));rows=[]
 for domain in [x['tgt'] for x in m['tasks'] if (x['cls'],x['src'])==(cls,source)]:
  a=[];bits=[];masks=[];indices=[]
  for i,r in enumerate(m['pools'][domain]):
   s,bit,mask,ix=infer(domain,r,i);a.append(s);bits.append(bit);masks.append(mask);indices.append(ix)
  data={k:np.array([r[k] for r in a]) for k in a[0]};cache=c.OUT/'cache';cache.mkdir(exist_ok=True);np.savez_compressed(cache/f'{cls}_{source}__{domain}.npz',names=np.array([r['name'] for r in m['pools'][domain]]),bits=np.array(bits),valid_bits=np.array(masks),**data)
  for kind,budget,seed,repeat,ix in c.subsets(m,domain):
   score=c.scores(src,c.aggregate(data,ix),mu,w);rows.append(dict(cls=cls,src=source,tgt=domain,kind=kind,budget=budget,seed=seed,repeat=repeat,indices=json.dumps(ix.tolist()),source_F1=sf,**score))
 hook.remove();c.save(c.OUT/f'worker_{cls}_{source}.csv',rows);(c.OUT/f'guard_{cls}_{source}.json').write_text(json.dumps(dict(denied=denied,source_whitelist=sorted(allowed),target_truth_reads=0),indent=1));print('DONE',cls,source,flush=True)
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
