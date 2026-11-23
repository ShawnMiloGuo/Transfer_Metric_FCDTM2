# -*- coding: utf-8 -*-
from common import *
import numpy as np,time
sys.path.insert(0,str(OLD/'.deps'))
from threadpoolctl import threadpool_limits
from kernel import grouped_kernel,from_sums
threadpool_limits(1)
cls=int(sys.argv[1]);src=sys.argv[2];start=time.time();denied=score_guard();out=O/'workers';out.mkdir(exist_ok=True);ko=O/'kernel_sums';ko.mkdir(exist_ok=True)
m=json.loads((OLD/'artifacts/manifest_full.json').read_text());mi=next(x for x in m['models'] if x['cls']==cls and x['src']==src);assert sha(mi['ck'])==mi['sha']
import torch
sys.path.insert(0,'/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2');torch.set_num_threads(1)
model=torch.load(mi['ck'],map_location='cpu',weights_only=False).eval();v=(model.outc.weight[1]-model.outc.weight[0]).detach().flatten().double().numpy();bias=float((model.outc.bias[1]-model.outc.bias[0]).detach());w=-10*v-bias;del model
sourcefile=OLD/'artifacts/features'/f'{cls}_{src}_source.npz';sc=dict(np.load(sourcefile));sx=sc['z'].astype('float64');sfmeta=json.loads((OLD/'artifacts/source'/f'{cls}_{src}.json').read_text());bw=sfmeta['ATC_thresholds']['shared256']['bandwidth'];mu=sc['z'][sc['source_y'].astype(bool)].mean(0,dtype='float64');den=np.where(abs(mu)<1e-6,1e-6,mu)
cache=ko/f'{cls}_{src}_source.npz'
if cache.exists():
 z=dict(np.load(cache));assert z['feature_sha'].item()==sha(sourcefile) and z['bandwidth'].item()==bw;ss=float(z['sum'])
else:
 sk=grouped_kernel(sx,sc['offsets'],bw=bw,progress=lambda i,n:print('SOURCE',cls,src,i,n,round(time.time()-start),flush=True));ss=float(sk.sum());np.savez_compressed(cache,sum=ss,matrix=sk,bandwidth=bw,feature_sha=sha(sourcefile))
rows=[r for r in read(OLD/'artifacts/predictions_full.csv') if int(r['cls'])==cls and r['src']==src and r['panel']=='full2048'];answer=[];weighted_error=0.;inputs={str(sourcefile):sha(sourcefile)}
for tgt in sorted({r['tgt'] for r in rows}):
 tf=OLD/'artifacts/features'/f'{cls}_{src}__{tgt}.npz';tc=dict(np.load(tf));tx=tc['z'].astype('float64');inputs[str(tf)]=sha(tf);kf=ko/f'{cls}_{src}__{tgt}.npz'
 if kf.exists():
  z=dict(np.load(kf));assert z['feature_sha'].item()==sha(tf) and z['bandwidth'].item()==bw;tt=z['target_self'];st=z['source_cross']
 else:
  tt=grouped_kernel(tx,tc['offsets'],bw=bw,progress=lambda i,n:print('TARGET',cls,src,tgt,i,n,round(time.time()-start),flush=True));cross=grouped_kernel(sx,sc['offsets'],tx,tc['offsets'],bw=bw,progress=lambda i,n:print('CROSS',cls,src,tgt,i,n,round(time.time()-start),flush=True));st=cross.sum(0);np.savez_compressed(kf,target_self=tt,source_cross=st,source_cross_image_matrix=cross,bandwidth=bw,feature_sha=sha(tf),source_feature_sha=sha(sourcefile))
 nt=np.diff(tc['offsets']);summ=[]
 for i in range(len(nt)):
  lo,hi=tc['offsets'][i:i+2];zz=tc['z'][lo:hi];pp=tc['p'][lo:hi];yp=tc['pred'][lo:hi];summ.append(dict(hsum=zz[yp].sum(0,dtype='float64'),hn=int(yp.sum()),ssum=(zz*pp[:,None]).sum(0,dtype='float64'),sn=float(pp.sum(dtype='float64'))))
 for r in [r for r in rows if r['tgt']==tgt]:
  ids=json.loads(r['indices']);hs=sum(summ[i]['hsum'] for i in ids);hn=sum(summ[i]['hn'] for i in ids);ps=sum(summ[i]['ssum'] for i in ids);pn=sum(summ[i]['sn'] for i in ids);h=hs/hn if hn else np.full_like(mu,np.nan);s=ps/pn if pn else np.full_like(mu,np.nan);dh=(h-mu)/den;ds=(s-mu)/den
  for name,d in [('FCDTM_HardPL',dh),('FCDTM_SoftPL',ds)]:
   value=float(d@w);old=float(r[name]);assert np.isclose(value,old,rtol=1e-10,atol=1e-9,equal_nan=True),(cls,src,tgt,name,value,old)
   if np.isfinite(value):weighted_error=max(weighted_error,abs(value-old))
  new={k:r[k] for k in ['cls','src','tgt','panel','kind','budget','seed','repeat','indices','source_F1']};new.update({k:float(r[k]) for k in METHODS if k not in ['OTCE_PL_SrcCal','MMD_RBF_global']});new.update(MMD_RBF_global=from_sums(ss,len(sx),tt,nt,st,ids),FCDTM_HardSigned=float(dh.sum()),FCDTM_HardAbs=float(abs(dh).sum()),FCDTM_SoftSigned=float(ds.sum()),FCDTM_SoftAbs=float(abs(ds).sum()));answer.append(new)
 print('TASK_DONE',cls,src,tgt,len(answer),round(time.time()-start),flush=True)
save(out/f'{cls}_{src}.csv',answer);dump(out/f'{cls}_{src}_audit.json',dict(target_reference_reads=0,guard_probe_blocked=True,unexpected_denials=denied[1:],records=len(answer),source_points=len(sx),source_bandwidth=bw,inputs=inputs,checkpoint_sha=mi['sha'],weighted_score_max_error=weighted_error,seconds=time.time()-start,code_sha={p.name:sha(p) for p in [B/'code/common.py',B/'code/kernel.py',Path(__file__)]}));print('WORKER_DONE',cls,src,len(answer),round(time.time()-start),flush=True)
