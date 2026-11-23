# -*- coding: utf-8 -*-
from common import *
import numpy as np,time
sys.path.insert(0,str(OLD/'.deps'));sys.path.insert(0,str(B/'.deps'))
from threadpoolctl import threadpool_limits
threadpool_limits(1)
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from ot_quantized import Q,stats,aggregate,transport
cls=int(sys.argv[1]);src=sys.argv[2];start=time.time();denied=score_guard();out=O/'ot_workers';out.mkdir(exist_ok=True);temp=O/'ot_cache';temp.mkdir(exist_ok=True)
m=json.loads((OLD/'artifacts/manifest_full.json').read_text());meta=json.loads((OLD/'artifacts/source'/f'{cls}_{src}.json').read_text());sf=OLD/'artifacts/features'/f'{cls}_{src}_source.npz';sc=dict(np.load(sf));z=sc['z'];ys=sc['source_y'].astype(int);so=sc['offsets'];seed=20261123+cls*100+['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8'].index(src)
cache=temp/f'{cls}_{src}_source.npz'
if cache.exists():
 c=dict(np.load(cache));assert c['input_sha'].item()==sha(sf) and len(c['centers'])==Q;centers=c['centers'];ids=c['ids']
else:
 km=MiniBatchKMeans(n_clusters=Q,init='k-means++',init_size=3*Q,batch_size=4096,n_init=1,max_iter=50,max_no_improvement=10,reassignment_ratio=.01,random_state=seed);km.fit(z);centers=km.cluster_centers_;ids=pairwise_distances_argmin(z,centers);np.savez_compressed(cache,centers=centers,ids=ids,input_sha=sha(sf),random_seed=seed)
 print('SOURCE_QUANTIZER_DONE',cls,src,round(time.time()-start),flush=True)
a=stats(z,ys,ids,Q);by_source=[stats(z[so[i]:so[i+1]],sc['pred'][so[i]:so[i+1]].astype(int),ids[so[i]:so[i+1]],Q) for i in range(len(so)-1)]
calrows=[];validation=[]
# Original auxiliary images and source-only responses are reused, now with all
# Full2048 source/auxiliary pixels and the new numerical transport scheme.
for j,r in enumerate(meta['calibration'].get('rows',[])):
 target=aggregate(by_source,r['indices']);component,detail=transport(a,target);calrows.append(dict(role=r['role'],indices=r['indices'],AD=r['AD'],W=float(component[0]),CE=float(component[1]),**detail))
 if r['role']=='check':
  rng=np.random.default_rng(seed+10000+j);ii=rng.choice(len(z),256,False);pool=np.concatenate([np.arange(so[i],so[i+1]) for i in r['indices']]);jj=rng.choice(pool,256,False);cost=cdist(z[ii].astype(float),z[jj].astype(float),'sqeuclidean');u,v=linear_sum_assignment(cost);joint=np.zeros((2,2));np.add.at(joint,(ys[ii[u]],sc['pred'][jj[v]].astype(int)),1/256);mass=joint.sum(1);ce=-sum(joint[i,k]*np.log(joint[i,k]/mass[i]) for i in range(2) for k in range(2) if joint[i,k]>0);exact=float(cost[u,v].mean());comp,diag=transport(stats(z[ii],ys[ii],ids[ii],Q),stats(z[jj],sc['pred'][jj].astype(int),ids[jj],Q));assert comp[0]>=exact-1e-8;validation.append(dict(source_images=r['indices'],exact_256_W=exact,approx_256_W=float(comp[0]),relative_W_error=float((comp[0]-exact)/max(exact,1e-12)),exact_256_CE=float(ce),approx_256_CE=float(comp[1]),absolute_CE_error=float(abs(comp[1]-ce)),**diag))
 print('SOURCE_CAL',cls,src,j+1,round(time.time()-start),flush=True)
train=[r for r in calrows if r['role']=='fit' and np.isfinite(r['AD'])];check=[r for r in calrows if r['role']=='check' and np.isfinite(r['AD'])];cal=None
if len(train)>=3:
 xx=np.array([[r['W'],r['CE']] for r in train]);yy=np.array([r['AD'] for r in train]);center=xx.mean(0);scale=np.maximum(xx.std(0),1e-8);xx=(xx-center)/scale;inter=yy.mean();coef=np.linalg.solve(xx.T@xx+1e-3*np.eye(2),xx.T@(yy-inter));cal=dict(center=center.tolist(),scale=scale.tolist(),intercept=float(inter),coef=coef.tolist(),ridge_lambda=.001,defined_fit=len(train),defined_check=len(check));cal['source_check_MAE']=float(np.mean([abs(inter+((np.array([r['W'],r['CE']])-center)/scale)@coef-r['AD']) for r in check])) if check else None
rows=[r for r in read(OLD/'artifacts/predictions_full.csv') if int(r['cls'])==cls and r['src']==src and r['panel']=='full2048'];answer=[];inputs={str(sf):sha(sf)};maxmargin=0.
for tgt in sorted({r['tgt'] for r in rows}):
 tf=OLD/'artifacts/features'/f'{cls}_{src}__{tgt}.npz';tc=dict(np.load(tf));inputs[str(tf)]=sha(tf);z=tc['z'];p=tc['pred'].astype(int);off=tc['offsets'];cache=temp/f'{cls}_{src}__{tgt}.npz'
 if cache.exists():
  c=dict(np.load(cache));assert c['input_sha'].item()==sha(tf) and c['center_sha'].item()==sha(temp/f'{cls}_{src}_source.npz');ids=c['ids']
 else:
  ids=pairwise_distances_argmin(z,centers);np.savez_compressed(cache,ids=ids,input_sha=sha(tf),center_sha=sha(temp/f'{cls}_{src}_source.npz'))
 by_target=[stats(z[off[i]:off[i+1]],p[off[i]:off[i+1]],ids[off[i]:off[i+1]],Q) for i in range(len(off)-1)]
 for j,r in enumerate(rr for rr in rows if rr['tgt']==tgt):
  component,diag=transport(a,aggregate(by_target,json.loads(r['indices'])));maxmargin=max(maxmargin,diag['marginal_error']);score=cal['intercept']+((component-np.array(cal['center']))/np.array(cal['scale']))@cal['coef'] if cal else np.nan
  answer.append({**{k:r[k] for k in ['cls','src','tgt','panel','kind','budget','seed','repeat','indices']},'OTCE_PL_SrcCal':float(score),**diag})
  if (j+1)%10==0:print('OT_SCORES',cls,src,tgt,j+1,round(time.time()-start),flush=True)
 print('OT_TASK_DONE',cls,src,tgt,round(time.time()-start),flush=True)
save(out/f'{cls}_{src}.csv',answer);dump(out/f'{cls}_{src}_source_validation.json',validation);dump(out/f'{cls}_{src}_calibration.json',dict(calibration=cal,rows=calrows,quantizer_cells=Q,quantizer_source_only=True,random_seed=seed));dump(out/f'{cls}_{src}_audit.json',dict(records=len(answer),approximation='source-fitted Q1024 cells; exact weighted transport of centroids; feasible product lift, including within-cell variance cost and class proportions',full_input_points_used=True,target_reference_reads=0,guard_probe_blocked=True,unexpected_denials=denied[1:],inputs=inputs,max_marginal_error=maxmargin,seconds=time.time()-start,code_sha={p.name:sha(p) for p in [B/'code/common.py',B/'code/ot_quantized.py',Path(__file__)]}));print('OT_WORKER_DONE',cls,src,round(time.time()-start),flush=True)
