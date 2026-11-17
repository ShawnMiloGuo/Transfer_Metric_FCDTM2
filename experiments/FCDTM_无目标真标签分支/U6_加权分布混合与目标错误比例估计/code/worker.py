"""Source labels only; each worker is audited by native open/openat trace."""
import os,sys,json
from pathlib import Path
import numpy as np
import common as c

def blocks(h,names):
    keys=[c.u0.grid(str(n)) for n in names];return np.array([h[np.array(keys)==k].sum(0) for k in sorted(set(keys))])

def main(cls,source):
    m=c.check(cls,source);t=next(t for t in m['models'] if (t['cls'],t['src'])==(cls,source))
    import torch
    from osgeo import gdal
    torch.set_num_threads(2);torch.set_num_interop_threads(1);gdal.UseExceptions();sys.path.insert(0,str(c.REPO));assert c.sha(t['ck'])==t['sha']
    srcfile=c.U0/f'artifacts/source/{cls}_{source}.json';oldfile=c.U5/f'artifacts/source/{cls}_{source}.json';cfg=json.loads(srcfile.read_text());oldcfg=json.loads(oldfile.read_text());stats={k:np.array(v) for k,v in cfg['stats'].items()};proto=np.array(oldcfg['prototype'])
    allowed={str(c.DATA/source/'train_val/annotations/val'/n) for n in t['ref']}
    oldcaches={str(c.U5/f"artifacts/unlabeled_cache/{cls}_{source}__{tt['tgt']}.npz") for tt in m['tasks'] if (tt['cls'],tt['src'])==(cls,source)}
    allowed_data={str(srcfile),str(oldfile)}|oldcaches;denied=[]
    def guard(path):
        if not isinstance(path,(str,bytes,Path)):return
        p=str(Path(os.fsdecode(path)).absolute())
        bad=('/annotations/' in p and p not in allowed) or any(s in p for s in ['逐次评价','评价真值','/U3_','/首轮实验综合分析'])
        if '/experiments/' in p and not p.endswith(('.py','.pyc')) and not p.startswith(str(c.BASE)) and p not in allowed_data:bad=True
        if bad:denied.append(p);raise PermissionError('U6 predictor blocked evaluation or target labels')
    sys.addaudithook(lambda event,args:guard(args[0]) if event=='open' else None);original=gdal.Open
    def safeopen(path,*a,**kw):guard(path);return original(path,*a,**kw)
    gdal.Open=safeopen
    target=next(tt['tgt'] for tt in m['tasks'] if (tt['cls'],tt['src'])==(cls,source));r=m['pools'][target][0]
    for p in [c.DATA/target/'train_val/annotations'/r['split']/r['name'],c.U5/'artifacts/batch4/散点数据.csv']:
        try:
            if str(p).endswith('.tif'):gdal.Open(str(p))
            else:open(p,'rb')
            raise AssertionError('Access guard failed')
        except PermissionError:pass
    model=torch.load(t['ck'],map_location='cpu',weights_only=False).eval();holder={};hook=model.outc.register_forward_pre_hook(lambda mod,args:holder.update(z=args[0]));v=(model.outc.weight[1]-model.outc.weight[0]).detach().flatten().double().numpy();bias=float((model.outc.bias[1]-model.outc.bias[0]).detach())
    def infer(domain,r):
        x=gdal.Open(str(c.DATA/domain/'train_val/images'/r['split']/r['name'])).ReadAsArray().astype('float32');assert x.shape==(4,256,256)
        with torch.inference_mode():logits=model(torch.from_numpy(x)[None])[0];z=holder.pop('z')[0].numpy().reshape(len(v),-1).T;margin=(logits[1]-logits[0]).numpy().ravel()
        assert np.isfinite(z).all() and np.isfinite(margin).all();assert abs(z@v+bias-margin).max()<1e-3
        return z,margin,margin>0
    zs=[];ms=[];ps=[];ys=[]
    for i,name in enumerate(t['ref']):
        z,margin,pred=infer(source,dict(name=name,split='val'));y=gdal.Open(str(c.DATA/source/'train_val/annotations/val'/name)).ReadAsArray().ravel()==cls;ix=np.random.default_rng(200116+cls*1000+i).choice(65536,512,replace=False);zs.append(z[ix]);ms.append(margin[ix]);ps.append(pred[ix]);ys.append(y[ix])
        if (i+1)%16==0:print('SOURCE',cls,source,i+1,flush=True)
    zs=np.array(zs);ms=np.array(ms);ps=np.array(ps);ys=np.array(ys);fit=np.array([n in cfg['fit_names'] for n in t['ref']]);cal=np.array([n in cfg['cal_names'] for n in t['ref']]);test=~(fit|cal)
    recomputed,counts=c.u5.prototype(zs[fit].reshape(-1,len(v)),ps[fit].ravel(),ys[fit].ravel());assert np.allclose(recomputed,proto,rtol=1e-12,atol=1e-12)
    sources={};params={};testrows=[]
    for mode in c.MODES:
        x=c.features(zs.reshape(-1,len(v)),ms.ravel(),ps.ravel(),stats,proto,mode).reshape(64,512,3);edges=c.bin_edges(x[fit].reshape(-1,3),ps[fit].ravel());h=np.array([c.hist(c.bin_ids(xx,pp,edges),pp,pp!=yy) for xx,pp,yy in zip(x,ps,ys)]);sources[mode]=h;params[mode]=c.calibrate(h,fit,cal);params[mode]['edges']=edges;params[mode]['fit_support']=counts
        for i in np.flatnonzero(test):
            state=h[i];actual=np.array([state[1,0,0].sum(),state[1,1,0].sum(),state[0,1,0].sum()]);forecast,d=c.mixture(state.sum(1),h[fit].sum(0),params[mode]['lam']);testrows.append(dict(mode=mode,name=t['ref'][i],true_F1=c.f1(actual),predicted_F1=c.f1(forecast),absolute_error=abs(c.f1(actual)-c.f1(forecast))))
    sourceout=c.OUT/'source';sourceout.mkdir(exist_ok=True);(sourceout/f'{cls}_{source}.json').write_text(json.dumps(dict(source_F1=cfg['source_F1'],params=params,fit_names=cfg['fit_names'],cal_names=cfg['cal_names'],test_names=cfg['test_names'],test_predictions=testrows),default=c.u0.serialize,indent=1));np.savez_compressed(sourceout/f'{cls}_{source}.npz',names=np.array(t['ref']),fit=fit,cal=cal,test=test,**sources)
    cache=c.OUT/'unlabeled_cache';cache.mkdir(exist_ok=True);rows=[];diagnostics=[]
    for target in [tt['tgt'] for tt in m['tasks'] if (tt['cls'],tt['src'])==(cls,source)]:
        histograms={mode:[] for mode in c.MODES};samples={mode:[] for mode in c.MODES};sample_ix=[]
        with np.load(c.U5/f'artifacts/unlabeled_cache/{cls}_{source}__{target}.npz') as old:
            assert old['names'].tolist()==[r['name'] for r in m['pools'][target]]
            for i,r in enumerate(m['pools'][target]):
                z,margin,pred=infer(target,r);assert np.array_equal(np.packbits(pred),old['bits'][i]);ix=np.random.default_rng(260117+cls*1000+c.DOM.index(source)*100+c.DOM.index(target)*40+i).choice(65536,512,replace=False);sample_ix.append(ix)
                for mode in c.MODES:
                    x=c.features(z,margin,pred,stats,proto,mode);ids=c.bin_ids(x,pred,np.array(params[mode]['edges']));histograms[mode].append(c.hist(ids,pred));samples[mode].append(ids[ix])
                if (i+1)%16==0:print('TARGET',cls,source,target,i+1,flush=True)
            histograms={k:np.array(v) for k,v in histograms.items()};np.savez_compressed(cache/f'{cls}_{source}__{target}.npz',names=old['names'],bits=old['bits'],sample_indices=np.array(sample_ix,dtype=np.int32),**{'hist_'+k:v for k,v in histograms.items()},**{'sample_bins_'+k:np.array(v) for k,v in samples.items()})
            for kind,budget,repeat,ix in c.subsets(m,target):
                sf=cfg['source_F1'];rr=dict(cls=cls,src=source,tgt=target,kind=kind,budget=budget,repeat=repeat,indices=json.dumps(ix.tolist()),source_F1=sf,source_only=sf,confidence=float(old['confidence'][ix].mean()))
                for mode in c.MODES:
                    state=sources[mode][fit].sum(0);h=histograms[mode][ix].sum(0);forecast,d=c.mixture(h,state,params[mode]['lam']);rr['mix_'+mode]=c.f1(forecast);rr['mix_'+mode+'_raw']=c.f1(c.mixture(h,state,0.)[0])
                    dr=dict(cls=cls,src=source,tgt=target,kind=kind,budget=budget,repeat=repeat,mode=mode,lam=params[mode]['lam'],**{'side'+str(s)+'_'+k:v for s,dd in enumerate(d) for k,v in dd.items()})
                    if kind=='batch4':
                        a=blocks(histograms[mode][ix],old['names'][ix]);b=blocks(sources[mode][fit],np.array(t['ref'])[fit]);dr.update(c.bootstrap(a,b,params[mode]['lam'],260217+cls*1000+c.DOM.index(source)*100+c.DOM.index(target)*10+repeat));dr.update(target_spatial_blocks=len(a),source_spatial_blocks=len(b))
                    diagnostics.append(dr)
                for method,oldmode in [('U5_weighted','error_weighted'),('global_margin','global_margin'),('global_weighted','global_weighted'),('ridge6','ridge6')]:rr[method]=c.f1(old['risk_'+oldmode][ix])
                n=old['hard_mass'][ix,1].sum();rr['hard']=c.u0.score(old['hard_sum'][ix,1].sum(0)/n,np.array(oldcfg['source_true_mu']),np.array(oldcfg['head_weight'])) if n else np.nan;mass=old['soft_mass'][ix].sum();rr['soft']=c.u0.score(old['soft_sum'][ix].sum(0)/mass,np.array(oldcfg['source_true_mu']),np.array(oldcfg['head_weight'])) if mass>1e-8 else np.nan
                for method in c.RISK:rr[method+'_AD']=sf-rr[method]
                rows.append(rr)
    c.save(c.OUT/f'worker_{cls}_{source}.csv',rows)
    (c.OUT/f'diagnostics_{cls}_{source}.json').write_text(json.dumps(diagnostics,default=c.u0.serialize,indent=1));(c.OUT/f'guard_{cls}_{source}.json').write_text(json.dumps(dict(status='completed',denied_probes=denied,source_whitelist=sorted(allowed)),indent=1));hook.remove();print('DONE',cls,source,flush=True)
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
