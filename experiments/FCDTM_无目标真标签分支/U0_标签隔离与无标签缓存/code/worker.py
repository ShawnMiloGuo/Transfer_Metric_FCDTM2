import json,sys
from pathlib import Path
import numpy as np
import common as c
def main(cls,source):
    m=c.check();t=next(t for t in m['models'] if (t['cls'],t['src'])==(cls,source));assert c.sha(t['ck'])==t['sha']
    import torch
    from osgeo import gdal
    torch.set_num_threads(2);torch.set_num_interop_threads(1);gdal.UseExceptions();sys.path.insert(0,str(c.REPO))
    allowed={str(c.DATA/source/'train_val/annotations/val'/n) for n in t['ref']};events=[]
    def guard(path):
        if not isinstance(path,(str,bytes,Path)):return
        p=str(Path(path).absolute())
        if '/annotations/' in p and p not in allowed:events.append(dict(path=p,status='denied'));raise PermissionError('Target/unauthorized annotation blocked')
        if '/experiments/' in p and not p.startswith(str(c.BR)) and not p.endswith(('.py','.pyc')):events.append(dict(path=p,status='denied'));raise PermissionError('Historical experiment data blocked')
    sys.addaudithook(lambda event,args:guard(args[0]) if event=='open' else None)
    original=gdal.Open
    def safeopen(path,*args,**kwargs):guard(path);return original(path,*args,**kwargs)
    gdal.Open=safeopen
    target=next(d for d in c.DOM if d!=source)
    for p in [c.DATA/target/'train_val/annotations/train/probe.tif',c.REPO/'experiments/E18_E13融合模型权重分布差异/artifacts/cache/probe.npz']:
        try:
            if p.suffix=='.tif':gdal.Open(str(p))
            else:open(p,'rb')
            raise AssertionError('Leak guard failed')
        except PermissionError:pass
    model=torch.load(t['ck'],map_location='cpu',weights_only=False).eval();holder={};hook=model.outc.register_forward_pre_hook(lambda mod,args:holder.update(z=args[0]))
    v=(model.outc.weight[1]-model.outc.weight[0]).detach().flatten().double().numpy();b=float((model.outc.bias[1]-model.outc.bias[0]).detach());weight=-10*v-b
    def infer(domain,r,source_label=False):
        root=c.DATA/domain/'train_val';x=gdal.Open(str(root/'images'/r['split']/r['name'])).ReadAsArray().astype('float32');assert x.shape==(4,256,256) and np.isfinite(x).all()
        with torch.inference_mode():logits=model(torch.from_numpy(x)[None])[0];z=holder.pop('z')[0].numpy();margin=(logits[1]-logits[0]).numpy();prob=logits.softmax(0)[1].numpy()
        assert np.isfinite(margin).all();pred=margin>0;zz=z.reshape(len(v),-1).T;mf=margin.ravel();pf=prob.ravel();p=pred.ravel();identity=float(abs(zz@v+b-mf).max());assert identity<1e-3
        result=dict(hard_sum=np.array([zz[~p].sum(0,dtype='float64'),zz[p].sum(0,dtype='float64')]),hard_mass=np.array([(~p).sum(),p.sum()]),soft_sum=np.array([((1-pf)[:,None]*zz).sum(0,dtype='float64'),(pf[:,None]*zz).sum(0,dtype='float64')]),soft_mass=np.array([(1-pf).sum(dtype='float64'),pf.sum(dtype='float64')]),confidence=float(np.maximum(pf,1-pf).mean()),neg_entropy=float((pf*np.log(np.clip(pf,1e-12,1))+(1-pf)*np.log(np.clip(1-pf,1e-12,1))).mean()),bits=np.packbits(p),head_identity=identity)
        if source_label:
            y=gdal.Open(str(root/'annotations'/r['split']/r['name'])).ReadAsArray();assert y.shape==(256,256) and np.isin(y,np.arange(9)).all();fg=y.ravel()==cls;result.update(truth=np.array([(p&fg).sum(),(p&~fg).sum(),(~p&fg).sum()])/65536,true_fg_sum=zz[fg].sum(0,dtype='float64'),true_fg_mass=int(fg.sum()))
        return result,zz,mf,pf,p
    source_stats=[];samples=[];ys=[];ms=[];full_margins=[];full_probs=[]
    for i,name in enumerate(t['ref']):
        r,z,margin,prob,pred=infer(source,dict(name=name,split='val'),True);source_stats.append(r);full_margins.append(margin);full_probs.append(prob)
        y=gdal.Open(str(c.DATA/source/'train_val/annotations/val'/name)).ReadAsArray().ravel()==cls;ix=np.random.default_rng(200116+cls*1000+i).choice(65536,512,replace=False);samples.append(z[ix]);ys.append(y[ix]);ms.append(margin[ix])
        if (i+1)%16==0:print('SOURCE',cls,source,i+1,flush=True)
    sample=np.array(samples);labels=np.array(ys,dtype=float);margins=np.array(ms);rows=sorted(set(c.grid(n) for n in t['ref']));assert len(rows)>=5
    lo=max(1,int(.6*len(rows)));hi=max(lo+1,int(.8*len(rows)));groups=np.array([c.grid(n) for n in t['ref']]);fit=np.isin(groups,rows[:lo]);cal=np.isin(groups,rows[lo:hi]);test=~(fit|cal);assert min(fit.sum(),cal.sum(),test.sum())>=1
    zfit=sample[fit].reshape(-1,len(v));yfit=labels[fit].ravel();sigma=zfit.std(0);sigma=np.maximum(sigma,max(sigma.mean()*.001,1e-6));a=abs(v)*sigma;a=a/a.sum() if a.sum() else np.ones(len(v))/len(v)
    mu=np.array([zfit[yfit==k].mean(0) if (yfit==k).any() else zfit.mean(0) for k in [0,1]])
    stats=dict(mu=mu,sigma=sigma,a=a,margin_scale=max(float(margins[fit].std()),1e-6));qmodels={};validation=[]
    for mode in ['weighted','uniform','margin']:
        x=c.qfeatures(sample.reshape(-1,len(v)),margins.ravel(),stats,mode).reshape(64,512,-1);qmodels[mode]=c.fitq(x[fit].reshape(-1,x.shape[-1]),labels[fit].ravel(),x[cal].reshape(-1,x.shape[-1]),labels[cal].ravel());q=c.predictq(x[test].reshape(-1,x.shape[-1]),qmodels[mode]);yy=labels[test].ravel();validation.append(dict(mode=mode,Brier=float(((q-yy)**2).mean()),CE=float(-(yy*np.log(np.clip(q,1e-12,1))+(1-yy)*np.log(np.clip(1-q,1e-12,1))).mean()),status=qmodels[mode]['status']))
    sourceF1=c.f1([r['truth'] for r in source_stats]);scale=max(float(np.diff(np.quantile(margins.ravel(),[.25,.75]))[0]),1e-6)
    sx=[c.features(margin,prob,scale) for margin,prob in zip(full_margins,full_probs)]
    sx=np.array(sx);counts=np.array([r['truth'] for r in source_stats]);rmodels={k:c.fitridge(sx[fit],counts[fit],int(k[-1])) for k in ['ridge6','ridge2']}
    n=sum(r['true_fg_mass'] for r in source_stats);mu_true=sum(r['true_fg_sum'] for r in source_stats)/n if n else np.full(len(v),np.nan)
    mus={}
    for kind in ['hard','soft']:
        masses=sum(r[kind+'_mass'] for r in source_stats);sums=sum(r[kind+'_sum'] for r in source_stats);mus[kind]=np.divide(sums,masses[:,None],out=np.full_like(sums,np.nan),where=masses[:,None]>1e-8)
    dest=c.OUT/'source';dest.mkdir(exist_ok=True);(dest/f'{cls}_{source}.json').write_text(json.dumps(dict(source_F1=sourceF1,stats=stats,qmodels=qmodels,ridge=rmodels,validation=validation,fit_names=[n for n,x in zip(t['ref'],fit) if x],cal_names=[n for n,x in zip(t['ref'],cal) if x],test_names=[n for n,x in zip(t['ref'],test) if x],missing_source_class=not all((yfit==k).any() for k in [0,1]),prototype_floors=int((abs(mu_true)<1e-6).sum())),default=c.serialize,indent=1))
    score_rows=[];risk_rows=[];cache=c.OUT/'unlabeled_cache';cache.mkdir(exist_ok=True)
    for target in [tt['tgt'] for tt in m['tasks'] if (tt['cls'],tt['src'])==(cls,source)]:
        data=[]
        for i,r in enumerate(m['pools'][target]):
            result,z,margin,prob,pred=infer(target,r);x=c.features(margin,prob,scale)
            for mode in qmodels:
                q=c.predictq(c.qfeatures(z,margin,stats,mode),qmodels[mode]);result['risk_'+mode]=np.array([(pred*q).mean(),(pred*(1-q)).mean(),((1-pred)*q).mean()])
            for mode in rmodels:result['risk_'+mode]=c.ridge(x,rmodels[mode])
            data.append(result)
            if (i+1)%16==0:print('TARGET',cls,source,target,i+1,flush=True)
        np.savez_compressed(cache/f'{cls}_{source}__{target}.npz',names=np.array([r['name'] for r in m['pools'][target]]),**{k:np.array([r[k] for r in data]) for k in data[0]})
        for budget,repeat,ix in c.draws(target):
            s=dict(cls=cls,src=source,tgt=target,budget=budget,repeat=repeat,indices=json.dumps(ix.tolist()),confidence=float(np.mean([data[i]['confidence'] for i in ix])),neg_entropy=float(np.mean([data[i]['neg_entropy'] for i in ix])))
            means={}
            for kind in ['hard','soft']:
                mass=sum(data[i][kind+'_mass'] for i in ix);sums=sum(data[i][kind+'_sum'] for i in ix);means[kind]=np.divide(sums,mass[:,None],out=np.full_like(sums,np.nan),where=mass[:,None]>1e-8)
            s.update(hard=c.score(means['hard'][1],mu_true,weight),soft=c.score(means['soft'][1],mu_true,weight),matched_hard=c.score(means['hard'][1],mus['hard'][1],weight),matched_soft=c.score(means['soft'][1],mus['soft'][1],weight),uniform_hard=c.score(means['hard'][1],mu_true,-np.ones(len(v))/np.sqrt(len(v))),dual=c.score(means['hard'][1],mus['hard'][1],weight)-c.score(means['hard'][0],mus['hard'][0],weight),raw_hard=c.score(means['hard'][1],mu_true,weight,False));score_rows.append(s)
            rr={k:s[k] for k in ['cls','src','tgt','budget','repeat','indices']};rr['source_F1']=sourceF1;rr['source_only']=sourceF1
            for mode in ['ridge6','ridge2','weighted','uniform','margin']:rr[mode]=c.f1([data[i]['risk_'+mode] for i in ix])
            for mode in c.RISK:rr[mode+'_AD']=sourceF1-rr[mode]
            risk_rows.append(rr)
    hook.remove();c.save(c.U1/f'artifacts/worker_{cls}_{source}.csv',score_rows);c.save(c.U2/f'artifacts/worker_{cls}_{source}.csv',risk_rows)
    (c.OUT/f'guard_{cls}_{source}.json').write_text(json.dumps(dict(probes=events,source_annotation_whitelist=sorted(allowed),status='completed'),indent=1));print('DONE',cls,source,flush=True)
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
