import json,sys,os
from pathlib import Path
import numpy as np
import common as c
def main(cls,source):
    m=c.check(cls,source);t=next(t for t in m['models'] if (t['cls'],t['src'])==(cls,source))
    import torch
    from osgeo import gdal
    torch.set_num_threads(2);torch.set_num_interop_threads(1);gdal.UseExceptions();sys.path.insert(0,str(c.REPO));assert c.sha(t['ck'])==t['sha']
    sourcefile=c.SRC/f'artifacts/source/{cls}_{source}.json';sourcecfg=json.loads(sourcefile.read_text());stats={k:np.array(v) for k,v in sourcecfg['stats'].items()};allowed={str(c.DATA/source/'train_val/annotations/val'/n) for n in t['ref']};denied=[]
    allowed_data={str(sourcefile)}|{str(c.SRC/f"artifacts/unlabeled_cache/{cls}_{source}__{tt['tgt']}.npz") for tt in m['tasks'] if (tt['cls'],tt['src'])==(cls,source)}
    def guard(path):
        if not isinstance(path,(str,bytes,Path)):return
        p=str(Path(os.fsdecode(path)).absolute())
        bad=('/annotations/' in p and p not in allowed) or any(s in p for s in ['逐次评价','评价真值','/U3_','/首轮实验综合分析'])
        if '/experiments/' in p and not p.endswith(('.py','.pyc')) and not p.startswith(str(c.BASE)) and p not in allowed_data:bad=True
        if bad:denied.append(p);raise PermissionError('Unauthorized evaluation/target data blocked')
    sys.addaudithook(lambda event,args:guard(args[0]) if event=='open' else None);original=gdal.Open
    def safeopen(path,*args,**kw):guard(path);return original(path,*args,**kw)
    gdal.Open=safeopen
    target=next(tt['tgt'] for tt in m['tasks'] if (tt['cls'],tt['src'])==(cls,source));record=m['pools'][target][0]
    for p in [c.DATA/target/'train_val/annotations'/record['split']/record['name'],c.SRC/'artifacts/评价真值.csv']:
        try:
            if str(p).endswith('.tif'):gdal.Open(str(p))
            else:open(p,'rb')
            raise AssertionError('Guard failed')
        except PermissionError:pass
    model=torch.load(t['ck'],map_location='cpu',weights_only=False).eval();holder={};hook=model.outc.register_forward_pre_hook(lambda mod,args:holder.update(z=args[0]));v=(model.outc.weight[1]-model.outc.weight[0]).detach().flatten().double().numpy();b=float((model.outc.bias[1]-model.outc.bias[0]).detach());weight=-10*v-b
    def infer(domain,r):
        x=gdal.Open(str(c.DATA/domain/'train_val/images'/r['split']/r['name'])).ReadAsArray().astype('float32');assert x.shape==(4,256,256)
        with torch.inference_mode():logits=model(torch.from_numpy(x)[None])[0];z=holder.pop('z')[0].numpy().reshape(len(v),-1).T;mf=(logits[1]-logits[0]).numpy().ravel();prob=logits.softmax(0)[1].numpy().ravel()
        assert np.isfinite(z).all() and np.isfinite(mf).all();assert abs(z@v+b-mf).max()<1e-3
        return z,mf,prob,mf>0
    zs=[];ys=[];ms=[];ps=[];true_sum=np.zeros(len(v));true_n=0
    for i,name in enumerate(t['ref']):
        z,margin,prob,pred=infer(source,dict(name=name,split='val'));y=gdal.Open(str(c.DATA/source/'train_val/annotations/val'/name)).ReadAsArray().ravel()==cls;true_sum+=z[y].sum(0,dtype='float64');true_n+=int(y.sum());ix=np.random.default_rng(200116+cls*1000+i).choice(65536,512,replace=False);zs.append(z[ix]);ys.append(y[ix]);ms.append(margin[ix]);ps.append(pred[ix])
        if (i+1)%16==0:print('SOURCE',cls,source,i+1,flush=True)
    zs=np.array(zs);ys=np.array(ys);ms=np.array(ms);ps=np.array(ps);fit=np.array([n in sourcecfg['fit_names'] for n in t['ref']]);cal=np.array([n in sourcecfg['cal_names'] for n in t['ref']]);test=~(fit|cal);zfit=zs[fit].reshape(-1,len(v));yfit=ys[fit].ravel();pfit=ps[fit].ravel();proto,counts=c.prototype(zfit,pfit,yfit);models={};diagnostics=[]
    for mode in ['error_weighted','error_uniform','conditional_margin']:
        x=c.error_features(zs.reshape(-1,len(v)),ms.ravel(),ps.ravel(),stats,proto,mode).reshape(64,512,-1);models[mode]=[]
        for side in [0,1]:
            xf=x[fit].reshape(-1,x.shape[-1]);yc=ys[cal].ravel();pc=ps[cal].ravel();pf=pfit==side;cf=pc==side;error=(pfit!=yfit).astype(float);xc=x[cal].reshape(-1,x.shape[-1]);ec=(pc!=yc).astype(float)
            if not pf.any():m0=dict(constant=float((error.sum()+.5)/(len(error)+1)),status='empty_source_prediction_side')
            else:m0=c.u0.fitq(xf[pf],error[pf],xc[cf],ec[cf])
            models[mode].append(m0);xt=x[test].reshape(-1,x.shape[-1]);pt=ps[test].ravel();et=(pt!=ys[test].ravel()).astype(float);mask=pt==side;q=c.predict_error(xt[mask],m0);diagnostics.append(dict(mode=mode,side=side,fit_support=int(pf.sum()),cal_support=int(cf.sum()),test_support=int(mask.sum()),Brier=float(((q-et[mask])**2).mean()) if mask.any() else np.nan,status=m0['status']))
    sf=sourcecfg['source_F1'];mu=true_sum/true_n if true_n else np.full(len(v),np.nan);sourceout=c.OUT/'source';sourceout.mkdir(exist_ok=True);(sourceout/f'{cls}_{source}.json').write_text(json.dumps(dict(source_F1=sf,prototype=proto,state_counts=counts,models=models,validation=diagnostics,source_true_mu=mu,head_weight=weight,fit_names=sourcecfg['fit_names'],cal_names=sourcecfg['cal_names'],test_names=sourcecfg['test_names']),default=c.u0.serialize,indent=1))
    cache=c.OUT/'unlabeled_cache';cache.mkdir(exist_ok=True);rows=[]
    for target in [tt['tgt'] for tt in m['tasks'] if (tt['cls'],tt['src'])==(cls,source)]:
        arrays=[]
        with np.load(c.SRC/f'artifacts/unlabeled_cache/{cls}_{source}__{target}.npz') as old:
            assert old['names'].tolist()==[r['name'] for r in m['pools'][target]]
            for i,r in enumerate(m['pools'][target]):
                z,margin,prob,pred=infer(target,r);assert np.array_equal(np.packbits(pred),old['bits'][i]);data=dict(bits=np.packbits(pred),confidence=float(old['confidence'][i]),hard_sum=np.array([z[~pred].sum(0,dtype='float64'),z[pred].sum(0,dtype='float64')]),hard_mass=np.array([(~pred).sum(),pred.sum()]),soft_sum=(prob[:,None]*z).sum(0,dtype='float64'),soft_mass=float(prob.sum(dtype='float64')))
                for mode in models:
                    x=c.error_features(z,margin,pred,stats,proto,mode);error=np.empty(len(pred))
                    for side in [0,1]:mask=pred==side;error[mask]=c.predict_error(x[mask],models[mode][side])
                    q=np.where(pred,1-error,error);data['risk_'+mode]=np.array([(pred*q).mean(),(pred*(1-q)).mean(),((1-pred)*q).mean()])
                for mode,oldmode in [('global_margin','margin'),('global_weighted','weighted'),('global_uniform','uniform'),('ridge6','ridge6')]:data['risk_'+mode]=old['risk_'+oldmode][i]
                hm=data['hard_sum'][1]/data['hard_mass'][1] if data['hard_mass'][1] else np.full(len(v),np.nan);sm=data['soft_sum']/data['soft_mass'] if data['soft_mass']>1e-8 else np.full(len(v),np.nan);data['hard']=c.u0.score(hm,mu,weight);data['soft']=c.u0.score(sm,mu,weight);arrays.append(data)
                if (i+1)%16==0:print('TARGET',cls,source,target,i+1,flush=True)
        np.savez_compressed(cache/f'{cls}_{source}__{target}.npz',names=np.array([r['name'] for r in m['pools'][target]]),**{k:np.array([r[k] for r in arrays]) for k in arrays[0]})
        for budget,repeat,ix in c.draws(target):
            rr=dict(cls=cls,src=source,tgt=target,budget=budget,repeat=repeat,indices=json.dumps(ix.tolist()),source_F1=sf,source_only=sf,confidence=float(np.mean([arrays[i]['confidence'] for i in ix])))
            for mode in c.RISK:
                if mode!='source_only':rr[mode]=c.f1([arrays[i]['risk_'+mode] for i in ix])
                rr[mode+'_AD']=sf-rr[mode]
            n=sum(arrays[i]['hard_mass'][1] for i in ix);rr['hard']=c.u0.score(sum(arrays[i]['hard_sum'][1] for i in ix)/n,mu,weight) if n else np.nan;mass=sum(arrays[i]['soft_mass'] for i in ix);rr['soft']=c.u0.score(sum(arrays[i]['soft_sum'] for i in ix)/mass,mu,weight) if mass>1e-8 else np.nan;rows.append(rr)
    hook.remove();c.save(c.OUT/f'worker_{cls}_{source}.csv',rows);(c.OUT/f'guard_{cls}_{source}.json').write_text(json.dumps(dict(denied_probes=denied,source_whitelist=sorted(allowed),status='completed'),indent=1));print('DONE',cls,source,flush=True)
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
