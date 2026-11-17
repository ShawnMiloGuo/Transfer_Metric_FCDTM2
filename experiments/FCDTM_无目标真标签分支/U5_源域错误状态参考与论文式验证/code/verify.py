import json,sys
import numpy as np
import common as c
def unit():
    rng=np.random.default_rng(5);z=rng.normal(size=(80,4));p=np.r_[np.zeros(40,bool),np.ones(40,bool)];y=p.copy();y[0:2]=True;y[40:43]=False;proto,n=c.prototype(z,p,y);assert n.tolist()==[[38,2],[37,3]]
    for side in [0,1]:
        for error in [0,1]:
            selected=(p==side)&((p!=y)==bool(error));expected=(z[selected].sum(0)+512*z[p==side].mean(0))/(n[side,error]+512);assert np.allclose(proto[side,error],expected)
    stats=dict(sigma=np.ones(4),a=np.array([.1,.2,.3,.4]),margin_scale=1.)
    x=c.error_features(z,np.zeros(80),p,stats,proto,'error_weighted');assert x.shape==(80,3) and np.isfinite(x).all()
    diag=np.array([.5,3,10,2]);stats2=dict(sigma=diag,a=stats['a'],margin_scale=1.);assert np.allclose(x,c.error_features(z*diag,np.zeros(80),p,stats2,proto*diag,'error_weighted'))
    e=rng.random(80);q=np.where(p,1-e,e);counts=[np.mean(p*q),np.mean(p*(1-q)),np.mean((1-p)*q)];assert np.isclose(counts[0]+counts[1],p.mean()) and 0<=c.f1(counts)<=1
    print('UNIT PASS: state mapping/shrinkage, conditional indexing, scale invariance, legal error probabilities')
def final():
    m=c.check();c.check_seal();execution=json.loads((c.OUT/'execution.json').read_text());assert len(execution['workers'])==25 and all(r['code']==0 and r['audit']=='passed' for r in execution['workers']);rows=c.read(c.OUT/'predictions.csv');assert len(rows)==800;by={}
    for r in rows:by.setdefault((int(r['cls']),r['src'],r['tgt']),[]).append(r)
    maximum=0.
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt'];source=json.loads((c.OUT/f'source/{cls}_{src}.json').read_text());sf=source['source_F1'];mu=np.array(source['source_true_mu']);w=np.array(source['head_weight'])
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z:
            assert not any(k in z.files for k in ['truth','cal_truth','conf','labels'])
            assert z['names'].tolist()==[r['name'] for r in m['pools'][tgt]]
            for method in c.RISK:
                if method=='source_only':continue
                cc=z['risk_'+method];assert np.isfinite(cc).all() and np.all(cc>=0);assert np.allclose(cc[:,0]+cc[:,1],z['hard_mass'][:,1]/65536,atol=1e-6)
            for r in by[(cls,src,tgt)]:
                ix=np.array(json.loads(r['indices']));expected=next(inds for b,rep,inds in c.draws(tgt) if b==int(r['budget']) and rep==int(r['repeat']));assert np.array_equal(ix,expected)
                for method in c.RISK:
                    value=sf if method=='source_only' else c.f1(z['risk_'+method][ix]);assert np.isclose(value,float(r[method]),equal_nan=True,atol=1e-12);assert np.isclose(float(r[method+'_AD']),sf-value,equal_nan=True,atol=1e-12)
                    if np.isfinite(value):maximum=max(maximum,abs(value-float(r[method])))
                mass=z['hard_mass'][ix,1].sum();value=c.u0.score(z['hard_sum'][ix,1].sum(0)/mass,mu,w) if mass else np.nan;assert np.isclose(value,float(r['hard']),equal_nan=True,atol=1e-9)
    result=dict(status='passed',models=25,tasks=50,pool_predictions=800,source_images=1600,target_images=2000,access_audits=25,target_annotation_accesses=0,max_replay_difference=maximum,source_only_error_prototypes=True)
    (c.OUT/'最终核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=1));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':unit() if sys.argv[1]=='unit' else final()
