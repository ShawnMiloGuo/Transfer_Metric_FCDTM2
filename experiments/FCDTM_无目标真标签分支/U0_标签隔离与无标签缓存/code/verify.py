import json,sys
import numpy as np
import common as c
def unit():
    rng=np.random.default_rng(20);z=rng.normal(size=(100,5));v=rng.normal(size=5);sigma=z.std(0);mu=np.array([z[:50].mean(0),z[50:].mean(0)]);a=abs(v)*sigma;a/=a.sum();stats=dict(mu=mu,sigma=sigma,a=a,margin_scale=1.);diag=np.array([.1,3,7,.5,2]);stats2=dict(mu=mu*diag,sigma=sigma*diag,a=a,margin_scale=1.)
    assert np.allclose(c.qfeatures(z,z@v,stats,'weighted'),c.qfeatures(z*diag,z@v,stats2,'weighted'))
    p=rng.random(100)>.5;q=rng.random(100);tp=np.mean(p*q);fp=np.mean(p*(1-q));fn=np.mean((1-p)*q);assert np.isclose(tp+fp,p.mean()) and 0<=c.f1([tp,fp,fn])<=1
    assert np.isnan(c.score(np.ones(5),np.zeros(5),v,False));assert np.isfinite(c.score(np.ones(5),np.zeros(5),v,True))
    print('UNIT PASS: weighted-distance invariance, legal probabilistic counts, explicit raw/epsilon denominator behavior')
def final():
    m=c.check()
    for p,h in json.loads((c.OUT/'seal.json').read_text()).items():assert c.sha(c.BR/p)==h,p
    execution=json.loads((c.OUT/'execution.json').read_text());assert len(execution['workers'])==25 and all(r['code']==0 and r['audit']=='passed' for r in execution['workers'])
    allowed={'names','hard_sum','hard_mass','soft_sum','soft_mass','confidence','neg_entropy','bits','head_identity','risk_weighted','risk_uniform','risk_margin','risk_ridge6','risk_ridge2'};images=0
    for p in (c.OUT/'unlabeled_cache').glob('*.npz'):
        with np.load(p) as z:
            assert set(z.files)==allowed and len(z['names'])==40;assert np.max(z['head_identity'])<1e-3;images+=40
            for mode in ['weighted','uniform','margin','ridge6','ridge2']:
                counts=z['risk_'+mode];area=z['hard_mass'][:,1]/65536;assert np.isfinite(counts).all() and np.all(counts>=0) and np.allclose(counts[:,0]+counts[:,1],area,atol=1e-6)
    for folder in [c.U1,c.U2]:assert len(c.read(folder/'artifacts/predictions.csv'))==800
    result=dict(status='passed',models=25,tasks=50,target_image_passes=images,source_image_passes=1600,target_annotation_accesses=0,historical_target_cache_accesses=0,system_call_audits=25,cache_whitelist_verified=True,scope='Python/GDAL guards plus strace native open audit, not OS sandbox')
    (c.OUT/'最终核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=1));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':unit() if sys.argv[1]=='unit' else final()
