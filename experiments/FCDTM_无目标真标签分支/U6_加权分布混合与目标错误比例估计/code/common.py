"""U6 fixed histogram mixture estimator. No target labels in this module."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1];BR=BASE.parent;OUT=BASE/'artifacts'
U0=BR/'U0_标签隔离与无标签缓存';U5=BR/'U5_源域错误状态参考与论文式验证'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
u0=module('u6u0',U0/'code/common.py');u5=module('u6u5',U5/'code/common.py')
DATA=u0.DATA;REPO=u0.REPO;DOM=u0.DOM;sha=u0.sha;save=u0.save;read=u0.read;f1=u0.f1
MODES=['weighted','uniform'];BINS=8;LAMS=[0.,.001,.01,.1,1.];BOOT=100
RISK=['mix_weighted','mix_uniform','mix_weighted_raw','mix_uniform_raw','U5_weighted','global_margin','global_weighted','ridge6','source_only']
SCORE=['hard','soft','confidence'];PANEL=['U5_weighted','global_margin','global_weighted','mix_uniform','mix_weighted','ridge6']
CLASSES=['Crop','Forest','Grass','Shrub','Wetland','Water','Built','Bare']
def check(cls=None,src=None):
    m=json.loads((OUT/'manifest.json').read_text())
    for p,h in m['files'].items():
        if cls is not None and ('/artifacts/source/' in p or '/artifacts/unlabeled_cache/' in p):
            if not Path(p).name.startswith(f'{cls}_{src}.') and not Path(p).name.startswith(f'{cls}_{src}__'):continue
        assert sha(BR/p)==h,p
    return m
def grouping(m):
    result={}
    for di,d in enumerate(DOM):
        names=[r['name'] for r in m['pools'][d]];order=np.array(sorted(range(40),key=lambda i:names[i]));order=order[np.random.default_rng(250117+di).permutation(40)];result[d]=order.reshape(10,4).tolist()
    return result
def subsets(m,tgt):
    return [('batch4',4,k,np.array(ix)) for k,ix in enumerate(grouping(m)[tgt])]+[('pool',b,r,ix) for b,r,ix in u0.draws(tgt)]
def features(z,margin,pred,stats,proto,mode):return u5.error_features(z,margin,pred,stats,proto,'error_'+mode)
def bin_edges(x,pred):
    # Seven inner quantiles give eight bins. Repeated cuts are legal; ties use right bin.
    return np.array([[np.quantile(x[pred==s,j] if (pred==s).any() else x[:,j],np.arange(1,BINS)/BINS) for j in range(3)] for s in range(2)])
def bin_ids(x,pred,edges):
    ids=np.empty((len(x),3),np.uint8)
    for s in range(2):
        mask=pred==s
        for j in range(3):ids[mask,j]=np.searchsorted(edges[s,j],x[mask,j],side='right')
    return ids

def hist(ids,pred,error=None):
    a=np.zeros((2,2,3,BINS) if error is not None else (2,3,BINS),np.int64)
    for s in range(2):
        for j in range(3):
            if error is None:a[s,j]=np.bincount(ids[pred==s,j],minlength=BINS)
            else:
                for e in range(2):a[s,e,j]=np.bincount(ids[(pred==s)&(error==e),j],minlength=BINS)
    return a

def densities(h):
    # Jeffreys 0.5/bin smoothing, separately normalize every coordinate histogram.
    q=np.asarray(h,dtype=float)+.5;q/=q.sum(-1,keepdims=True);return q

def mixture(target_h,ref_h,lam):
    ref=densities(ref_h);mass=target_h[:,0].sum(-1);total=mass.sum();dens=densities(target_h)
    e=[];diag=[]
    for s in range(2):
        n=ref_h[s,:,0].sum(-1);prior=(n[1]+.5)/(n.sum()+1);delta=(ref[s,1]-ref[s,0]).ravel();y=(dens[s]-ref[s,0]).ravel();sep=float(delta@delta)/3
        numerator=float(delta@y)/3;denom=sep+lam
        raw=(numerator+lam*prior)/denom if sep>1e-10 else prior
        rate=float(np.clip(raw,0,1)) if mass[s]>0 else prior
        fitted=(1-rate)*ref[s,0]+rate*ref[s,1];residual=float(np.abs(dens[s]-fitted).sum()/6)
        e.append(rate);diag.append(dict(error_rate=rate,source_prior=float(prior),separation=sep,residual_TV=residual,boundary=bool(rate<1e-10 or rate>1-1e-10),empty_target_side=bool(mass[s]==0),nonidentifiable=bool(sep<=1e-10)))
    r=float(mass[1]/total) if total else 0.;counts=np.array([r*(1-e[1]),r*e[1],(1-r)*e[0]])
    return counts,diag

def calibrate(h,fit,cal):
    ref=h[fit].sum(0);ids=np.flatnonzero(cal);groups=[np.array([i]) for i in ids]+[ids[i:i+4] for i in range(0,len(ids),4)]
    records=[]
    for lam in LAMS:
        errors=[]
        for ix in groups:
            state=h[ix].sum(0);actual=np.array([state[1,0,0].sum(),state[1,1,0].sum(),state[0,1,0].sum()]);forecast,_=mixture(state.sum(1),ref,lam);a=f1(actual);b=f1(forecast)
            if np.isfinite(a) and np.isfinite(b):errors.append(abs(a-b))
        records.append(dict(lam=lam,calibration_MAE=float(np.mean(errors)) if errors else np.nan,groups=len(errors)))
    good=[r for r in records if np.isfinite(r['calibration_MAE'])];best=min(good,key=lambda r:(r['calibration_MAE'],-r['lam']))['lam'] if good else 1.
    residuals=[mixture(h[ix].sum((0,2)),ref,best)[1] for ix in groups]
    threshold=[float(np.quantile([d[s]['residual_TV'] for d in residuals],.95)) for s in range(2)]
    return dict(lam=best,grid=records,source_cal_residual95=threshold)

def bootstrap(h,ref_images,lam,seed):
    rng=np.random.default_rng(seed);rates=[];fs=[]
    for _ in range(BOOT):
        a=h[rng.integers(len(h),size=len(h))].sum(0);b=ref_images[rng.integers(len(ref_images),size=len(ref_images))].sum(0);count,d=mixture(a,b,lam);rates.append([q['error_rate'] for q in d]);fs.append(f1(count))
    rates=np.array(rates);good=np.array(fs);good=good[np.isfinite(good)]
    return dict(error_lo=np.quantile(rates,.025,axis=0).tolist(),error_hi=np.quantile(rates,.975,axis=0).tolist(),F1_lo=float(np.quantile(good,.025)) if len(good) else np.nan,F1_hi=float(np.quantile(good,.975)) if len(good) else np.nan)

def scenario(src,tgt):return ('region_S2' if 'sentinel2' in src else 'region_L8') if src.split('_')[0]!=tgt.split('_')[0] else ('sensor_GBA' if src.startswith('dwq') else 'sensor_XJ')
def check_seal():
    for p,h in json.loads((OUT/'seal.json').read_text())['files'].items():assert sha(BASE/p)==h,p
