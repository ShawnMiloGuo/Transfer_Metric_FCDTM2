import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1];BR=BASE.parent;OUT=BASE/'artifacts'
REPO=BR.parents[1];DATA=Path('/home/Shanxin.Guo/ZhangtuosCode/1_dataset/dataset')
DOM=['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8']
U1=BR/'U1_硬软伪标签FCDTM';U2=BR/'U2_权重引导双侧源风险'
RISK=['source_only','ridge6','ridge2','weighted','uniform','margin']
SCORE=['confidence','neg_entropy','hard','soft','matched_hard','matched_soft','uniform_hard','dual','raw_hard']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def save(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def f1(x):
    x=np.asarray(x);x=x.sum(0) if x.ndim>1 else x;t,p,n=x;d=2*t+p+n;return float(2*t/d) if d else np.nan
def grid(n):return int(n.split('_i')[-1].split('j')[0])
def draws(d,n=40):
    return [(b,r,np.random.default_rng(180116+100*DOM.index(d)+r).permutation(n)[:b]) for r in range(5) for b in [8,16,24]]+[(40,0,np.arange(n))]
def check():
    m=json.loads((OUT/'manifest.json').read_text())
    for p,h in m['files'].items():assert sha(BR/p)==h,p
    return m
def sigmoid(x):
    from scipy.special import expit
    return expit(x)
def features(margin,prob,scale):
    pred=margin>0
    return np.array([pred.mean(),np.maximum(prob,1-prob).mean(),margin[pred].mean()/scale if pred.any() else 0.,margin[~pred].mean()/scale if (~pred).any() else 0.,((margin>0)&(margin<=.25*scale)).mean(),((margin>=-.25*scale)&(margin<=0)).mean()])
def qfeatures(z,margin,stats,mode):
    if mode=='margin':return (margin/stats['margin_scale'])[:,None]
    a=stats['a'] if mode=='weighted' else np.ones(z.shape[1])/z.shape[1]
    distances=[(np.abs((z-mu)/stats['sigma'])*a).sum(1) for mu in stats['mu']]
    return np.column_stack([margin/stats['margin_scale'],*distances])
def fitq(x,y,xc,yc):
    from scipy.optimize import minimize,minimize_scalar
    center=x.mean(0);scale=np.maximum(x.std(0),1e-6);xx=(x-center)/scale;pi=(y.sum()+.5)/(len(y)+1)
    initial=np.r_[np.log(pi/(1-pi)),np.zeros(x.shape[1])]
    def fun(b):
        logit=b[0]+xx@b[1:];q=sigmoid(logit);loss=np.mean(np.logaddexp(0,logit)-y*logit)+.005*(b[1:]@b[1:]);g=np.r_[(q-y).mean(),xx.T@(q-y)/len(y)+.01*b[1:]];return loss,g
    if len(np.unique(y))<2:beta=initial;status='source_missing_class'
    else:
        fit=minimize(fun,initial,jac=True,method='L-BFGS-B',options={'maxiter':200});assert fit.success,fit.message;beta=fit.x;status='fit'
    offset=0.
    if len(np.unique(yc))==2:
        z=beta[0]+((xc-center)/scale)@beta[1:];res=minimize_scalar(lambda a:np.mean(np.logaddexp(0,z+a)-yc*(z+a)),bounds=(-10,10),method='bounded');offset=float(res.x)
    return dict(center=center,scale=scale,beta=beta,offset=offset,status=status)
def predictq(x,model):return sigmoid(model['beta'][0]+((x-model['center'])/model['scale'])@model['beta'][1:]+model['offset'])
def fitridge(x,c,width):
    x=x[:,:width];mu=x.mean(0);sd=np.maximum(x.std(0),1e-6);xx=(x-mu)/sd;y=c[:,[0,2]];inter=y.mean(0);coef=np.linalg.solve(xx.T@xx+10*np.eye(width),xx.T@(y-inter));return dict(mu=mu,sd=sd,coef=coef,inter=inter,width=width)
def ridge(x,p):
    q=p['inter']+((x[:p['width']]-p['mu'])/p['sd'])@p['coef'];area=x[0];tp=np.clip(q[0],0,area);fn=np.clip(q[1],0,1-area);return np.array([tp,area-tp,fn])
def score(target,source,w,protect=True):
    if not np.isfinite(target).all() or not np.isfinite(source).all():return np.nan
    den=np.where(abs(source)<1e-6,1e-6,source) if protect else source
    with np.errstate(divide='ignore',invalid='ignore'):s=float(((target-source)/den)@w)
    return s if np.isfinite(s) else np.nan
def serialize(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v))
