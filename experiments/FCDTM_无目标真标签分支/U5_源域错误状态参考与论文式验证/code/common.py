import hashlib,importlib.util,json,csv,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1];BR=BASE.parent;OUT=BASE/'artifacts';SRC=BR/'U0_标签隔离与无标签缓存'
spec=importlib.util.spec_from_file_location('u0common',SRC/'code/common.py');u0=importlib.util.module_from_spec(spec);spec.loader.exec_module(u0)
DATA=u0.DATA;REPO=u0.REPO;DOM=u0.DOM;sha=u0.sha;save=u0.save;read=u0.read;f1=u0.f1;draws=u0.draws;grid=u0.grid
RISK=['error_weighted','error_uniform','conditional_margin','global_margin','global_weighted','global_uniform','ridge6','source_only']
SCORE=['hard','soft','confidence'];PANEL=['hard','global_margin','global_weighted','error_uniform','error_weighted','ridge6']
def check(cls=None,source=None):
    m=json.loads((OUT/'manifest.json').read_text())
    for p,h in m['files'].items():
        if cls is not None:
            if '/artifacts/source/' in p and not p.endswith(f'/{cls}_{source}.json'):continue
            if '/artifacts/unlabeled_cache/' in p and not Path(p).name.startswith(f'{cls}_{source}__'):continue
        assert sha(BR/p)==h,p
    return m
def error_features(z,margin,pred,stats,proto,mode):
    if mode=='conditional_margin':return (margin/stats['margin_scale'])[:,None]
    a=stats['a'] if mode=='error_weighted' else np.ones(z.shape[1])/z.shape[1]
    # proto[prediction, error state] with error=0 correct, 1 wrong.
    side=np.asarray(pred,dtype=int)
    d0=(abs((z-proto[side,0])/stats['sigma'])*a).sum(1);d1=(abs((z-proto[side,1])/stats['sigma'])*a).sum(1)
    return np.column_stack([margin/stats['margin_scale'],d0,d1])
def prototype(z,p,y):
    result=np.zeros((2,2,z.shape[1]));counts=np.zeros((2,2),int);parent=[]
    for side in [0,1]:
        sp=p==side;mu=z[sp].mean(0) if sp.any() else z.mean(0);parent.append(mu)
        for error in [0,1]:
            keep=sp&((p!=y)==bool(error));counts[side,error]=keep.sum();result[side,error]=(z[keep].sum(0,dtype='float64')+512*mu)/(keep.sum()+512)
    return result,counts
def predict_error(x,model):return np.full(len(x),model['constant']) if 'constant' in model else u0.predictq(x,model)
def check_seal():
    for p,h in json.loads((OUT/'seal.json').read_text()).items():assert sha(BASE/p)==h,p
