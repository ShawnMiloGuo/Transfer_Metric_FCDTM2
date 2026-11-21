from pathlib import Path
import sys,json,itertools,time
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'.deps'));import numpy as np;import metrics as m
from logme_exact_moments import check_author

def main():
 ref=B/'references';checks={};checks['LogME_author_ICML']=check_author(ref/'LogME_author.py.txt')
 rng=np.random.default_rng(71);z=rng.normal(size=(256,8));p=rng.uniform(.01,.99,256);y=p>.5
 h=m.hscore(m.stats(z,p,y));g=np.array([z[y==v].mean(0) for v in y]);expected=np.trace(np.linalg.pinv(np.cov(z,rowvar=False),rcond=1e-15)@np.cov(g,rowvar=False));assert abs(h-expected)<1e-10;checks['Hscore_direct']=float(abs(h-expected))
 code={};exec((ref/'LEEP_reference.py.txt').read_text(),code);assert abs(m.leep(np.c_[1-p,p],y.astype(int))-code['LEEP'](np.c_[1-p,p],y.astype(int)))<1e-12;checks['LEEP_reference']=True
 code={};exec((ref/'ATC_helper.py.txt').read_text(),code)
 for score in [m.mc(p),m.negent(p)]:
  correct=rng.integers(0,2,len(p));_,th=code['find_ATC_threshold'](score,correct);assert m.atc_threshold(score,correct)==th
 checks['ATC_author_threshold_both']=True
 assert m.atc_threshold(m.negent(p),np.ones(len(p),dtype=int))==float('-inf')
 assert m.atc_threshold(m.negent(p),np.zeros(len(p),dtype=int))==float('inf')
 checks['ATC_boundary_extension_all_correct_or_all_wrong']=True
 x=rng.normal(size=(6,3));zz=rng.normal(size=(6,3));ys=np.array([0,0,0,1,1,1]);yt=np.array([0,1,0,1,0,1]);v,diag=m.transport(x,zz,ys,yt);d=m.d2(x,zz);brute=min(sum(d[i,p[i]] for i in range(6))/6 for p in itertools.permutations(range(6)));assert abs(v[0]-brute)<1e-12;assert abs(diag['mass']-1)<1e-12;checks['OT_exact_against_all_720_permutations']=True
 # Known conditional entropy under the unique zero-distance identity coupling.
 xx=np.arange(4,dtype=float)[:,None];c,_=m.transport(xx,xx,np.array([0,0,1,1]),np.array([0,1,0,1]));assert abs(c[1]-np.log(2))<1e-12;checks['OTCE_conditional_entropy_log2']=True
 bw=2.;v=m.mmd(x,zz,bw);ex=(sum(np.exp(-sum((x[i]-x[j])**2)/bw) for i in range(6) for j in range(6) if i!=j)/30+sum(np.exp(-sum((zz[i]-zz[j])**2)/bw) for i in range(6) for j in range(6) if i!=j)/30-2*sum(np.exp(-sum((x[i]-zz[j])**2)/bw) for i in range(6) for j in range(6))/36);assert abs(v-ex)<1e-12;checks['MMD_unbiased_explicit_loops']=True
 from sklearn.decomposition import PCA
 from sklearn.mixture import GaussianMixture
 val,info=m.nleep(z,y.astype(int));pc=PCA(n_components=.8,svd_solver='full').fit_transform(z);gg=GaussianMixture(n_components=10,covariance_type='full',n_init=1,max_iter=100,tol=1e-3,reg_covar=1e-6,random_state=123).fit(pc);expected=m.leep(gg.predict_proba(pc).astype('float32'),y.astype(int));assert val==expected;checks['NLEEP_GMM_then_LEEP_reference']=dict(score=val,**info)
 start=time.time();value,info=m.nleep(rng.normal(size=(8192,32)),rng.integers(0,2,8192));checks['NLEEP_8192_timing']=dict(seconds=time.time()-start,**info)
 checks['target_reference_read']=False;(B/'artifacts/unit_full.json').write_text(json.dumps(checks,indent=2));print('UNIT PASSED',json.dumps(checks),flush=True)
if __name__=='__main__':main()
