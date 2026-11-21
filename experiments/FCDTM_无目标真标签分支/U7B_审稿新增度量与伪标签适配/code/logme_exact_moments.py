"""ICML 2021 LogME using exact sufficient statistics, not a ridge proxy.
Equations and 11-step evidence iteration match the author's _fit_icml method.
X.T@X, X.T@y and y.T@y suffice; no target label is requested (y is self-PL).
"""
import numpy as np

def logme_icml(s):
 gram=np.asarray(s['outer'],dtype=np.float64);n=int(s['n']);d=len(gram)
 v,e,vh=np.linalg.svd(gram,full_matrices=True);scores=[]
 C=2 if s['cn'][1]>0 else 1  # Author classification rule: max(predicted label)+1.
 for c in range(C):
  xy=np.asarray(s['cs'][c],dtype=np.float64);yy=float(s['cn'][c]);tmp=vh@xy;alpha=beta=1.;lam=1.
  for _ in range(11):
   gamma=(e/(e+lam)).sum();w=v@(tmp*beta/(alpha+beta*e));ade=float(w@w)
   # Algebraic equality with ||y-Xw||^2, from the exact Gram and cross-product.
   bde=float(yy-2*w@xy+w@gram@w)
   alpha=gamma/(ade+1e-5);beta=(n-gamma)/(bde+1e-5);new=alpha/beta
   if abs(new-lam)/lam<.01:break
   lam=new
  with np.errstate(divide='ignore',invalid='ignore'):
   evidence=d/2*np.log(alpha)+n/2*np.log(beta)-.5*np.log(alpha+beta*e).sum()-beta/2*(bde+1e-5)-alpha/2*(ade+1e-5)-n/2*np.log(2*np.pi)
  scores.append(float(evidence/n))
 return float(np.mean(scores))

def check_author(reference):
 # Remove only the optional JIT decorator/import for this independent local check.
 code=reference.read_text().replace('from numba import njit','def njit(f): return f');namespace={};exec(compile(code,str(reference),'exec'),namespace)
 rng=np.random.default_rng(290121);results=[]
 for name,scale,singular in [('full_rank',1.,False),('small_scale',1e-4,False),('large_scale',1e4,False),('singular',1.,True),('one_group',1.,False)]:
  x=rng.normal(size=(200,8))*scale
  if singular:x[:,7]=x[:,0]
  y=(x[:,0]>0).astype(int) if name!='one_group' else np.zeros(200,dtype=int)
  s=dict(n=len(x),outer=x.T@x,cs=np.array([x[y==j].sum(0) for j in range(2)]),cn=np.bincount(y,minlength=2))
  author=namespace['LogME']()._fit_icml(x,y);own=logme_icml(s);diff=abs(author-own)
  assert np.isclose(author,own,rtol=1e-7,atol=1e-8),(name,author,own,diff)
  results.append(dict(case=name,author=float(author),exact_moments=float(own),absolute_difference=float(diff)))
 return results
if __name__=='__main__':
 from pathlib import Path
 print(check_author(Path(__file__).resolve().parents[1]/'references/LogME_author.py.txt'))
