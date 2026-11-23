"""Independent pair enumeration and literal-replication checks, optional SciPy oracle."""
import json,math,sys
import numpy as np
from common import B,O
from ranking_stats import coefficients,block_weights

def ref(x,y):
 if len(x)<3:return np.nan,np.nan
 p=q=tx=ty=0
 for i in range(len(x)):
  for j in range(i):
   a=np.sign(x[i]-x[j]);b=np.sign(y[i]-y[j]);p+=a*b>0;q+=a*b<0;tx+=a!=0;ty+=b!=0
 tau=(p-q)/math.sqrt(tx*ty) if tx*ty else np.nan
 def rank(v):return np.array([sum(u<t for u in v)+(sum(u==t for u in v)+1)/2 for t in v])
 a,b=rank(x),rank(y);a-=a.mean();b-=b.mean();den=math.sqrt(float(a@a)*float(b@b));rho=float(a@b)/den if den else np.nan
 return tau,rho
rng=np.random.default_rng(20261123);errors=[];oracle=[]
try:from scipy.stats import kendalltau,spearmanr;scipy=True
except ImportError:scipy=False
if '--require-scipy' in sys.argv:assert scipy,'SciPy oracle required'
for case in range(40):
 x=rng.integers(0,5,12).astype(float);y=rng.integers(0,6,12).astype(float)
 if case==0:x[:]=0
 if case==1:y=x.copy()
 if case==2:y=-x
 w=rng.multinomial(15,np.ones(12)/12,size=10);a,b=coefficients(x,y,w)
 for j,ww in enumerate(w):
  xx=np.repeat(x,ww);yy=np.repeat(y,ww);v=ref(xx,yy)
  for got,want in zip([a[j],b[j]],v):
   assert (np.isnan(got) and np.isnan(want)) or abs(got-want)<1e-12;errors.append(abs(got-want) if np.isfinite(want) else 0)
  if scipy:
   if np.ptp(xx)==0 or np.ptp(yy)==0:continue
   vv=kendalltau(xx,yy,variant='b').statistic,spearmanr(xx,yy).statistic
   for got,want in zip([a[j],b[j]],vv):assert abs(got-want)<1e-12;oracle.append(abs(got-want))
w,_=block_weights(['a','a','b','c'],5000,1);assert np.array_equal(w[:,0],w[:,1]);assert np.array_equal(w,block_weights(['a','a','b','c'],5000,1)[0])
result=dict(literal_repeat_cases=400,max_pair_reference_error=max(errors),scipy_oracle=scipy,max_scipy_error=max(oracle) if oracle else None,cluster_members_stay_together=True,seed_reproducible=True,constant_remains_NA=True)
(O/'unit_stats.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
