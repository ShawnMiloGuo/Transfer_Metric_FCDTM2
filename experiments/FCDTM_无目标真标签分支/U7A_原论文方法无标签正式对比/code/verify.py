"""Pre-truth replay using a separate Gaussian FD expression and direct sums."""
import json
import numpy as np
from scipy.linalg import sqrtm
import common as c

def main():
 m=c.check()
 for p,h in json.loads((c.OUT/'seal.json').read_text()).items():assert c.sha(c.BASE/p)==h,p
 rows=c.read(c.OUT/'predictions.csv');maxd=0.;count=0
 for t in m['tasks']:
  cl,src,tgt=t['cls'],t['src'],t['tgt'];p=json.loads((c.OUT/f'source/{cl}_{src}.json').read_text());mu=np.array(p['mu']);w=np.array(p['weight'])
  with np.load(c.OUT/f'cache/{cl}_{src}__{tgt}.npz') as z,np.load(c.OUT/f'source/{cl}_{src}.npz') as s:
   ss=c.aggregate(s,np.arange(64));ms,cs=c.gaussian(ss)
   for r in rows:
    if (int(r['cls']),r['src'],r['tgt'])!=(cl,src,tgt):continue
    tt=c.aggregate(z,np.array(json.loads(r['indices'])));mt,ct=c.gaussian(tt);fd=float((ms-mt)@(ms-mt)+np.trace(cs)+np.trace(ct)-2*np.trace(sqrtm(cs@ct)).real);expected=max(fd,0.);observed=float(r['FD_global']);assert abs(expected-observed)<1e-5*max(1,abs(observed)),(expected,observed);maxd=max(maxd,abs(expected-observed))
    sums=c.scores(ss,tt,mu,w)
    for k,v in sums.items():assert np.isclose(v,float(r[k]),equal_nan=True,rtol=1e-10,atol=1e-8)
    count+=len(c.METHODS)
 assert count==26400
 result=dict(status='passed',records=len(rows),score_replays=count,FD_independent_scipy_sqrtm_max_difference=maxd,target_labels_read=False)
 (c.OUT/'预测复核.json').write_text(json.dumps(result,indent=1));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
