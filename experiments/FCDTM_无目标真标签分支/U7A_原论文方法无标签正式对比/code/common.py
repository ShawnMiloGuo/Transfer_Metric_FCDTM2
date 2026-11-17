import importlib.util,json,hashlib,csv
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1];BR=BASE.parent;OUT=BASE/'artifacts';U0=BR/'U0_标签隔离与无标签缓存'
s=importlib.util.spec_from_file_location('u0',U0/'code/common.py');u0=importlib.util.module_from_spec(s);s.loader.exec_module(u0)
DATA=u0.DATA;REPO=u0.REPO;sha=u0.sha;save=u0.save;read=u0.read;f1=u0.f1;DOM=u0.DOM
METHODS=['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL','GBC_diagonal_PL','FCDTM_HardPL_unprotected','FCDTM_SoftPL_unprotected']
PANEL=METHODS[:6];SEEDS=list(range(290117,290122))
def check():
 m=json.loads((OUT/'manifest.json').read_text())
 for p,h in m['files'].items():assert sha(BASE/p)==h,p
 return m

def summary(z,pred,prob):
 # uniform sampled pixels, global full covariance sufficient statistics
 return dict(n=len(z),sum=z.sum(0,dtype='float64'),outer=z.astype('float64').T@z,cn=np.array([(~pred).sum(),pred.sum()]),cs=np.array([z[~pred].sum(0,dtype='float64'),z[pred].sum(0,dtype='float64')]),cq=np.array([(z[~pred].astype('float64')**2).sum(0),(z[pred].astype('float64')**2).sum(0)]),soft_sum=(z*prob[:,None]).sum(0,dtype='float64'),soft_mass=prob.sum(dtype='float64'))
def aggregate(z,ix):return {k:z[k][ix].sum(0) for k in ['n','sum','outer','cn','cs','cq','soft_sum','soft_mass']}
def gaussian(s):
 n=s['n'];mu=s['sum']/n;cov=(s['outer']-np.outer(s['sum'],s['sum'])/n)/max(n-1,1);return mu,(cov+cov.T)/2

def frechet(a,b):
 m,c=gaussian(a);n,d=gaussian(b);v,u=np.linalg.eigh(c);root=(u*np.sqrt(np.maximum(v,0)))@u.T;middle=root@d@root;tr=np.sqrt(np.maximum(np.linalg.eigvalsh((middle+middle.T)/2),0)).sum();x=float((m-n)@(m-n)+np.trace(c)+np.trace(d)-2*tr)
 if x<-1e-6*max(1,np.trace(c)+np.trace(d)):raise ArithmeticError('FD numerical PSD failure')
 return max(x,0.)
def gbc(s,spherical=True):
 if min(s['cn'])<2:return np.nan
 mu=s['cs']/s['cn'][:,None];var=np.maximum(s['cq']/s['cn'][:,None]-mu**2,1e-4)
 if spherical:var=np.broadcast_to(var.mean(1)[:,None],var.shape)
 avg=var.mean(0);distance=.125*((mu[0]-mu[1])**2/avg).sum()+.5*np.log(avg/np.sqrt(var[0]*var[1])).sum()
 return float(-2*np.exp(-max(distance,0.)))
def scores(src,t,mu,w):
 den=np.where(abs(mu)<1e-6,1e-6,mu)
 hard=t['cs'][1]/t['cn'][1] if t['cn'][1] else np.full_like(mu,np.nan);soft=t['soft_sum']/t['soft_mass'] if t['soft_mass'] else np.full_like(mu,np.nan)
 def f(v,d):
  with np.errstate(divide='ignore',invalid='ignore'):r=float(((v-mu)/d)@w)
  return r if np.isfinite(r) else np.nan
 center=t['sum']/t['n'];disp=sum(t['cn'][j]*np.sum((t['cs'][j]/t['cn'][j]-center)**2) for j in [0,1] if t['cn'][j]);ds=float(np.log(disp)) if disp>0 else -np.inf
 return dict(FCDTM_HardPL=f(hard,den),FCDTM_SoftPL=f(soft,den),FD_global=frechet(src,t),DS_pixel=ds,GBC_spherical_PL=gbc(t),GBC_diagonal_PL=gbc(t,False),FCDTM_HardPL_unprotected=f(hard,mu),FCDTM_SoftPL_unprotected=f(soft,mu))
def oriented(r,k):return -float(r[k]) if k in ['DS_pixel','GBC_spherical_PL','GBC_diagonal_PL'] else float(r[k])
def subsets(m,domain):
 names=[r['name'] for r in m['pools'][domain]];order=np.array(sorted(range(40),key=lambda i:names[i]));di=DOM.index(domain)
 for seed in SEEDS:
  groups=order[np.random.default_rng(seed+100*di).permutation(40)].reshape(10,4)
  for j,ix in enumerate(groups):yield 'batch4',4,seed,j,ix
 for b,r,ix in u0.draws(domain):yield 'pool',b,0,r,ix
