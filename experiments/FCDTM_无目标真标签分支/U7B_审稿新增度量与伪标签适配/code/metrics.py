"""Frozen U7B definitions; target labels are predictions, never reference masks."""
import numpy as np
from logme_exact_moments import logme_icml
from score_moments import hscore
METHODS=['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL','AC','Entropy','ATC_MC','ATC_NE','DoC','LogME_PL','Hscore_PL','LEEP_PL','NCE_Pred_PL','NLEEP_PL','OTCE_PL_SrcCal','MMD_RBF_global']
BAD={'FCDTM_HardPL','FCDTM_SoftPL','FD_global','Entropy','DoC','OTCE_PL_SrcCal','MMD_RBF_global'}
def stats(z,p,y=None):
 z=np.asarray(z,dtype='float64');y=(p>.5) if y is None else np.asarray(y,dtype=bool);n=len(z)
 return dict(n=n,sum=z.sum(0),outer=z.T@z,cn=np.bincount(y.astype(int),minlength=2),cs=np.array([z[~y].sum(0),z[y].sum(0)]),cq=np.array([(z[~y]**2).sum(0),(z[y]**2).sum(0)]),soft_sum=(z*p[:,None]).sum(0),soft_mass=float(p.sum()))
def aggregate(a,ix):return {k:a[k][ix].sum(0) for k in ['n','sum','outer','cn','cs','cq','soft_sum','soft_mass']}
def mc(p):return np.maximum(p,1-p)
def negent(p):return p*np.log(np.maximum(p,1e-20))+(1-p)*np.log(np.maximum(1-p,1e-20))
def atc_threshold(scores,correct):
 errors=int((correct==0).sum())
 if errors==0:return float('-inf')
 if errors==len(correct):return float('inf')
 order=np.argsort(scores);x=scores[order];y=correct[order];fp=errors;fn=0;best=abs(fp-fn);threshold=0.
 for i in range(len(y)):
  if y[i]==0:fp-=1
  else:fn+=1
  if abs(fp-fn)<best:best=abs(fp-fn);threshold=float(x[i])
 return threshold

def leep(prob,y):
 prob=np.asarray(prob,dtype='float64');joint=np.array([prob[y==j].sum(0)/len(y) for j in range(2)]);mass=joint.sum(0)
 cond=np.divide(joint,mass[None,:],out=np.zeros_like(joint),where=mass[None,:]>0);emp=(prob@cond.T)[np.arange(len(y)),y]
 return float(np.log(emp).mean()) if np.all(emp>0) else np.nan

def nleep(z,y):
 from sklearn.decomposition import PCA
 from sklearn.mixture import GaussianMixture
 from sklearn.exceptions import ConvergenceWarning
 import warnings
 zz=PCA(n_components=.8,svd_solver='full').fit_transform(z)
 g=GaussianMixture(n_components=10,covariance_type='full',n_init=1,max_iter=100,tol=1e-3,reg_covar=1e-6,random_state=123).fit(zz)
 value=leep(g.predict_proba(zz).astype('float32'),y)
 return value,dict(converged=bool(g.converged_),iterations=int(g.n_iter_),pca_dimensions=zz.shape[1])

def d2(x,y):
 z=(x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T
 return np.maximum(z,0)
def transport(x,y,ys,yt):
 # Equal-size, uniform empirical measures: linear assignment is exact EMD.
 from scipy.optimize import linear_sum_assignment
 assert len(x)==len(y);cost=d2(x,y);i,j=linear_sum_assignment(cost);n=len(x);joint=np.zeros((2,2));np.add.at(joint,(ys[i],yt[j]),1/n);mass=joint.sum(1);ce=-sum(joint[a,b]*np.log(joint[a,b]/mass[a]) for a in range(2) for b in range(2) if joint[a,b]>0)
 return np.array([cost[i,j].mean(),ce]),dict(mass=float(joint.sum()),max_row_mass_error=0.,solver='uniform_equal_n_exact_assignment')
def mmd(x,y,bandwidth):
 kxx=np.exp(-d2(x,x)/bandwidth);kyy=np.exp(-d2(y,y)/bandwidth);kxy=np.exp(-d2(x,y)/bandwidth);n=len(x);m=len(y)
 return float((kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(m*(m-1))-2*kxy.mean())

def context(z,p,ys,weight,yp=None):
 yp=(p>.5) if yp is None else yp;ss=stats(z,p,yp);smean=z[ys==1].mean(0,dtype='float64') if (ys==1).any() else np.full(z.shape[1],np.nan);xx=d2(z[:256],z[:256]);positive=xx[np.triu_indices(len(xx),1)];positive=positive[positive>0];bandwidth=max(float(np.median(positive)) if len(positive) else 0.,1e-6)
 return dict(stats=ss,mu=smean,weight=weight,z=z,p=p,y=ys,AC=float(mc(p).mean()),ATC_MC=atc_threshold(mc(p),yp==ys),ATC_NE=atc_threshold(negent(p),yp==ys),bandwidth=bandwidth)

def score(z,p,ctx,old,full_stats=None,otcal=None,shared=False,yp=None):
 yp=(p>.5) if yp is None else yp;s=full_stats if full_stats is not None else stats(z,p,yp);r=old.scores(ctx['stats'],s,ctx['mu'],ctx['weight']);r={k:r[k] for k in METHODS[:5]};r.update({k:np.nan for k in METHODS[5:]});status={}
 r.update(AC=float(mc(p).mean()),Entropy=float(-negent(p).mean()),ATC_MC=float((mc(p)>=ctx['ATC_MC']).mean()),ATC_NE=float((negent(p)>=ctx['ATC_NE']).mean()),DoC=float(ctx['AC']-mc(p).mean()),Hscore_PL=hscore(s),LogME_PL=logme_icml(s),LEEP_PL=leep(np.column_stack([1-p,p]),yp.astype(int)),NCE_Pred_PL=0.)
 try:r['NLEEP_PL'],status['NLEEP_PL']=nleep(z,yp.astype(int))
 except Exception as e:status['NLEEP_PL']=dict(error=repr(e))
 if shared:
  r['MMD_RBF_global']=mmd(ctx['z'],z,ctx['bandwidth'])
  components,diag=transport(ctx['z'],z,ctx['y'],yp.astype(int));status['OTCE_components']=dict(W=float(components[0]),CE=float(components[1]),**diag)
  if otcal is not None:r['OTCE_PL_SrcCal']=float(otcal['intercept']+((components-otcal['mean'])/otcal['scale'])@otcal['coef'])
 return r,status
