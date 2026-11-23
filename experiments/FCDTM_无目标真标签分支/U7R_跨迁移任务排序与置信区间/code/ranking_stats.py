"""Exact tau-b and average-tie Spearman under integer resampling multiplicities.
Not scipy.weightedtau: integer weights represent literal repeated observations.
"""
import numpy as np

def coefficients(x,y,weights=None):
 x=np.asarray(x,float);y=np.asarray(y,float);n=len(x)
 w=np.ones((1,n)) if weights is None else np.atleast_2d(weights).astype(float)
 assert x.shape==y.shape and w.shape[1]==n and np.all(w>=0)
 if not n:return np.full(len(w),np.nan),np.full(len(w),np.nan)
 sx=np.sign(x[:,None]-x[None,:]);sy=np.sign(y[:,None]-y[None,:])
 num=((w@(sx*sy))*w).sum(1);tx=((w@(sx!=0))*w).sum(1);ty=((w@(sy!=0))*w).sum(1)
 den=np.sqrt(tx*ty);tau=np.divide(num,den,out=np.full(len(w),np.nan),where=den>0)
 # Weighted midranks equal ranks in the literally duplicated bootstrap sample.
 ax=(x[None,:]<x[:,None]).astype(float)+.5*(x[None,:]==x[:,None]);ay=(y[None,:]<y[:,None]).astype(float)+.5*(y[None,:]==y[:,None]);rx=w@ax.T+.5;ry=w@ay.T+.5;total=w.sum(1)
 mx=np.divide((w*rx).sum(1),total,out=np.zeros(len(w)),where=total>0);my=np.divide((w*ry).sum(1),total,out=np.zeros(len(w)),where=total>0);dx=rx-mx[:,None];dy=ry-my[:,None];cov=(w*dx*dy).sum(1);den=np.sqrt((w*dx*dx).sum(1)*(w*dy*dy).sum(1));rho=np.divide(cov,den,out=np.full(len(w),np.nan),where=den>0)
 enough=(w>0).sum(1)>=3;tau[~enough]=np.nan;rho[~enough]=np.nan
 return np.clip(tau,-1,1),np.clip(rho,-1,1)

def block_weights(labels,reps,seed):
 groups=sorted(set(labels));lookup={g:i for i,g in enumerate(groups)};indices=np.array([lookup[g] for g in labels]);rng=np.random.default_rng(seed);counts=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=reps)
 return counts[:,indices],groups

def interval(values,usable_blocks,reps):
 v=np.asarray(values);v=v[np.isfinite(v)];valid=len(v)
 if usable_blocks<3:return np.nan,np.nan,valid,'fewer_than_3_blocks'
 if valid<max(1000,int(.8*reps)):return np.nan,np.nan,valid,'insufficient_finite_replicates'
 lo,hi=np.quantile(v,[.025,.975]);return float(lo),float(hi),valid,'defined_descriptive_interval'
