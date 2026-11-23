# -*- coding: utf-8 -*-
"""Exact empirical RBF U-statistic using all sampled points, image-block sums."""
import numpy as np,time

def kernel_sum(x,y,bw):
 x=np.asarray(x,dtype='float64');y=np.asarray(y,dtype='float64');d=(x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T;np.maximum(d,0,out=d);d*=(-1/bw);np.exp(d,out=d);return float(d.sum(dtype='float64'))

def grouped_kernel(x,xoff,y=None,yoff=None,bw=1.,progress=None):
 symmetric=y is None
 if symmetric:y=x;yoff=xoff
 out=np.zeros((len(xoff)-1,len(yoff)-1),dtype='float64')
 for i in range(len(out)):
  for j in range(i if symmetric else 0,out.shape[1]):
   out[i,j]=kernel_sum(x[xoff[i]:xoff[i+1]],y[yoff[j]:yoff[j+1]],bw)
   if symmetric:out[j,i]=out[i,j]
  if progress and (i+1)%8==0:progress(i+1,len(out))
 return out

def from_sums(ss,n,tt,nt,st,indices):
 ids=np.asarray(indices,dtype=int);m=int(nt[ids].sum());kx=float(tt[np.ix_(ids,ids)].sum());cross=float(st[ids].sum())
 return float((ss-n)/(n*(n-1))+(kx-m)/(m*(m-1))-2*cross/(n*m))

def direct(x,y,bw):
 n=len(x);m=len(y);return (kernel_sum(x,x,bw)-n)/(n*(n-1))+(kernel_sum(y,y,bw)-m)/(m*(m-1))-2*kernel_sum(x,y,bw)/(n*m)
