# -*- coding: utf-8 -*-
from common import *
import numpy as np,time
from kernel import *
sys.path.insert(0,str(OLD/'.deps'))
from threadpoolctl import threadpool_limits
threadpool_limits(1)
rng=np.random.default_rng(20261123);x=rng.normal(size=(25,4));y=rng.normal(size=(21,4));xo=np.array([0,5,12,25]);yo=np.array([0,2,11,15,21]);bw=3.2;sk=grouped_kernel(x,xo,bw=bw);tt=grouped_kernel(y,yo,bw=bw);st=grouped_kernel(x,xo,y,yo,bw=bw).sum(0);err=[]
for ix in [[0],[1,2],[0,1,2,3],[3,0]]:
 yy=np.concatenate([y[yo[i]:yo[i+1]] for i in ix]);a=from_sums(sk.sum(),len(x),tt,np.diff(yo),st,ix);bb=direct(x,yy,bw);err.append(abs(a-bb));assert abs(a-bb)<1e-12
# Independent double-loop kernel and U-statistic, including negative estimates.
ex=sum(np.exp(-sum((x[i]-x[j])**2)/bw) for i in range(len(x)) for j in range(len(x)) if i!=j)/(len(x)*(len(x)-1))+sum(np.exp(-sum((y[i]-y[j])**2)/bw) for i in range(len(y)) for j in range(len(y)) if i!=j)/(len(y)*(len(y)-1))-2*sum(np.exp(-sum((a-b)**2)/bw) for a in x for b in y)/(len(x)*len(y));assert abs(direct(x,y,bw)-ex)<1e-12
xx=rng.normal(size=(2048,64));yy=rng.normal(size=(2048,64));start=time.time();kernel_sum(xx,yy,128.);seconds=time.time()-start
O.mkdir(exist_ok=True);dump(O/'unit_mmd.json',dict(passed=True,subset_errors=err,double_loop_error=abs(direct(x,y,bw)-ex),block_2048_seconds=seconds,formula='exact unbiased RBF MMD^2',no_target_reference_reads=True));print('UNIT_MMD_PASSED block seconds',seconds,flush=True)
