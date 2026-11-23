# -*- coding: utf-8 -*-
from common import *
import numpy as np
sys.path.insert(0,str(OLD/'.deps'));sys.path.insert(0,str(B/'.deps'))
from threadpoolctl import threadpool_limits
threadpool_limits(1)
from scipy.optimize import linear_sum_assignment,linprog
from scipy.spatial.distance import cdist
from ot_quantized import stats,transport
score_guard();rng=np.random.default_rng(91);x=rng.normal(size=(6,3));y=rng.normal(size=(6,3));ls=np.array([0,0,0,1,1,1]);lt=np.array([0,1,0,1,1,0]);cost=cdist(x,y,'sqeuclidean');i,j=linear_sum_assignment(cost);ex=float(cost[i,j].mean());joint=np.zeros((2,2));np.add.at(joint,(ls[i],lt[j]),1/6);mass=joint.sum(1);ce=-sum(joint[a,b]*np.log(joint[a,b]/mass[a]) for a in range(2) for b in range(2) if joint[a,b]>0);s,d=transport(stats(x,ls,np.arange(6),6),stats(y,lt,np.arange(6),6));assert abs(s[0]-ex)<1e-10 and abs(s[1]-ce)<1e-10
x=rng.normal(size=(4,3));y=rng.normal(size=(6,3));ls=np.array([0,0,1,1]);lt=np.array([0,1,0,1,0,1]);C=cdist(x,y,'sqeuclidean');eq=np.zeros((10,24))
for a in range(4):eq[a,a*6:(a+1)*6]=1
for b in range(6):eq[4+b,b::6]=1
lp=linprog(C.ravel(),A_eq=eq,b_eq=np.r_[np.ones(4)/4,np.ones(6)/6],bounds=(0,None),method='highs');assert lp.success
s,d=transport(stats(x,ls,np.arange(4),4),stats(y,lt,np.arange(6),6));assert abs(s[0]-lp.fun)<1e-10
# With nontrivial cells, lift cluster plan to every pair of points and verify
# its full cost, class joint and marginals by explicit point loops.
ia=np.array([0,0,1,1]);ib=np.array([0,0,0,1,1,1]);a=stats(x,ls,ia,2);b=stats(y,lt,ib,2);s,d=transport(a,b)
import ot
pa=a['n']/len(x);pb=b['n']/len(y);cc=cdist(a['s']/a['n'][:,None],b['s']/b['n'][:,None],'sqeuclidean');P=ot.emd(pa,pb,cc,numThreads=1);full=np.array([[P[ia[i],ib[j]]/(a['n'][ia[i]]*b['n'][ib[j]]) for j in range(6)] for i in range(4)]);assert np.max(abs(full.sum(1)-.25))<1e-12 and np.max(abs(full.sum(0)-1/6))<1e-12;assert abs(float((full*C).sum())-s[0])<1e-10
joint=np.zeros((2,2))
for i in range(4):
 for j in range(6):joint[ls[i],lt[j]]+=full[i,j]
mass=joint.sum(1);ce=-sum(joint[a,b]*np.log(joint[a,b]/mass[a]) for a in range(2) for b in range(2) if joint[a,b]>0);assert abs(s[1]-ce)<1e-10 and s[0]>=lp.fun-1e-10
O.mkdir(exist_ok=True);dump(O/'unit_ot.json',dict(passed=True,point_identity_matches_assignment=True,unequal_points_matches_linprog=True,product_lift_cost_and_class_joint_checked=True,product_lift_feasible=True,approximate_cost_not_below_exact=True,POT_version=ot.__version__,target_reference_reads=0));print('UNIT_OT_PASSED',flush=True)
