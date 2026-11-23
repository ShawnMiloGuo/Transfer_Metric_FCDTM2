# -*- coding: utf-8 -*-
"""Full-input quantized feasible coupling for OTCE, explicitly approximate.
Clustering is source-feature-only. Every input point contributes to sufficient
statistics; target batches never use other target images to form their cells.
"""
import numpy as np
from scipy.spatial.distance import cdist
Q=1024

def stats(z,labels,ids,k):
 z=np.asarray(z,dtype='float64');n=np.bincount(ids,minlength=k).astype('float64');s=np.zeros((k,z.shape[1]));np.add.at(s,ids,z);sq=np.bincount(ids,weights=(z*z).sum(1),minlength=k);lab=np.zeros((k,2));np.add.at(lab,(ids,np.asarray(labels,int)),1.)
 return dict(n=n,s=s,q=sq,labels=lab)

def aggregate(a,indices):return {key:sum(a[i][key] for i in indices) for key in ['n','s','q','labels']}

def transport(a,b):
 import ot
 ia=a['n']>0;ib=b['n']>0;na=a['n'][ia];nb=b['n'][ib];pa=na/na.sum();pb=nb/nb.sum();ma=a['s'][ia]/na[:,None];mb=b['s'][ib]/nb[:,None];va=np.maximum(a['q'][ia]/na-(ma*ma).sum(1),0);vb=np.maximum(b['q'][ib]/nb-(mb*mb).sum(1),0);cost=cdist(ma,mb,'sqeuclidean');plan,log=ot.emd(pa,pb,cost,numItermax=1000000,numThreads=1,log=True,check_marginals=True)
 if log.get('warning'):raise ArithmeticError('Quantized OT solver warning: '+str(log['warning']))
 margin=max(float(abs(plan.sum(1)-pa).max()),float(abs(plan.sum(0)-pb).max()));assert margin<1e-8 and plan.min()>-1e-12
 centroid_cost=float(np.sum(plan*cost));variances=float(pa@va+pb@vb);lifted_cost=centroid_cost+variances
 sa=a['labels'][ia]/na[:,None];tb=b['labels'][ib]/nb[:,None];joint=sa.T@plan@tb;mass=joint.sum(1);ce=-sum(joint[i,j]*np.log(joint[i,j]/mass[i]) for i in range(2) for j in range(2) if joint[i,j]>0);assert abs(joint.sum()-1)<1e-8
 # This is the exact cost and class joint of the feasible product-lifted plan,
 # not the exact full empirical OT optimum. Keep this distinction explicit.
 return np.array([lifted_cost,float(ce)]),dict(source_points=int(na.sum()),target_points=int(nb.sum()),source_cells=len(na),target_cells=len(nb),marginal_error=margin,centroid_cost=centroid_cost,within_cell_cost=variances,lifted_W=lifted_cost,CE=float(ce),source_quantization_rms=float(np.sqrt(pa@va)),target_quantization_rms=float(np.sqrt(pb@vb)),feasible_plan_cost_upper_bound=True,exact_full_OT=False)
