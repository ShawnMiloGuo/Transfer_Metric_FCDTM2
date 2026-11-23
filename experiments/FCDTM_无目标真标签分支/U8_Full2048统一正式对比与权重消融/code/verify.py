# -*- coding: utf-8 -*-
from common import *
import numpy as np
from scipy.stats import kendalltau,spearmanr,pearsonr
from collections import defaultdict
rows=read(O/'evaluated.csv');tasks=read(O/'task_data.csv');stats=read(O/'ranking.csv');cc=read(O/'batch_correlations.csv');m=json.loads((OLD/'artifacts/manifest_full.json').read_text())
for p,h in json.loads((O/'score_seal.json').read_text()).items():assert sha(B/p)==h,p
old={tuple(r[k] for k in ['cls','src','tgt','kind','budget','seed','repeat']):r for r in read(OLD/'artifacts/predictions_full.csv') if r['panel']=='full2048'};n=0
for r in rows:
 k=tuple(r[k] for k in ['cls','src','tgt','kind','budget','seed','repeat']);t=old[k]
 for method in METHODS:
  if method in ['MMD_RBF_global','OTCE_PL_SrcCal']:continue
  a,b=float(r[method]),float(t[method]);assert a==b or np.isnan(a) and np.isnan(b),(k,method);n+=1
err=0.;point=0
for r in stats:
 keys={tuple(k) for k in m['cohorts'][r['cohort']]};q=[t for t in tasks if (int(t['cls']),t['src'],t['tgt']) in keys and (r['scenario']=='all' or t['scenario']==r['scenario']) and (r['scope']=='own' or t['common']=='True') and np.isfinite(float(t[r['method']])) and np.isfinite(float(t['true_AD']))];x=np.array([float(t[r['method']])*(1 if r['method'] in BAD else -1) for t in q]);y=np.array([float(t['true_AD']) for t in q]);a=kendalltau(x,y).statistic;b=spearmanr(x,y).statistic
 for actual,want in [(a,float(r['tau_b'])),(b,float(r['Spearman']))]:
  assert abs(actual-want)<1e-10 or np.isnan(actual) and np.isnan(want);err=max(err,abs(actual-want)) if np.isfinite(actual) else err
 point+=1
# Independent literal-duplication check of actual U8 class-block bootstrap.
sys.path.insert(0,str(BR/'U7R_跨迁移任务排序与置信区间/code'));from ranking_stats import coefficients,block_weights
q=[t for t in tasks if (int(t['cls']),t['src'],t['tgt']) in {tuple(k) for k in m['cohorts']['D42']} and t['common']=='True'];w,_=block_weights([t['cls'] for t in q],5000,20261123);bn=0
for method in ['MMD_RBF_global','OTCE_PL_SrcCal','FCDTM_SoftSigned','FCDTM_SoftAbs','FCDTM_SoftPL']:
 x=np.array([float(t[method])*(1 if method in BAD else -1) for t in q]);y=np.array([float(t['true_AD']) for t in q]);tt,ss=coefficients(x,y,w)
 for i in [0,13,120,2100,4999]:
  ix=np.repeat(np.arange(len(q)),w[i].astype(int));a=kendalltau(x[ix],y[ix]).statistic;b=spearmanr(x[ix],y[ix]).statistic;assert abs(a-tt[i])<1e-10 or np.isnan(a) and np.isnan(tt[i]);assert abs(b-ss[i])<1e-10 or np.isnan(b) and np.isnan(ss[i]);bn+=1
# Every batch correlation and seed uses direct SciPy ranks/correlation.
g=defaultdict(list)
for t in rows:
 if t['kind']=='batch4':g[t['cls'],t['src'],t['tgt'],t['seed']].append(t)
for r in cc:
 q=[t for t in g[r['cls'],r['src'],r['tgt'],r['seed']] if (r['scope']=='own' or t['common']=='True') and np.isfinite(float(t[r['method']])) and np.isfinite(float(t['true_AD']))];x=np.array([float(t[r['method']])*(1 if r['method'] in BAD else -1) for t in q]);y=np.array([float(t['true_AD']) for t in q])
 if len(q)<4 or np.ptp(x)==0 or np.ptp(y)==0:a=b=np.nan
 else:a=pearsonr((x-x.min())/np.ptp(x),(y-y.min())/np.ptp(y)).statistic;b=spearmanr(x,y).statistic
 assert abs(a-float(r['Pearson']))<1e-10 or np.isnan(a) and np.isnan(float(r['Pearson']));assert abs(b-float(r['Spearman']))<1e-10 or np.isnan(b) and np.isnan(float(r['Spearman']))
dump(O/'independent_verification.json',dict(status='passed',preserved_baseline_values=n,ranking_points=point,batch_correlations=len(cc),literal_bootstrap_checks=bn,max_ranking_error=err,score_seal_verified=True))
print('VERIFIED',point,len(cc),bn)
