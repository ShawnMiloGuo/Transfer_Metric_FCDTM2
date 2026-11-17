"""Additional post-seal statistics from evaluation CSV only, no model retuning."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import kendalltau,pearsonr
import common as c

def avg(v):
 a=[float(x) for x in v if np.isfinite(float(x))];return float(np.mean(a)) if a else np.nan

def main():
 m=c.check();rows=c.read(c.OUT/'逐项评价.csv');corr=c.read(c.OUT/'逐任务相关.csv');out=[];rng=np.random.default_rng(270217)
 for cohort,keys in m['cohorts'].items():
  keys={tuple(x) for x in keys}
  for method in c.PANEL:
   by=defaultdict(list);pooled=[];cluster=defaultdict(list)
   for r in rows:
    if r['kind']!='batch4' or (int(r['cls']),r['src'],r['tgt']) not in keys:continue
    x=c.oriented(r,method);y=float(r['true_AD'])
    if np.isfinite(x) and np.isfinite(y):
     by[int(r['cls']),r['src'],r['tgt']].append((x,y))
     if int(r['seed'])==c.SEEDS[0]:pooled.append((x,y))
   points=np.array([(avg(x for x,y in v),avg(y for x,y in v)) for v in by.values()]);tau=float(kendalltau(points[:,0],points[:,1]).statistic) if len(points)>2 else np.nan
   for r in corr:
    if r['scope']=='common' and r['method']==method and (int(r['cls']),r['src'],r['tgt']) in keys and np.isfinite(float(r['Pearson'])):cluster[int(r['cls']),r['src']].append(float(r['Pearson']))
   values=[avg(np.arctanh(np.clip(x,-1+1e-7,1-1e-7)) for x in v) for v in cluster.values()];a=np.array(values);boots=np.tanh(np.mean(a[rng.integers(len(a),size=(1000,len(a)))],axis=1)) if len(a) else np.array([np.nan]);pool=np.array(pooled);pr=float(pearsonr(pool[:,0]/max(np.max(np.abs(pool[:,0])),np.finfo(float).tiny),pool[:,1]).statistic) if len(pool)>3 and np.ptp(pool[:,0])>0 else np.nan
   out.append(dict(cohort=cohort,method=method,task_Kendall_tau_b=tau,task_points=len(points),pooled_r_first_seed=pr,model_clusters=len(a),cluster_Fisher_r=float(np.tanh(a.mean())) if len(a) else np.nan,bootstrap95_lo=float(np.nanquantile(boots,.025)),bootstrap95_hi=float(np.nanquantile(boots,.975))))
 c.save(c.OUT/'补充排序与簇区间.csv',out);p=c.BASE/'实验结果分析.md';s=p.read_text();s+='\n\n## 补充 跨任务排序与不确定性\n\n下表给出D42跨任务Kendall tau-b和按源模型簇重采样的描述性区间。合并相关容易受类别尺度影响，只作辅助。簇数有限，区间不证明独立新区域泛化；未做Holm配对显著性结论。\n\n|方法|任务Kendall tau-b|模型簇Fisher r|95%重采样范围|\n|---|---:|---:|---|\n'
 for r in out:
  if r['cohort']=='D42':s+=f"|{r['method']}|{r['task_Kendall_tau_b']:.4f}|{r['cluster_Fisher_r']:.4f}|[{r['bootstrap95_lo']:.4f}, {r['bootstrap95_hi']:.4f}]|\n"
 s=s.replace('未完成额外源模型簇区间、任务Kendall及合并相关时，不声称完整解决统计意见。','补充源模型簇区间、任务Kendall及合并相关见末节及CSV；配对显著性和独立确认仍未完成，不声称完整解决统计意见。');p.write_text(s)
if __name__=='__main__':main()
