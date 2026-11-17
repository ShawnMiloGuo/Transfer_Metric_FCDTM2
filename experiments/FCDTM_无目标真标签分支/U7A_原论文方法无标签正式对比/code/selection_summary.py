"""Final prespecified equal-task/equal-budget model-selection aggregation."""
from pathlib import Path
import csv,math,statistics,collections
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'artifacts'
METHODS=['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL']
rows=list(csv.DictReader((OUT/'模型选择.csv').open()));summary=[]
for cohort in ['P37','D42','E50']:
 groups=collections.defaultdict(dict)
 for r in rows:
  if r['cohort']==cohort and int(r['budget'])<40:groups[r['cls'],r['tgt'],r['budget'],r['repeat']][r['method']]=r
 eligible=[v for v in groups.values() if all(k in v and math.isfinite(float(v[k]['regret'])) for k in METHODS)]
 for method in METHODS:
  result=dict(cohort=cohort,method=method,common_configs=len(eligible),total_configs=len(groups))
  for column in ['regret','Top1']:
   budgets=collections.defaultdict(list)
   for g in eligible:
    r=g[method];budgets[r['cls'],r['tgt'],r['budget']].append(float(r[column]))
   tasks=collections.defaultdict(list)
   for key,v in budgets.items():tasks[key[:2]].append(statistics.mean(v))
   result[column]=statistics.mean(statistics.mean(v) for v in tasks.values()) if tasks else float('nan');result['target_class_groups']=len(tasks)
   result[column+'_config_weighted']=statistics.mean(float(g[method][column]) for g in eligible) if eligible else float('nan')
  result['max_loss']=max((float(g[method]['regret']) for g in eligible),default=float('nan'));summary.append(result)
with (OUT/'模型选择宏汇总.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
p=BASE/'实验结果分析.md';s=p.read_text();s=s.replace('软FCDTM选中最优模型的比例约69.2%，FD约91.0%，平均选择遗憾分别为0.0419和0.0050。','按计划对类别—目标域及预算等权汇总，软FCDTM选中最优模型的比例约68.1%，FD约91.1%，平均选择遗憾分别为0.0419和0.0050。')
s=s.split('\n\n## 按计划口径汇总的模型选择主结果')[0]
s+='\n\n## 按计划口径汇总的模型选择主结果\n\n上面的共同配置表将266个配置等权，作为透明的原始汇总保留。正式选择主结果按计划先对同类别、同目标、同预算的重复抽样平均，再对预算和类别—目标组等权，因此Top-1会略有不同；不是重新运行模型或改变选中的候选。两种汇总均支持FD本轮选模优于FCDTM。完整三任务清单及两种权重见模型选择宏汇总.csv。\n\n|方法|共同配置/全部|类别—目标组|平均选择遗憾↓|Top-1↑|最大损失↓|\n|---|---:|---:|---:|---:|---:|\n'
for r in summary:
 if r['cohort']=='D42':s+=f"|{r['method']}|{r['common_configs']}/{r['total_configs']}|{r['target_class_groups']}|{r['regret']:.4f}|{r['Top1']:.4f}|{r['max_loss']:.4f}|\n"
p.write_text(s)
print('Wrote equal-task/equal-budget selection summary; predictions and choices unchanged')
