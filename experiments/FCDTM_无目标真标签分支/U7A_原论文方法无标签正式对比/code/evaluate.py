"""Separate post-seal evaluator; only this script reads target truth."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
import common as c

def corr(rs,k):
 q=[r for r in rs if np.isfinite(c.oriented(r,k)) and np.isfinite(r['true_AD'])];x=np.array([c.oriented(r,k) for r in q]);y=np.array([r['true_AD'] for r in q]);ok=len(q)>3 and np.ptp(x)>0 and np.ptp(y)>0
 if ok:x=(x-x.min())/np.ptp(x);y=(y-y.min())/np.ptp(y)
 return len(q),float(pearsonr(x,y).statistic) if ok else np.nan,float(spearmanr(x,y).statistic) if ok else np.nan

def avg(v):
 a=[float(x) for x in v if np.isfinite(float(x))];return float(np.mean(a)) if a else np.nan

def main():
 import subprocess,sys
 subprocess.check_call([sys.executable,str(c.BASE/'code/verify.py')])
 from osgeo import gdal
 m=c.check()
 for p,h in json.loads((c.OUT/'seal.json').read_text()).items():assert c.sha(c.BASE/p)==h,p
 truth={};rows=[];tasks=defaultdict(list);replay=[]
 ys={d:np.array([gdal.Open(str(c.DATA/d/'train_val/annotations'/r['split']/r['name'])).ReadAsArray() for r in pool]) for d,pool in m['pools'].items()}
 for t in m['tasks']:
  cls,src,tgt=t['cls'],t['src'],t['tgt'];p=json.loads((c.OUT/f'source/{cls}_{src}.json').read_text())
  with np.load(c.OUT/f'cache/{cls}_{src}__{tgt}.npz') as z,np.load(c.OUT/f'source/{cls}_{src}.npz') as s:
   pred=np.unpackbits(z['bits'],axis=1,count=65536).astype(bool);valid=np.unpackbits(z['valid_bits'],axis=1,count=65536).astype(bool);fg=ys[tgt].reshape(40,-1)==cls;truth[cls,src,tgt]=np.array([[(a&b&v).sum(),(a&~b&v).sum(),(~a&b&v).sum()] for a,b,v in zip(pred,fg,valid)])
 for raw in c.read(c.OUT/'predictions.csv'):
  r=dict(raw)
  for k in c.METHODS+['source_F1']:r[k]=float(r[k])
  for k in ['cls','budget','seed','repeat']:r[k]=int(r[k])
  ix=np.array(json.loads(r['indices']));key=r['cls'],r['src'],r['tgt'];r['true_F1']=c.f1(truth[key][ix]);r['true_AD']=r['source_F1']-r['true_F1'];r['common']=bool(np.isfinite(r['true_AD']) and all(np.isfinite(r[k]) for k in c.PANEL));r['scenario']=('region_S2' if 'sentinel2' in r['src'] else 'region_L8') if r['src'].split('_')[0]!=r['tgt'].split('_')[0] else ('sensor_GBA' if r['src'].startswith('dwq') else 'sensor_XJ');rows.append(r)
  if r['kind']=='batch4':tasks[key+(r['seed'],)].append(r)
 c.save(c.OUT/'逐项评价.csv',rows);cr=[]
 for key,rs in tasks.items():
  for scope,data in [('own',rs),('common',[r for r in rs if r['common']])]:
   for k in c.METHODS:
    n,pr,sr=corr(data,k);cr.append(dict(cls=key[0],src=key[1],tgt=key[2],seed=key[3],scenario=rs[0]['scenario'],scope=scope,method=k,K=n,Pearson=pr,Spearman=sr))
 c.save(c.OUT/'逐任务相关.csv',cr);summ=[]
 primary=c.PANEL[:5];sets=[{(r['cls'],r['src'],r['tgt'],r['seed']) for r in cr if r['scope']=='common' and r['method']==k and np.isfinite(r['Pearson']) and np.isfinite(r['Spearman'])} for k in primary];joint=set.intersection(*sets)
 c.save(c.OUT/'各自可定义覆盖.csv',[dict(method=k,task_seeds=len(q)) for k,q in zip(primary,sets)])
 for cohort,keys in m['cohorts'].items():
  keys={tuple(k) for k in keys}
  for scenario in ['all','sensor_GBA','sensor_XJ','region_S2','region_L8']:
   for k in c.METHODS:
    a=[r for r in cr if (r['cls'],r['src'],r['tgt']) in keys and r['scope']=='common' and r['method']==k and (r['cls'],r['src'],r['tgt'],r['seed']) in joint and (scenario=='all' or r['scenario']==scenario)];by=defaultdict(list)
    for r in a:by[r['seed']].append(r)
    fish=[float(np.tanh(avg(np.arctanh(np.clip(r['Pearson'],-1+1e-7,1-1e-7)) for r in v))) for v in by.values()];rho=[avg(r['Spearman'] for r in v) for v in by.values()];summ.append(dict(cohort=cohort,scenario=scenario,method=k,Fisher_r=avg(fish),Fisher_r_seed_sd=float(np.nanstd(fish)),rho=avg(rho),rho_seed_sd=float(np.nanstd(rho)),finite_correlations=sum(np.isfinite(r['Pearson']) for r in a),common_points=sum(r['K'] for r in a)))
 c.save(c.OUT/'相关汇总.csv',summ);choices=[]
 for cohort,keys in m['cohorts'].items():
  keys={tuple(k) for k in keys};by=defaultdict(list)
  for r in rows:
   if r['kind']=='pool' and (r['cls'],r['src'],r['tgt']) in keys:by[r['cls'],r['tgt'],r['budget'],r['repeat']].append(r)
  for key,rs in by.items():
   if len(rs)<2:continue
   y=np.array([r['true_F1'] for r in rs])
   for k in c.PANEL:
    score=np.array([c.oriented(r,k) for r in rs]);ok=np.isfinite(y).all() and not np.isnan(score).any()
    if ok:
     best=score.min()
     if np.isinf(best):win=score==best
     else:
      finite=np.isfinite(score);scale=max(float(np.abs(score[finite]).max()),np.finfo(float).tiny);win=finite&(np.abs(score-best)<=1e-12*scale)
     regret=float(y.max()-y[win].mean());top=float(np.isclose(y,y.max(),atol=1e-12,rtol=0)[win].mean())
    else:regret=top=np.nan
    choices.append(dict(cohort=cohort,cls=key[0],tgt=key[1],budget=key[2],repeat=key[3],method=k,regret=regret,Top1=top))
 c.save(c.OUT/'模型选择.csv',choices)
 # Reconstruct every saved score exclusively from the sealed sufficient statistics.
 for t in m['tasks']:
  cl,src,tgt=t['cls'],t['src'],t['tgt'];p=json.loads((c.OUT/f'source/{cl}_{src}.json').read_text())
  with np.load(c.OUT/f'cache/{cl}_{src}__{tgt}.npz') as z,np.load(c.OUT/f'source/{cl}_{src}.npz') as s:
   for r in rows:
    if (r['cls'],r['src'],r['tgt'])!=(cl,src,tgt):continue
    values=c.scores(c.aggregate(s,np.arange(64)),c.aggregate(z,np.array(json.loads(r['indices']))),np.array(p['mu']),np.array(p['weight']))
    for k,x in values.items():assert np.isclose(x,r[k],equal_nan=True,atol=1e-8,rtol=1e-10),(k,x,r[k])
 (c.OUT/'复算核验.json').write_text(json.dumps(dict(status='passed',records=len(rows),scores_replayed=len(rows)*len(c.METHODS),target_truth_only_postseal=True),indent=1))
 import plots
 plots.draw(tasks,summ)
 lines=['# U7A实验结果分析 通俗版','','## 先说明本轮验证什么','','本轮比较论文已有的FD、DS、GBC和FCDTM。目标真实标签只在预测全部封存后用于核对答案。FCDTM硬/软版和GBC明确使用伪标签，因此不能冒称原版有标签方法。模型冻结不微调。当前仍是历史开发池，不是独立网格确认。','','## 方法与读图','','FCDTM硬版把目标预测前景作为特征分组；软版按前景概率加权。FD比较全部源/目标采样特征分布，不用类标签。DS按预测分组，用平方距离再取log，修正旧代码漏平方。GBC比较预测两类特征的重叠，源自作者负重叠和定义。全部方法共用每图2048个均匀采样像素和同一层特征。','','横轴为方向统一后的原生分数，越大表示预期越差；纵轴是真实F1下降。它们不是同一单位，不画1:1。相关高说明同一模型在不同图组上的表现变化较一致，不等于具体F1估得准，也不自动代表更会筛选候选。','','## 主要结果 D42清单','','原论文表中37任务、正文声明42任务、现有扩展50任务分别报告，不能混用均值。这里用正文补齐42任务作为主清单。主要五方法使用共同可定义任务/种子交集，对角GBC是敏感性对照并单列覆盖。Pearson先做Fisher z再汇总，rho为任务均值；5个预设种子均报告，不挑最好seed。','','|场景|方法|Fisher r|种子标准差|rho|有效相关数|','|---|---|---:|---:|---:|---:|']
 for r in summ:
  if r['cohort']=='D42' and r['method'] in c.PANEL:lines.append(f"|{r['scenario']}|{r['method']}|{r['Fisher_r']:.4f}|{r['Fisher_r_seed_sd']:.4f}|{r['rho']:.4f}|{r['finite_correlations']}|")
 lines+=['','## 不能忽略的限制','','FCDTM源均值接近0的维度使用1e-6保护，另有不保护分数敏感性。空硬前景、GBC伪类样本不足、恒定分数分别记录，不以0替代失败；共同可算点和各方法自身覆盖见CSV。FD完整协方差、GBC球形与对角均已区分。r/rho不能当作AD预测误差，因此本轮不对原生分数计算MAE。原生分数选模只是启发式，最小AD不一定对应最大目标F1。目标F1越高越好，但不能用它调整分数方向。','','每个seed每任务最多10个四图点，关联样本少且影像/模型重复，不称独立大样本显著性。5种子标准差不是独立区域置信区间。未完成额外源模型簇区间、任务Kendall及合并相关时，不声称完整解决统计意见。U7B新增方法、源校准AD和新架构不属于本轮。','','## 论文式总览图']
 for s in ['sensor_GBA','sensor_XJ','region_S2','region_L8']:lines+=['',f'![{s}](artifacts/figures/paper_{s}.png)']
 lines+=['','所有50个逐任务图及PDF在artifacts/figures；每张使用第一预设种子，不挑图。不完整的后续统计仍列TODO，不用本轮对比代替独立确认。']
 (c.BASE/'实验结果分析.md').write_text('\n'.join(lines)+'\n');subprocess.check_call([sys.executable,str(c.BASE/'code/post_stats.py')]);print('EVALUATED AND PLOTTED',flush=True)
if __name__=='__main__':main()
