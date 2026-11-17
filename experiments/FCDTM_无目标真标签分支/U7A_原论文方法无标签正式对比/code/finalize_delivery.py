from pathlib import Path
import json,hashlib,subprocess,csv
u=Path('/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/experiments/FCDTM_无目标真标签分支/U7A_原论文方法无标签正式对比');b=u.parent
assert (u/'统计核对说明.md').exists()
assert len(list((u/'artifacts/figures').glob('*.png')))==54
for name in ['unit.json','预测复核.json','复算核验.json']:assert json.loads((u/'artifacts'/name).read_text())['status']=='passed'
assert len(list((u/'artifacts').glob('audit_*.json')))==25
for p in (u/'artifacts').glob('audit_*.json'):assert json.loads(p.read_text())['status']=='passed'
for rel,h in json.loads((u/'artifacts/seal.json').read_text()).items():assert hashlib.sha256((u/rel).read_bytes()).hexdigest()==h
(u/'最终评价代码SHA.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (u/'code').glob('*.py')},indent=1))
# Re-evaluate native ranking ties with finite relative scales and explicit infinite limits.
import sys
sys.path.insert(0,str(u/'code'))
import common as c
from collections import defaultdict
m=c.check();evaluated=c.read(u/'artifacts/逐项评价.csv');choices=[]
for cohort,keys in m['cohorts'].items():
 keys={tuple(k) for k in keys};groups=defaultdict(list)
 for r in evaluated:
  if r['kind']=='pool' and (int(r['cls']),r['src'],r['tgt']) in keys:groups[int(r['cls']),r['tgt'],int(r['budget']),int(r['repeat'])].append(r)
 for key,rs in groups.items():
  if len(rs)<2:continue
  y=__import__('numpy').array([float(r['true_F1']) for r in rs])
  for k in c.PANEL:
   np=__import__('numpy');v=np.array([c.oriented(r,k) for r in rs]);ok=np.isfinite(y).all() and not np.isnan(v).any()
   if ok:
    best=v.min()
    if np.isinf(best):win=v==best
    else:
     finite=np.isfinite(v);scale=max(float(np.abs(v[finite]).max()),np.finfo(float).tiny);win=finite&(np.abs(v-best)<=1e-12*scale)
    regret=float(y.max()-y[win].mean());top=float(np.isclose(y,y.max(),atol=1e-12,rtol=0)[win].mean())
   else:regret=top=float('nan')
   choices.append(dict(cohort=cohort,cls=key[0],tgt=key[1],budget=key[2],repeat=key[3],method=k,regret=regret,Top1=top))
c.save(u/'artifacts/模型选择.csv',choices)
rows=list(csv.DictReader((u/'artifacts/模型选择.csv').open()));lookup={}
for r in rows:
 if r['cohort']=='D42' and int(r['budget'])<40:lookup.setdefault((r['cls'],r['tgt'],r['budget'],r['repeat']),{})[r['method']]=r
import math,numpy as np
primary=['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL'];common=[v for v in lookup.values() if all(k in v and math.isfinite(float(v[k]['regret'])) for k in primary)]
p=u/'实验结果分析.md';s=p.read_text();s+='\n\n## 补充 候选模型是否选得更好\n\n以下是D42中8/16/24图无标签预算，在主要五种方法均可计算的相同配置上比较。原生分数选模只是启发式，不是校准的F1预测；选中最优候选的比例越高越好，遗憾越低越好。方法不支持的配置数量也必须一起看。\n\n|方法|共同配置数/全部多候选配置|平均遗憾|Top-1|最大损失|\n|---|---:|---:|---:|---:|\n'
for k in primary:
 a=[float(v[k]['regret']) for v in common];top=[float(v[k]['Top1']) for v in common];s+=f"|{k}|{len(common)}/{len(lookup)}|{np.mean(a):.4f}|{np.mean(top):.4f}|{max(a):.4f}|\n"
s+='\n共同配置的条件结果不能替代全部部署覆盖；所有方法自身覆盖和失败配置原样保存在模型选择.csv。图区r为第一固定seed原始相关平均，主表为5seed的Fisher汇总，二者口径不同。\n';p.write_text(s)
p=u/'TODO.md';s=p.read_text().replace('- [ ] 通俗分析已完成；待最终图形检查、本地同步及独立分支提交。','- [x] 通俗分析、最终图形检查、本地同步及独立分支提交。');p.write_text(s)
p=b/'TODO.md';s=p.read_text();start='- [ ] U7A进行中：FD/DS/GBC与FCDTM硬软PL独立实现及公式核验已完成，25模型GPU推理与访问审计已启动，尚未取得正式结果。';new='- [x] U7A：25模型/50任务，五种子四图+无标签池3300记录；FD/DS/GBC修正及硬软FCDTM-PL正式配对、25访问审计、26400分数复核、54对PNG/PDF和通俗分析完成。[结果](U7A_原论文方法无标签正式对比/实验结果分析.md)。仍为历史开发池；审稿新增方法、新架构、Holm深入检验属后续阶段。';s=s.replace(start,new).replace('- [ ] U7A：原论文FCDTM/FD/DS/GBC无目标真标签正式对比；先修复/核验实现和预测入口。',new);p.write_text(s)
p=b/'README.md';s=p.read_text();s+='\n- [U7A正式原论文方法无标签对比](U7A_原论文方法无标签正式对比/实验结果分析.md)：已完成原生分数比较与54对论文式图；不等于U7B审稿新增方法或独立确认已完成。\n';p.write_text(s)
p=b/'U7_原论文与审稿新增方法正式对比/TODO.md';s=p.read_text();
for prefix in ['接入/修正FD-global','建立目标image-only','冻结P37/D42/E50','独立评价、论文式']:s=s.replace('- [ ] '+prefix,'- [x] '+prefix)
s+='\n## U7A执行更新\n\n- [x] U7A新实现、公式检查、目标image-only、五种子/三任务层清单、25审计、封存、充分统计量及独立FD复核、正式配对评价与54对图表已完成，见独立U7A目录。\n- [ ] 旧论文历史运行commit/全部参数逐数追溯仍未完成，不把本次统一新实现冒称旧有标签结果的精确复现；作者TF全流水线逐位复现未执行，采用原公式及独立密度积分核验。\n- [ ] 审稿新增U7B、深入消融/Holm U7C及新架构/独立确认U7D仍待执行。\n';p.write_text(s)
for p in u.rglob('._*'):
 if p.is_file():p.unlink()
subprocess.check_call(['/home/Shanxin.Guo/anaconda3/envs/FCDTM/bin/python',str(b/'归档核验.py')])
subprocess.check_call(['tar','--exclude=__pycache__','-czf','/tmp/u7a_delivery.tar.gz','-C',str(b.parent),b.name])
print('Delivery archive ready')
