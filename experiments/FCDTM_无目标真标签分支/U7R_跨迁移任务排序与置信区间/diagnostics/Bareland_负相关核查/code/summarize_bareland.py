# -*- coding: utf-8 -*-
from pathlib import Path
import csv,json,math,hashlib,re,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parents[1];A=D/'artifacts';F=D/'figures';F.mkdir(exist_ok=True)
read=lambda n:list(csv.DictReader((A/n).open()))
means=read('mean_correlations.csv');tasks=read('task_diagnostics.csv');models=json.loads((A/'model_checks.json').read_text());proof=json.loads((A/'verification.json').read_text());batch=[r for r in read('batch_diagnostics.csv') if r['kind']=='batch4']
fmt=lambda v:f'{float(v):.3f}'
def name(s):return s.replace('dwq','GBA').replace('xj','XJ').replace('landsat8','L8').replace('sentinel2','S2').replace('_','-')
def row(src,tgt,method):return next(r for r in means if r['src']==src and r['tgt']==tgt and r['method']==method)
# Check exact displayed rounded numbers, not only underlying CSVs.
parser=argparse.ArgumentParser();parser.add_argument('--paper',required=True);args=parser.parse_args();paper=Path(args.paper);checks=[]
for filename in ['tasks_region.tex','tasks_sensor.tex']:
 for line in (paper/'data'/filename).read_text().splitlines():
  if '& Bareland &' not in line:continue
  cells=[s.strip().replace('\\\\','').strip() for s in line.split('&')];src=next(r['src'] for r in tasks if name(r['src'])==cells[0]);tgt=next(r['tgt'] for r in tasks if name(r['tgt'])==cells[1]);actual=[]
  for method in ['FCDTM_HardPL','FCDTM_SoftPL','FD_global','DS_pixel','GBC_spherical_PL']:
   r=row(src,tgt,method);actual.extend([fmt(r['Pearson']),fmt(r['Spearman'])])
  assert actual==cells[3:],(filename,src,tgt,actual,cells[3:]);checks.append(dict(table=filename,src=src,tgt=tgt,all_ten_cells_match=True))
assert len(checks)==8
(A/'table_verification.json').write_text(json.dumps(dict(status='passed',Bareland_rows=8,table_cells_checked=80,rows=checks),ensure_ascii=False,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,4,figsize=(15,7));seedvals=sorted({r['seed'] for r in batch});cmap=plt.get_cmap('tab10')
for ax,t in zip(axs.flat,tasks):
 src,tgt=t['src'],t['tgt'];rr=[r for r in batch if r['src']==src and r['tgt']==tgt];x=np.array([float(r['FCDTM_SoftPL']) for r in rr]);y=np.array([float(r['AD']) for r in rr]);ok=np.isfinite(x)&np.isfinite(y)
 for i,s in enumerate(seedvals):
  ix=np.array([r['seed']==s for r in rr])&ok;ax.scatter(x[ix],y[ix],s=22,alpha=.65,color=cmap(i),label=str(i+1))
 r=row(src,tgt,'FCDTM_SoftPL');ax.set_title(name(src)+' → '+name(tgt),fontsize=10);ax.text(.98,.98,'Mean within-seed\nr='+fmt(r['Pearson'])+', rho='+fmt(r['Spearman'])+'\nnegative r seeds: '+r['negative_Pearson_seeds']+'/'+r['seeds'],transform=ax.transAxes,va='top',ha='right',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.8));ax.grid(alpha=.2);ax.set_xlabel('FCDTM-SoftPL (native AD direction)');ax.set_ylabel('Actual AD = source F1 - target F1')
fig.suptitle('Bareland audit: unchanged scores and all five original partitions',fontsize=13);fig.tight_layout(rect=(0,.02,1,.94));fig.text(.5,.005,'Points reuse the same 40 target images across five partitions; they are not independent spatial tests.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(F/('Bareland_soft_scatter.'+ext),dpi=200)
plt.close(fig)
lines=['# 表X/XI Bareland负相关核查','', '## 核查结论','', '独立复算未发现这8行的抄表、符号、类别编号、模型缓存对应或F1计算错误。负数是度量与AD的负相关，不是负的分类精度；不能通过取绝对值或逐任务翻方向修正。模型表现和伪标签质量确有问题，属于方法适用性与模型质量问题。核查不能证明原训练流程、全部影像的配准或标签本身完全正确。','', '## 实际做了什么','',f"- 4个checkpoint SHA一致；对4个源缓存及8个目标缓存各重推理第0张，共{proof['fresh_images_checked']}张，与缓存特征/概率/预测核对通过。",'- 评分阶段禁止目标标注/评价数据读取；从缓存特征独立重算528条Full2048记录、5种方法，再封存；未改旧模型或分数。','- 评价阶段重新读取256张源参考标签，核对采样标签与首张源图完整预测计数；重新读取目标标签复算8任务的TP/FP/FN、F1及AD。','- SciPy独立复算200个方法—任务—种子相关；核对表X/XI裸地8行80个数值，三位小数完全一致。',f"- F1/AD最大差异：{proof['max_F1_AD_difference']:.3g}；各分数最大绝对差见verification.json。FD独立sqrtm在秩亏矩阵产生警告，但与旧PSD算法在预设容差内一致；不影响FCDTM的负相关结论。",'', '## 模型本身的表现','', '|源模型|源参考F1|源参考采样真实前景占比|近零分母维数/总维数|','|---|---:|---:|---:|']
for m in models:lines.append(f"|{name(m['src'])}|{m['source_F1']:.3f}|{100*m['source_sampled_foreground_fraction']:.2f}%|{m['near_zero_dimensions']}/{m['feature_dimensions']}|")
lines+=['','源F1基于固定64张参考图的合并像素；采样占比仅用于诊断。大湾区裸地模型的源参考F1明显低于新疆模型。不能将相关性的负号简单归咎于程序，也不能将低F1解释为其唯一原因。','', '## 8个任务的完整40图表现','', '|源→目标|源F1|目标F1|AD|精确率|召回率|真实裸地占比|预测裸地占比|','|---|---:|---:|---:|---:|---:|---:|---:|']
for t in tasks:lines.append('|'+name(t['src'])+'→'+name(t['tgt'])+'|'+ '|'.join(fmt(t[k]) for k in ['source_F1','target_F1','AD','precision','recall'])+'|'+ '|'.join(f'{100*float(t[k]):.2f}%' for k in ['true_foreground_fraction','pred_foreground_fraction'])+'|')
lines+=['','AD为负表示目标F1高于源参考F1，与相关系数为负不是一回事。预测前景与真实前景占比差异，以及精确率/召回率失衡，说明仅用预测前景的均值不一定能代表真实分类错误。','', '## 表格负相关是否稳定，以及伪标签的作用','', '|源→目标|硬版r|软版r|软版ρ|软版负r种子/5|真实前景均值诊断r|','|---|---:|---:|---:|---:|---:|']
for t in tasks:
 src,tgt=t['src'],t['tgt'];h=row(src,tgt,'FCDTM_HardPL');s=row(src,tgt,'FCDTM_SoftPL');o=row(src,tgt,'oracle_FCDTM');lines.append(f"|{name(src)}→{name(tgt)}|{fmt(h['Pearson'])}|{fmt(s['Pearson'])}|{fmt(s['Spearman'])}|{s['negative_Pearson_seeds']}/5|{fmt(o['Pearson'])}|")
lines+=['','“真实前景均值诊断”仅在封存后，将目标前景均值换为真实标签分组，其余源均值、权重及方向不变。它仅用于解释原因，违反本分支预测时的零目标标签约束，不能列作新的无标签预测方法，也不能据此调参。','', '## 三个关键失败案例','', '大湾区L8→大湾区S2：软版r=-0.407，5个种子均为负。目标精确率仅0.271，预测裸地15.25%，真实裸地7.28%，预测前景受到大量误报污染。换真实前景均值后r约0.008，只是接近无关联，并未恢复有效预测。','', '大湾区S2→大湾区L8：软版r=-0.240（4/5种子为负），硬版却为0.112。换真实前景均值后r约0.172，提示软概率加权与伪标签偏差会影响结果；但这不能证明调整伪标签就能稳定解决问题。','', '大湾区L8→新疆L8：软版r=-0.355，5个种子均为负；使用真实前景均值后仍为-0.295。此任务完整目标F1反而高达0.879、源参考F1仅0.395，说明不能把负相关简单解释为目标模型整体精度很差，或全部归因于伪标签污染。原有带符号的前景加权均值变化也不保证随F1下降单调增加。','', '四个模型均无源均值绝对值小于1e-6的特征维度（0/64），且FCDTM独立重算误差至多约5.7e-13，本次没有支持“近零分母数值爆炸导致负相关”的证据。','', '## 如何解释和处理','', '1. 伪标签前景并不等于真实裸地；在模型较弱或漏报严重时，预测前景均值可能主要反映“模型挑出的部分裸地”，而非全部裸地的迁移变化。','2. F1同时由误报与漏报决定；当前FCDTM以目标前景均值为主，不能保证其带符号的加权变化与F1下降在每个任务上同向。真实前景诊断可区分伪标签偏差与原度量本身的限制，但不构成因果证明。','3. 每个种子仅10个四图批次，且5个种子重用同40图；负号的重复情况属于划分敏感性，不能当作5次独立空间验证。','4. 所有任务保留原方向及负相关结果。后续应在独立数据上做弱模型、伪标签可靠性、背景误报/前景漏报和权重消融；不能为了转正而在这些目标结果上选择符号或参数。','', '![裸地8任务软版散点](figures/Bareland_soft_scatter.png)','', '复现：服务器运行code/audit_bareland.py（依赖原U7B缓存、GDAL、PyTorch、SciPy）；绘图及逐单元核查运行code/summarize_bareland.py --paper /本地/fcdtm-label-free。','']
(D/'实验结果分析.md').write_text('\n'.join(lines))
print('All 80 table cells verified; report and scatter figures created.')
