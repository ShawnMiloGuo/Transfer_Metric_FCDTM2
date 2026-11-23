"""Render all-method task-ranking evidence from the sealed evaluation only."""
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from common import *
assert (O/'evaluation.json').exists()
F=B/'figures';F.mkdir(exist_ok=True);rows=read(O/'跨任务排序统计.csv');tasks=read(O/'任务级数据.csv');meta=json.loads((O/'task_manifest.json').read_text())
for r in tasks:
 r['cls']=int(r['cls']);r['true_AD']=float(r['true_AD']);r['common']=r['common']=='True'
 for k in METHODS:r[k]=float(r[k])
def stat(k,p='full2048',c='D42',s='all',block='class'):
 return next(r for r in rows if r['method']==k and r['panel']==p and r['cohort']==c and r['scenario']==s and r['scope']=='common' and r['bootstrap_block']==block)
def cell(r,key):
 if not r:return '--'
 point=fmt(r['tau_b'] if key=='tau' else r['Spearman']);lo,hi=fmt(r[key+'_lo']),fmt(r[key+'_hi'])
 if point=='N/A':return 'N/A'
 return r'\shortstack{$'+point+r'$\\$['+lo+','+hi+r']$}' if lo!='N/A' else r'$'+point+r'$ (CI N/A)'
ls=[]
for k in METHODS:
 v=[LABEL[k]]
 for panel in PANELS:
  r=stat(k,panel) if k in PANELS[panel] else None;v.extend([cell(r,'tau'),cell(r,'rho')])
 ls.append(' & '.join(v)+r' \\')
q=stat('FCDTM_SoftPL');z=stat('FCDTM_SoftPL','shared256')
tex=r'''\begin{table*}[!t]
\color{revisioncolor}\centering\footnotesize
\setlength{\tabcolsep}{6pt}
\caption{Across-Transfer-Task Ranking Against Actual AD (D42)}
\label{tab:task_ranking}
\begin{tabular}{lcccc}\hline
Method & \multicolumn{2}{c}{Full2048} & \multicolumn{2}{c}{Shared256} \\
 & Kendall $\tau_b$ [95\% CI] & Spearman $\rho$ [95\% CI] & Kendall $\tau_b$ [95\% CI] & Spearman $\rho$ [95\% CI] \\\hline
'''+ '\n'.join(ls)+r'''
\hline\end{tabular}
\par\smallskip{\footnotesize Each point is one class--source-model--target-domain task using all 40 target images, not a four-image batch or partition seed. Common support: '''+q['valid_tasks']+'/42 (Full2048), '+z['valid_tasks']+r'''/42 (Shared256). Native directions are fixed toward larger AD. Intervals use 5000 class-block resamples ('''+q['usable_blocks']+r''' represented classes), retaining all within-class tasks. They are descriptive conditional intervals, not independent-region confidence statements. Source-model-block and own-support results accompany the data. NCE's constant score yields N/A, not zero correlation. -- denotes a method outside that panel.}
\end{table*}
'''
(O/'task_ranking_main.tex').write_text(tex)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':14,'axes.labelsize':16,'pdf.fonttype':42,'ps.fonttype':42})
index=[];annotation=[]
for panel,methods in PANELS.items():
 for cohort in COHORTS:
  for scen in SCENARIOS:
   if cohort!='D42' and scen!='all':continue
   rr=[stat(k,panel,cohort,scen) for k in methods];fig,axes=plt.subplots(1,2,figsize=(12.6,9),sharey=True)
   for j,(ax,key,field) in enumerate(zip(axes,['tau','rho'],['tau_b','Spearman'])):
    for i,r in enumerate(rr):
     v=float(r[field]);lo,hi=float(r[key+'_lo']),float(r[key+'_hi']);color='#245080' if 'FCDTM' in r['method'] else '#28796c' if r['method'] in ['FD_global','MMD_RBF_global'] else '#555555'
     if np.isfinite(v):
      if np.isfinite(lo) and np.isfinite(hi):ax.plot([lo,hi],[i,i],color=color,lw=1.8);ax.plot([lo,lo],[i-.10,i+.10],color=color);ax.plot([hi,hi],[i-.10,i+.10],color=color)
      ax.plot(v,i,'o',color=color,markersize=5)
     else:ax.text(.05,i,'N/A (constant / unsupported)',va='center',fontsize=10,color='gray')
    ax.axvline(0,color='#aaa',lw=.8,ls='--');ax.set_xlim(-1.02,1.02);ax.set_ylim(len(rr)-.5,-.5);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Kendall '+r'$\tau_b$' if key=='tau' else 'Spearman '+r'$\rho$');ax.set_yticks(range(len(rr)));ax.set_yticklabels([LABEL[r['method']] for r in rr] if j==0 else []);ax.tick_params(axis='y',labelleft=j==0)
   # Shared y axis tick labels must be set once, not overwritten through shared locator.
   axes[0].set_yticklabels([LABEL[r['method']] for r in rr]);axes[1].tick_params(labelleft=False)
   r=rr[0];fig.suptitle(f"{cohort} | {scen.replace('_',' ')} | {panel}\nTask-level ordering; 95% class-block percentile intervals",fontsize=16);fig.text(.5,.018,f"Common tasks: {r['valid_tasks']}/{r['nominal_tasks']}; class blocks: {r['usable_blocks']}; 5000 resamples. Historical domains held fixed.",ha='center',fontsize=11);fig.tight_layout(rect=(0,.045,1,.93),w_pad=2)
   name=f'{cohort}_{panel}_{scen}_intervals'
   for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),dpi=170,bbox_inches='tight')
   plt.close(fig);index.append(dict(kind='intervals',cohort=cohort,panel=panel,scenario=scen,file='figures/'+name+'.pdf',methods=methods))
families={'original':['DS_pixel','FD_global','GBC_spherical_PL','FCDTM_HardPL','FCDTM_SoftPL'],'output':['AC','Entropy','ATC_MC','ATC_NE','FCDTM_SoftPL'],'label_evidence':['LogME_PL','Hscore_PL','LEEP_PL','NCE_Pred_PL','FCDTM_SoftPL'],'confidence_clustering':['DoC','DoC_Acc_SrcCal','NLEEP_PL','FCDTM_SoftPL'],'transport':['OTCE_PL_SrcCal','MMD_RBF_global','FCDTM_SoftPL']}
classes={1:'Cropland',2:'Forest',3:'Grassland',4:'Shrubland',5:'Wetland',6:'Water',7:'Built-up',8:'Bareland'};markers={'region_S2':'o','region_L8':'s','sensor_GBA':'^','sensor_XJ':'D'};DOM=['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8']
for panel,methods in PANELS.items():
 for cohort in ['D42','E50']:
  ck={tuple(k) for k in meta['cohorts'][cohort]};q=[r for r in tasks if r['panel']==panel and (r['cls'],r['src'],r['tgt']) in ck and r['common']]
  with PdfPages(F/f'{cohort}_{panel}_task_scatter_book.pdf') as book:
   for family,keys in families.items():
    if family=='transport' and panel=='full2048':continue
    fig,axes=plt.subplots(3,2,figsize=(12.6,13.2))
    for ax,k in zip(axes.ravel(),keys):
     xx=[];yy=[]
     for r in q:
      x=r[k]*(1 if k in BAD else -1);y=r['true_AD'];color=plt.get_cmap('tab10')(r['cls']-1);filled=DOM.index(r['src'])<DOM.index(r['tgt']);ax.scatter(x,y,s=35,marker=markers[r['scenario']],facecolor=color if filled else 'none',edgecolor=color,linewidth=.9);xx.append(x);yy.append(y)
     s=stat(k,panel,cohort);caption=(r'$\tau_b$'+f" = {fmt(s['tau_b'])} [{fmt(s['tau_lo'])}, {fmt(s['tau_hi'])}]\n"+r'$\rho$'+f" = {fmt(s['Spearman'])} [{fmt(s['rho_lo'])}, {fmt(s['rho_hi'])}]\n"+f"N = {s['valid_tasks']}; class blocks = {s['usable_blocks']}\n95% class-block CI")
     ax.text(.985,.985,caption,transform=ax.transAxes,ha='right',va='top',fontsize=12.5,bbox=dict(facecolor='white',edgecolor='#ccc',alpha=.96,boxstyle='round,pad=.25'));ax.set_xlabel(('' if k in BAD else '-')+LABEL[k]+' (AD direction)',fontsize=13.5);ax.set_ylabel('Accuracy Drop (task level)',fontsize=13.5);ax.tick_params(labelsize=12)
     if yy:
      span=max(max(yy)-min(yy),.1);ax.set_ylim(min(yy)-.05*span,max(yy)+.62*span)
     ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4),useOffset=False);annotation.append(dict(panel=panel,cohort=cohort,method=k,tau_b=s['tau_b'],tau_lo=s['tau_lo'],tau_hi=s['tau_hi'],Spearman=s['Spearman'],rho_lo=s['rho_lo'],rho_hi=s['rho_hi'],N=len(q)))
    for ax in axes.ravel()[len(keys):]:ax.axis('off')
    handles=[Line2D([],[],marker='o',color='none',markerfacecolor=plt.get_cmap('tab10')(c-1),markeredgecolor=plt.get_cmap('tab10')(c-1),label=label,markersize=6) for c,label in classes.items() if any(r['cls']==c for r in q)]
    handles += [Line2D([],[],marker=m,color='none',markerfacecolor='white',markeredgecolor='black',label=sc.replace('_',' '),markersize=6) for sc,m in markers.items()]
    handles += [Line2D([],[],marker='o',color='none',markerfacecolor='black',markeredgecolor='black',label='Forward domain order',markersize=6),Line2D([],[],marker='o',color='none',markerfacecolor='white',markeredgecolor='black',label='Reverse domain order',markersize=6)]
    axes.ravel()[-1].legend(handles=handles,loc='center left',fontsize=11.5,labelspacing=.4,frameon=True);fig.suptitle(f"Across-transfer-task ranking | {cohort} | {panel}\n{family.replace('_',' ')}; one point per 40-image task",fontsize=16,y=.998);fig.tight_layout(rect=(0,0,1,.95),h_pad=2,w_pad=1.6)
    name=f'{cohort}_{panel}_{family}_task_scatter'
    for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),dpi=170,bbox_inches='tight')
    book.savefig(fig,bbox_inches='tight');plt.close(fig);index.append(dict(kind='task_scatter',cohort=cohort,panel=panel,scenario='all',file='figures/'+name+'.pdf',methods=keys))
(F/'index.json').write_text(json.dumps(index,indent=2));(F/'annotation_audit.json').write_text(json.dumps(annotation,indent=2))
# Plain-language analysis generated from the same rows.
s=stat('FCDTM_SoftPL');z=stat('FCDTM_SoftPL','shared256');m=stat('MMD_RBF_global','shared256');h=stat('FCDTM_HardPL')
text=f'''# U7R 跨迁移任务排序：实验结果分析

## 实验要回答什么

回应Reviewer 2第3(b)条：FCDTM能否把“哪些迁移任务下降更多”排对，而不只是在同一任务内跟踪四图批次。这次一个点就是“某地物类别、某源模型、某目标域”完整40图的任务。任务AD=源域F1−目标40图合并像素F1。排序AD不等于选择目标F1最高的模型；原表VII移入补充材料，结果保留。

沿用U7B全部18条目，Full2048实际适用16项、Shared256适用18项。D42主清单42任务（7类），P37／E50敏感性；E50包含8类。两采样面板分开。直接复用已有40图原生分数，不改变公式、方向或源校准。先封存100条任务分数，再独立连接原U7B评价缓存；评分入口无法读取目标标注或评价缓存，未重新读取任何标注影像。

## 主要结论

Full2048中软FCDTM：tau-b={fmt(s['tau_b'])}，95%类别块区间[{fmt(s['tau_lo'])}, {fmt(s['tau_hi'])}]；Spearman={fmt(s['Spearman'])}，区间[{fmt(s['rho_lo'])}, {fmt(s['rho_hi'])}]。这是跨{ s['valid_tasks']}个任务的关联，和之前批次Fisher r=0.634不是同一个指标。硬版tau-b={fmt(h['tau_b'])}，不能说软版所有终点都更好。

Shared256中软版tau-b={fmt(z['tau_b'])}，区间[{fmt(z['tau_lo'])}, {fmt(z['tau_hi'])}]，Spearman={fmt(z['Spearman'])}。区间跨零，说明在这个小像素预算的历史任务池上，正向排序的证据不够稳固。MMD的tau-b={fmt(m['tau_b'])}，区间[{fmt(m['tau_lo'])}, {fmt(m['tau_hi'])}]。不能据区间重叠或不重叠直接断言方法间显著差异，也不能拼接两面板的有利数字。

'''
for panel,methods in PANELS.items():
 text+='## '+panel+'：D42全部正式方法\n\n|方法|任务数|tau-b [95% CI]|Spearman [95% CI]|类别块|有效tau/rho重复|\n|---|---:|---|---|---:|---|\n'
 for k in methods:
  r=stat(k,panel);text+=f"|{LABEL[k]}|{r['valid_tasks']}/{r['nominal_tasks']}|{fmt(r['tau_b'])} [{fmt(r['tau_lo'])}, {fmt(r['tau_hi'])}]|{fmt(r['Spearman'])} [{fmt(r['rho_lo'])}, {fmt(r['rho_hi'])}]|{r['usable_blocks']}|{r['tau_valid_bootstrap']}/{r['rho_valid_bootstrap']}|\n"
text+='\n## 四种场景：软FCDTM\n\n|面板|场景|任务数|tau-b [95% CI]|Spearman [95% CI]|\n|---|---|---:|---|---|\n'
for panel in PANELS:
 for scen in SCENARIOS[1:]:
  r=stat('FCDTM_SoftPL',panel,s=scen);text+=f"|{panel}|{scen}|{r['valid_tasks']}|{fmt(r['tau_b'])} [{fmt(r['tau_lo'])}, {fmt(r['tau_hi'])}]|{fmt(r['Spearman'])} [{fmt(r['rho_lo'])}, {fmt(r['rho_hi'])}]|\n"
text+='''\n每场景只有少量任务，区间很宽，应保留不利和跨零结果。全部方法的四场景、三清单、own/common及源模型块敏感性均在跨任务排序统计.csv。不是只报告FCDTM。\n\n## 区间的含义与限制\n\n主区间按地物类别整块抽样5000次，保留同类内所有模型和迁移方向，避免把50个四图/种子结果重复算成独立任务。源模型块区间作为敏感性。类别块只有7或8个，且跨类别共用目标影像，不能完全消除依赖；它是固定历史域条件下的描述性区间，不是新地区总体置信结论，不是新的空间独立确认。至少3块且4000有效重复才给区间；常量或退化重复不补0。\n\nNCE自伪标签恒0，因此点估计和区间N/A，仍保留在正式对照中。共同任务缺失基于分数支持和AD是否可定义，不根据成绩挑选；own覆盖另存。Shared256共同支持少于42，必须与排序数值一起解释。\n\n## 为什么不能用本结果代替原表VII\n\n跨任务排序问“哪个任务的F1下降更多”，表VII问“对某个目标选哪个模型能取得更高F1”。源模型F1不同，下降少的模型未必目标F1最高。新任务排序直接回应审稿人；旧选模转为补充终点，保留FD／MMD优于FCDTM的结果，不因不利而删除。8／16／24仍是补充选模的无标签图数，本轮任务排序统一是40图池。
\n## 对返修结论的影响\n\n可以说FCDTM在较充分采样下，在固定历史任务池中存在跨任务AD排序关联；不能只靠批次相关宣称这一点，也不能说所有采样预算都稳定、更不能宣称最佳候选选择器。新结果仍未单独证明分类头权重相对无权重的贡献，U7C还需做；新架构和未用网格的确认属U7D。
\n本轮复用此前已评价过的历史池，是补充统计分析，不是盲测或前瞻性预注册确认。已有方法分数在新分析前已固定，新任务投影再次封存，未用目标AD拟合、重排符号、调参或挑方案。完整输入SHA、评分隔离及独立核验见artifacts。
'''

if (B/'analysis_notes.md').exists():text+='\n'+(B/'analysis_notes.md').read_text()
(B/'实验结果分析.md').write_text(text)
(B/'图表索引.md').write_text('# U7R 图表索引\n\n'+ '\n'.join(f"- [{r['cohort']} / {r['panel']} / {r['scenario']} / {r['kind']}]({r['file']})" for r in index)+'\n')
print('Wrote main ranking table, analysis and',len(index),'figure pairs',flush=True)
