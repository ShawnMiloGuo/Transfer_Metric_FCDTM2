# -*- coding: utf-8 -*-
from pathlib import Path
import sys,json,csv,math,shutil,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import *
parser=argparse.ArgumentParser();parser.add_argument('--paper',required=True);args=parser.parse_args();P=Path(args.paper);F=B/'figures';F.mkdir(exist_ok=True);D=P/'data/U8';D.mkdir(parents=True,exist_ok=True);PF=P/'Figs/U8';PF.mkdir(parents=True,exist_ok=True)
LABEL['OTCE_PL_SrcCal']='OTCE-Q1024-PL-SrcCal'
S=read(O/'summary.csv');C=read(O/'batch_correlations.csv');R=read(O/'ranking.csv');E=read(O/'evaluated.csv');T=read(O/'task_data.csv');M=json.loads((OLD/'artifacts/manifest_full.json').read_text());V=json.loads((O/'score_verification.json').read_text());cohort={tuple(k) for k in M['cohorts']['D42']}
scens=['region_S2','region_L8','sensor_GBA','sensor_XJ'];titles=['Cross-region S2','Cross-region L8','Cross-sensor GBA','Cross-sensor XJ'];classes=['','Cropland','Forest','Grassland','Shrubland','Wetland','Water','Built-up','Bareland']
for r in E:
 r['cls']=int(r['cls'])
 for k in METHODS+ABL+['true_AD']:r[k]=float(r[k])
 r['common']=r['common']=='True'
fmt=lambda x:'N/A' if not math.isfinite(float(x)) else f'{float(x):.3f}'
def sr(k,s='all'):return next(r for r in S if r['cohort']=='D42' and r['scope']=='common' and r['scenario']==s and r['method']==k)
def rk(k,s='all'):return next(r for r in R if r['cohort']=='D42' and r['scope']=='common' and r['scenario']==s and r['method']==k and r['bootstrap_block']=='class')
def texescape(s):return s.replace('_',r'\_').replace('&',r'\&')
def tab(filename,caption,label,header,rows,spec,foot='',resize=False):
 s='\\begin{table*}[!t]\n\\color{revisioncolor}\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
 if resize:s+='\\resizebox{\\textwidth}{!}{%\n'
 s+='\\begin{tabular}{'+spec+'}\\hline\n'+header+'\\\\\\hline\n'+'\n'.join(' & '.join(r)+r' \\' if isinstance(r,list) else r for r in rows)+'\n\\hline\\end{tabular}\n'
 if resize:s+='}\n'
 s+='\\par\\smallskip{\\footnotesize '+foot+'}\n\\end{table*}\n';(D/filename).write_text(s)
def best(vals):
 fin=[float(x) for x in vals if math.isfinite(float(x))];mx=max(fin) if fin else np.nan
 return [r'\textbf{'+fmt(x)+'}' if math.isfinite(float(x)) and abs(float(x)-mx)<=1e-12 else fmt(x) for x in vals]
rows=[]
for k in METHODS:
 row=[LABEL[k]]
 for scen in ['all']+scens:row += [fmt(sr(k,scen)['Fisher_r']),fmt(sr(k,scen)['rho'])]
 rows.append(row)
# Column-wise winners for overall/scenario method comparison.
for j in range(10):
 vals=[sr(k,(['all']+scens)[j//2])['Fisher_r' if j%2==0 else 'rho'] for k in METHODS]
 for i,v in enumerate(best(vals)):rows[i][j+1]=v
header=r'Method & \multicolumn{2}{c}{Overall} & \multicolumn{2}{c}{Region S2} & \multicolumn{2}{c}{Region L8} & \multicolumn{2}{c}{Sensor GBA} & \multicolumn{2}{c}{Sensor XJ} \\ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$'
tab('overall.tex','All-Method Overall and Scenario Batch Association With AD (D42; Full2048)','tab:overall',header,rows,'l'+'r'*10,r'Pearson uses task Fisher aggregation; Spearman is the arithmetic task mean, both averaged over five seeds. Bold is the largest entry per endpoint. All 16 methods use the same finite batch records. Complete coverage is recorded in the CSV. OTCE-Q1024-PL-SrcCal uses all sampled pixels through a source-fitted quantized feasible coupling; it is approximate, whereas MMD is exact. ',True)
rankrows=[]
for k in METHODS:
 r=rk(k);rankrows.append([LABEL[k],str(r['valid_tasks'])+'/'+str(r['nominal_tasks']),fmt(r['tau_b'])+' ['+fmt(r['tau_lo'])+', '+fmt(r['tau_hi'])+']',fmt(r['Spearman'])+' ['+fmt(r['rho_lo'])+', '+fmt(r['rho_hi'])+']'])
tab('ranking.tex','Across-Transfer-Task Ranking Against Actual AD (D42; Full2048)','tab:task_ranking',r'Method & Tasks & Kendall $\tau_b$ [95\% CI] & Spearman $\rho$ [95\% CI]',rankrows,'llrr',r'One task pools all 40 target images. Intervals use 5000 class-block resamples and are conditional descriptions of the historical task library; source-model-block sensitivity is supplied in the CSV. They do not establish pairwise superiority or independent-region generalization.')
def directionkeys(kind):
 keys=list(cohort)
 if kind=='region':return sorted([k for k in keys if k[1].split('_')[0]!=k[2].split('_')[0]],key=lambda k:(0 if k[1].startswith('dwq') else 1,0 if 'sentinel2' in k[1] else 1,k[0]))
 return sorted([k for k in keys if k[1].split('_')[0]==k[2].split('_')[0]],key=lambda k:(0 if 'sentinel2' in k[1] else 1,0 if k[1].startswith('dwq') else 1,k[0]))
def dirvalue(key,k,col):
 rr=[float(r[col]) for r in C if r['scope']=='common' and r['method']==k and (int(r['cls']),r['src'],r['tgt'])==key];assert len(rr)==5;finite=[x for x in rr if np.isfinite(x)];return float(np.mean(finite)) if finite else np.nan
def dn(s):return s.replace('dwq','GBA').replace('xj','XJ').replace('sentinel2','S2').replace('landsat8','L8').replace('_','-')
for kind in ['region','sensor']:
 rows=[]
 for col,title in [('Pearson',r'Pearson $r$'),('Spearman',r'Spearman $\rho$')]:
  rows.append(r'\multicolumn{19}{l}{\textbf{'+title+r'}}\\\hline');prev=None
  for key in directionkeys(kind):
   cls,src,tgt=key;group=('GBA to XJ' if src.startswith('dwq') else 'XJ to GBA') if kind=='region' else ('S2 to L8' if 'sentinel2' in src else 'L8 to S2')
   if group!=prev:rows.append(r'\multicolumn{19}{l}{'+group+r'}\\\hline');prev=group
   rows.append([dn(src),dn(tgt),classes[cls]]+best([dirvalue(key,k,col) for k in METHODS]))
 header='Source & Target & Class & '+' & '.join(r'\rotatebox{45}{'+texescape(LABEL[k])+'}' for k in METHODS)
 tab('tasks_'+kind+'.tex','All-Method Directional Cross-'+kind.title()+' Comparison (D42; Full2048)','tab:tasks_'+kind,header,rows,'lll'+'r'*16,r'Each entry is the arithmetic mean of five within-task seed correlations. Each row is bolded across all methods using unrounded values (ties within $10^{-12}$). Negative values are retained. Region direction is grouped before sensor; sensor direction is grouped before region. OTCE is the disclosed Q1024 approximation.',True)
for kind in ['region','sensor']:
 rows=[]
 for key in directionkeys(kind):
  vals=[]
  for family in ['Hard','Soft']:
   for k in ['FCDTM_'+family+'Signed','FCDTM_'+family+'Abs','FCDTM_'+family+'PL']:
    vals += [dirvalue(key,k,'Pearson'),dirvalue(key,k,'Spearman')]
  # Within hard/soft families and each endpoint.
  ss=[fmt(x) for x in vals]
  for offset in [0,6]:
   for j in [0,1]:
    indices=[offset+j,offset+2+j,offset+4+j]
    for i,v in zip(indices,best([vals[i] for i in indices])):ss[i]=v
  rows.append([dn(key[1]),dn(key[2]),classes[key[0]]]+ss)
 header=r'Source & Target & Class & \multicolumn{6}{c}{Hard pseudo-labels} & \multicolumn{6}{c}{Soft pseudo-labels} \\ & & & \multicolumn{2}{c}{Signed} & \multicolumn{2}{c}{Abs} & \multicolumn{2}{c}{Weighted} & \multicolumn{2}{c}{Signed} & \multicolumn{2}{c}{Abs} & \multicolumn{2}{c}{Weighted} \\ & & & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$'
 tab('ablation_'+kind+'.tex','Matched Weighting Ablation in Cross-'+kind.title()+' Tasks (D42; Full2048)','tab:ablation_'+kind,header,rows,'lll'+'r'*12,r'The same source reference, target grouping, denominator protection and common records are used in all three variants. Signed sums $d_j$; Abs sums $|d_j|$; Weighted uses the original head weights. No AD-based sign tuning. Bold compares the three variants within each hard/soft family and endpoint.',True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
index=[]
def output(fig,name):
 for ext in ['png','pdf']:
  fig.savefig(F/(name+'.'+ext),dpi=180,bbox_inches='tight');shutil.copy2(F/(name+'.'+ext),PF/(name+'.'+ext))
 plt.close(fig);index.append(name)
def scatter(ax,k,scen):
 rr=[r for r in E if r['kind']=='batch4' and r['seed']=='290117' and (r['cls'],r['src'],r['tgt']) in cohort and r['scenario']==scen and r['common']]
 for cls in sorted({r['cls'] for r in rr}):
  q=[r for r in rr if r['cls']==cls];x=[r[k]*(1 if k in BAD else -1) for r in q];y=[r['true_AD'] for r in q];ax.scatter(x,y,s=15,alpha=.65,label=classes[cls],color=plt.get_cmap('tab10')(cls-1))
 r=sr(k,scen);ax.text(.98,.98,'Fisher r='+fmt(r['Fisher_r'])+'\nmean rho='+fmt(r['rho'])+'\nn='+str(len(rr)),ha='right',va='top',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85));ax.set_title(LABEL.get(k,k),fontsize=10);ax.set_xlabel('AD-oriented metric');ax.set_ylabel('Actual F1 drop');ax.grid(alpha=.15)
families=[METHODS[i:i+4] for i in range(0,16,4)]
for scen,title in zip(scens,titles):
 for j,ks in enumerate(families):
  fig,axs=plt.subplots(2,2,figsize=(11,7.5))
  for ax,k in zip(axs.flat,ks):scatter(ax,k,scen)
  handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=7,fontsize=8);fig.suptitle(title+' | Full2048 | D42');fig.tight_layout(rect=(0,.05,1,.96));output(fig,'batch_'+scen+'_family'+str(j+1))
 for family in ['Soft','Hard']:
  pass
for family in ['Soft','Hard']:
 fig,axs=plt.subplots(4,3,figsize=(12,13))
 for i,(scen,title) in enumerate(zip(scens,titles)):
  for j,k in enumerate(['FCDTM_'+family+'Signed','FCDTM_'+family+'Abs','FCDTM_'+family+'PL']):scatter(axs[i,j],k,scen);axs[i,j].set_title(title+' | '+['Signed difference','Absolute difference','Head weighted'][j],fontsize=9)
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=7,fontsize=8);fig.tight_layout(rect=(0,.025,1,.98));output(fig,'ablation_'+family.lower())
for scen in ['all']+scens:
 fig,axs=plt.subplots(1,2,figsize=(11,7.5),sharey=True);yy=np.arange(16)
 for ax,stat,lo,hi,title in zip(axs,['tau_b','Spearman'],['tau_lo','rho_lo'],['tau_hi','rho_hi'],['Kendall tau-b','Spearman rho']):
  for i,k in enumerate(METHODS):
   r=rk(k,scen);v=float(r[stat]);l=float(r[lo]);h=float(r[hi]);ax.plot([l,h],[i,i],color='C0');ax.plot(v,i,'o',color='C0')
  ax.axvline(0,color='gray',ls='--',lw=.8);ax.set_xlim(-1,1);ax.set_title(title);ax.grid(axis='x',alpha=.2)
 axs[0].set_yticks(yy,[LABEL[k] for k in METHODS]);axs[0].invert_yaxis();fig.suptitle('Full2048 task ordering | '+scen+' | 95% class-block intervals');fig.tight_layout();output(fig,'ranking_'+scen)
for j,ks in enumerate(families):
 fig,axs=plt.subplots(2,2,figsize=(11,7.5))
 for ax,k in zip(axs.flat,ks):
  rr=[r for r in T if (int(r['cls']),r['src'],r['tgt']) in cohort and r['common']=='True'];x=[float(r[k])*(1 if k in BAD else -1) for r in rr];y=[float(r['true_AD']) for r in rr];ax.scatter(x,y,s=25);r=rk(k);ax.text(.98,.98,'tau='+fmt(r['tau_b'])+'\nrho='+fmt(r['Spearman']),transform=ax.transAxes,ha='right',va='top');ax.set_title(LABEL[k]);ax.set_xlabel('Task metric (AD direction)');ax.set_ylabel('Actual task AD');ax.grid(alpha=.2)
 fig.tight_layout();output(fig,'task_family'+str(j+1))
(O/'figure_index.json').write_text(json.dumps(index,indent=2))
# Report all result tables and limits without selecting favorable directions.
lines=['# U8 Full2048统一正式比较与权重消融','', '## 实验目标与约束','', '回答三个问题：统一像素输入后正式方法的比较是否改变；原FCDTM权重是否优于不加权；完整任务排序是否可靠。沿用25个冻结模型、50任务、源64图/目标40图、每图2048点。评分封存后才连接已有评价缓存。NCE和ATC-NE已移除；DoC两版按先前约定保留。D42为42任务主清单，P37和E50仅敏感性。','', 'MMD是全输入精确无偏RBF平方；OTCE使用全输入参与的Q1024源域量化可行耦合近似，保留类比例和簇内方差，并重新做源域校准。后者不是精确全点最优运输，不能把相同像素输入说成相同数值精度。','', '## 总体批次关联','', '|方法|Fisher r|Spearman|任务tau [95%区间]|','|---|---:|---:|---|']
for k in METHODS:
 s=sr(k);r=rk(k);lines.append('|'+LABEL[k]+'|'+fmt(s['Fisher_r'])+'|'+fmt(s['rho'])+'|'+fmt(r['tau_b'])+' ['+fmt(r['tau_lo'])+', '+fmt(r['tau_hi'])+']|')
lines+=['','## 匹配消融','', '|分组|场景|Signed r|Abs r|Weighted r|Signed rho|Abs rho|Weighted rho|','|---|---|---:|---:|---:|---:|---:|---:|']
for family in ['Hard','Soft']:
 for scen in ['all']+scens:
  ks=['FCDTM_'+family+'Signed','FCDTM_'+family+'Abs','FCDTM_'+family+'PL'];lines.append('|'+family+'|'+scen+'|'+'|'.join(fmt(sr(k,scen)[c]) for c in ['Fisher_r','rho'] for k in ks)+'|')
lines+=['','## 计算与局限','',json.dumps(V,ensure_ascii=False,indent=2),'','所有负相关保留。原带符号差异、绝对差异和加权差异使用相同有效记录；各方向详细结果、全部种子与缺失覆盖见CSV。相关性与排序不等于AD绝对预测误差。近似OTCE的源域小规模校验只能量化该诊断范围的误差，不能证明其全输入最优性。类别块仅7/8个，且类别共享图像，区间不等于新地区泛化或方法间显著性。模型仍是二分类U-Net，另一骨干和空间独立确认未完成。','']
(B/'实验结果分析.md').write_text('\n'.join(lines)+(B/'analysis_notes.md').read_text())
for p in O.glob('*.csv'):shutil.copy2(p,D/p.name)
for p in O.glob('*.json'):shutil.copy2(p,D/p.name)
print('OUTPUTS_COMPLETE',len(index),flush=True)
