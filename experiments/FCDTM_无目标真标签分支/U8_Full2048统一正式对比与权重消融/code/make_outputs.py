# -*- coding: utf-8 -*-
"""Full2048 tables, all-method figures and matched original-style ablation."""
from common import *
import numpy as np,math,statistics,argparse
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
P=argparse.ArgumentParser();P.add_argument('--paper');args=P.parse_args();T=O/'tables';T.mkdir(exist_ok=True);F=B/'figures';F.mkdir(exist_ok=True)
LABEL['OTCE_PL_SrcCal']='OTCE-PL-SrcCal-Q1024'
SHORT=dict(zip(METHODS,['FCDTM-H','FCDTM-S','FD','DS','GBC','AC','Entropy','ATC-MC','DoC-gap','DoC-Cal','LogME','H-score','LEEP','NLEEP','OTCE-Q1024','MMD']))
CL={1:'Cropland',2:'Forest',3:'Grassland',4:'Shrubland',5:'Wetland',6:'Water',7:'Built-up',8:'Bareland'};DOM=['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8'];SC=['region_S2','region_L8','sensor_GBA','sensor_XJ'];ST={'region_S2':'Cross-region: Sentinel-2','region_L8':'Cross-region: Landsat-8','sensor_GBA':'Cross-sensor: GBA','sensor_XJ':'Cross-sensor: Xinjiang','all':'All transfer tasks'}
MF={'original':['DS_pixel','FD_global','GBC_spherical_PL','FCDTM_HardPL','FCDTM_SoftPL'],'output':['AC','Entropy','ATC_MC','FCDTM_SoftPL'],'evidence':['LogME_PL','Hscore_PL','LEEP_PL','NLEEP_PL','FCDTM_SoftPL'],'calibration_transport':['DoC','DoC_Acc_SrcCal','OTCE_PL_SrcCal','MMD_RBF_global','FCDTM_SoftPL']}
raw=read(O/'逐项评价.csv');S=read(O/'相关汇总.csv');D=read(O/'方向任务结果.csv');R=read(O/'跨任务排序统计.csv');C=read(O/'逐任务相关.csv');SEL=read(O/'模型选择宏汇总.csv');meta=json.loads((O/'manifest.json').read_text());index=[];annotations=[];tableaudit=[]
for r in raw:
 for k in METHODS+ABL+['true_AD']:r[k]=float(r[k])
 for k in ['cls','seed','budget','repeat']:r[k]=int(r[k])
 for k in ['common','common_soft','common_hard']:r[k]=r[k]=='True'
fmt=lambda v:f'{float(v):.3f}' if math.isfinite(float(v)) else 'N/A'
name=lambda s:s.replace('dwq','GBA').replace('xj','XJ').replace('sentinel2','S2').replace('landsat8','L8').replace('_','-')
def summary(k,sc='all',co='D42',scope='common'):return next(r for r in S if r['cohort']==co and r['scenario']==sc and r['scope']==scope and r['method']==k)
def ranking(k,sc='all',co='D42',scope='common'):return next(r for r in R if r['cohort']==co and r['scenario']==sc and r['scope']==scope and r['method']==k and r['bootstrap_block']=='class')
def bold(v,vals):
 finite=[float(x) for x in vals if math.isfinite(float(x))];text=fmt(v);return r'\textbf{'+text+'}' if finite and math.isfinite(float(v)) and abs(float(v)-max(finite))<=1e-12 else text

def table(filename,caption,label,header,rows,cols,note,size='footnotesize'):
 text='\\begin{table*}[!t]\n\\color{revisioncolor}\\centering\\'+size+'\n\\setlength{\\tabcolsep}{3pt}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+cols+'}\\toprule\n'+header+r'\\\midrule'+'\n'+'\n'.join(' & '.join(r)+r'\\' for r in rows)+'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\par\smallskip{\footnotesize '+note+'}\n\\end{table*}\n';(T/filename).write_text(text)
# Combined former VI + VIII, a single full-input panel.
rows=[]
for k in METHODS:
 r=[LABEL[k]]
 for sc in ['all']+SC:
  ss=summary(k,sc);v=ss['Fisher_r'];best=bold(v,[summary(x,sc)['Fisher_r'] for x in METHODS]);sd=fmt(ss['Fisher_r_seed_sd']);r.extend([best+(r' $\pm$ '+sd if sc=='all' else ''),bold(ss['rho'],[summary(x,sc)['rho'] for x in METHODS])])
 rows.append(r)
head=r'Method & \multicolumn{2}{c}{Overall} & \multicolumn{2}{c}{Region S2} & \multicolumn{2}{c}{Region L8} & \multicolumn{2}{c}{Sensor GBA} & \multicolumn{2}{c}{Sensor XJ} \\ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$'
table('overall_combined.tex','Overall and Scenario-Specific Batch Association With Actual AD (D42; Full2048)','tab:overall',head,rows,'l'+'r'*10,r'All methods use the same sampled inputs (2048 valid pixels/image) and common finite-score batches. Pearson $r$ is Fisher-aggregated across tasks within each seed, then averaged over five seeds; $\rho$ is the corresponding mean within-task Spearman coefficient. Overall $\pm$ values are partition-seed s.d. Bold identifies the largest point estimate in each column, not statistical superiority. OTCE-Q1024 is a full-input approximate transport adaptation; MMD is evaluated exactly by image-block kernel sums. Coverage remains in the CSV.')
rows=[]
for k in METHODS:
 r=ranking(k);rows.append([LABEL[k],fmt(r['tau_b'])+' ['+fmt(r['tau_lo'])+', '+fmt(r['tau_hi'])+']',fmt(r['Spearman'])+' ['+fmt(r['rho_lo'])+', '+fmt(r['rho_hi'])+']'])
r=ranking(METHODS[0]);table('task_ranking.tex','Across-Transfer-Task Ranking Against Actual AD (D42; Full2048)','tab:task_ranking',r'Method & Kendall $\tau_b$ [95\% CI] & Spearman $\rho$ [95\% CI]',rows,'lrr',f"One observation is a complete 40-image task. Common support: {r['valid_tasks']}/{r['nominal_tasks']} tasks. Intervals use 5000 class-block resamples and describe the historical task library; source-model-block and own-support sensitivity accompany the CSV. They do not establish independent-region generalization or pairwise significance. OTCE uses the explicitly approximate Q1024 full-input coupling.")

def order(key,kind):
 cls,src,tgt=key
 return (0 if src.startswith('dwq') else 1,0 if 'sentinel2' in src else 1,cls) if kind=='region' else (0 if 'sentinel2' in src else 1,0 if src.startswith('dwq') else 1,cls)
def details(key,kind):
 cls,src,tgt=key
 if kind=='region':return ['GBA' if src.startswith('dwq') else 'XJ','GBA' if tgt.startswith('dwq') else 'XJ','S2' if 'sentinel2' in src else 'L8',CL[cls]]
 return ['S2' if 'sentinel2' in src else 'L8','S2' if 'sentinel2' in tgt else 'L8','GBA' if src.startswith('dwq') else 'XJ',CL[cls]]
def average(k,keys,scope):
 a=[r for r in C if (int(r['cls']),r['src'],r['tgt']) in keys and r['method']==k and r['scope']==scope];by=defaultdict(list)
 for r in a:by[r['seed']].append(r)
 rr=[];ss=[]
 for vals in by.values():
  pp=[float(r['Pearson']) for r in vals if np.isfinite(float(r['Pearson']))];sp=[float(r['Spearman']) for r in vals if np.isfinite(float(r['Spearman']))];rr.append(float(np.tanh(np.mean(np.arctanh(np.clip(pp,-1+1e-7,1-1e-7))))));ss.append(float(np.mean(sp)))
 return float(np.mean(rr)),float(np.mean(ss))
def direction_table(kind,co='D42',scope='common',ablation=False):
 keys={tuple(k) for k in meta['cohorts'][co] if (k[1].split('_')[0]!=k[2].split('_')[0])==(kind=='region')};keys=sorted(keys,key=lambda k:order(k,kind));methods=METHODS if not ablation else ['FCDTM_SoftSigned','FCDTM_SoftAbs','FCDTM_SoftPL'];lookup={(int(r['cls']),r['src'],r['tgt'],r['method']):r for r in D if r['cohort']==co and r['scope']==scope};caption=('All-Method Directional Cross-'+kind.title()+' Comparison' if not ablation else 'Matched Soft-Pseudo-Label Weighting Ablation in Cross-'+kind.title()+' Transfers')+f' ({co}; Full2048)';label=('tab:tasks_'+kind if not ablation else 'tab:ablation_'+kind);filename=('tasks_' if not ablation else 'ablation_')+kind+('' if co=='D42' else '_'+co)+'.tex';out=['\\begin{table*}[p]','\\color{revisioncolor}\\centering\\scriptsize',r'\setlength{\tabcolsep}{1.3pt}',r'\renewcommand{\arraystretch}{1.05}','\\caption{'+caption+'}','\\label{'+label+('' if co=='D42' else '_'+co)+'}'];heads=['Source','Target','Sensor' if kind=='region' else 'Region','Class']
 metrics=['Pearson','Spearman']
 for metric in metrics:
  out.extend([r'\textbf{'+('(a) Pearson $r$' if metric=='Pearson' else '(b) Spearman $\rho$')+r'}\par\smallskip',r'\resizebox{\textwidth}{!}{\begin{tabular}{llll'+'r'*len(methods)+r'}\toprule'])
  labels=[(r'\shortstack{Feature\\difference}' if k.endswith('Signed') else r'\shortstack{Difference\\+ abs}' if k.endswith('Abs') else r'\shortstack{Difference\\+ weight}') if ablation else r'\rotatebox{55}{'+SHORT[k]+'}' for k in methods];out.append(' & '.join(heads+labels)+r'\\\midrule');previous=None
  for key in keys:
   group=order(key,kind)[0]
   if previous!=group:
    title=('GBA $\rightarrow$ XJ' if group==0 else 'XJ $\rightarrow$ GBA') if kind=='region' else ('Sentinel-2 $\rightarrow$ Landsat-8' if group==0 else 'Landsat-8 $\rightarrow$ Sentinel-2');out.append(r'\multicolumn{'+str(4+len(methods))+r'}{l}{\textit{'+title+r'}}\\');previous=group
   vals=[lookup[*key,k][metric] for k in methods];out.append(' & '.join(details(key,kind)+[bold(v,vals) for v in vals])+r'\\');tableaudit.append(dict(file=filename,metric=metric,cls=key[0],src=key[1],tgt=key[2],methods=methods,values=[float(v) for v in vals],best_methods=[k for k,v in zip(methods,vals) if np.isfinite(float(v)) and abs(float(v)-max(float(x) for x in vals if np.isfinite(float(x))))<=1e-12]))
  vals=[average(k,set(keys),scope)[0 if metric=='Pearson' else 1] for k in methods];out.append(r'\midrule\multicolumn{4}{l}{'+('Fisher aggregate' if metric=='Pearson' else 'Mean task correlation')+'} & '+' & '.join(bold(v,vals) for v in vals)+r'\\\bottomrule\end{tabular}}\par\smallskip')
 out.append(r'{\footnotesize Rows average five seed-specific within-task coefficients. All methods share identical finite-score batches; the Pearson footer uses Fisher aggregation. Bold marks the largest unrounded row value (ties within $10^{-12}$), including negative results. '+('The three variants differ only in dimensionwise aggregation; weighted entries exactly match the formal comparison on this common support.' if ablation else 'Abbreviations follow the method register; OTCE-Q1024 is approximate and source-calibrated. All 16 formal entries appear in both coefficient panels.')+r'}');out.append(r'\end{table*}');(T/filename).write_text('\n'.join(out)+'\n')
for co in ['D42','E50']:
 for kind in ['region','sensor']:
  direction_table(kind,co);direction_table(kind,co,'ablation_soft',True)
rows=[]
for k in METHODS:
 r=next(r for r in SEL if r['cohort']=='D42' and r['method']==k);rows.append([LABEL[k],f"{float(r['regret']):.4f}",f"{100*float(r['Top1']):.1f}",f"{float(r['max_loss']):.4f}"])
table('supp_selection.tex','Supplementary Native-Score Model Selection (D42; Full2048)','tab:supp_selection',r'Method & Mean regret & Top-1 (\%) & Maximum loss',rows,'lrrr','Budgets 8/16/24 count unlabeled images, with five fixed draws each. Choice uses only sealed scores. Target F1 is evaluated on the same sampled subset after sealing, not an independent spatial holdout. All 16 methods share '+r['common_configs']+'/'+r['total_configs']+' supported configurations; repeats, budgets and class--target groups are macro-averaged. OTCE is the Q1024 full-input approximation.')
rows=[]
for co in ['P37','D42','E50']:
 for k in ['FCDTM_HardPL','FCDTM_SoftPL','FD_global','Hscore_PL','OTCE_PL_SrcCal','MMD_RBF_global']:
  s=summary(k,co=co);r=ranking(k,co=co);rows.append([co,LABEL[k],fmt(s['Fisher_r']),fmt(s['rho']),fmt(r['tau_b']),fmt(r['Spearman'])])
table('cohorts.tex','Task-List Sensitivity Under the Unified Full2048 Protocol','tab:cohorts',r'Cohort & Method & Batch $r$ & Batch $\rho$ & Task $\tau_b$ & Task $\rho$',rows,'llrrrr','P37 is the originally tabulated task list, D42 the completed original design and E50 the available eight-class extension. These historical pools are sensitivity analyses, not independent spatial confirmations. All-method estimates and intervals accompany the CSV.')
# Method register: no target reference labels enter score generation.
rows=[];required={'FCDTM_HardPL','FCDTM_SoftPL','ATC_MC','DoC_Acc_SrcCal','OTCE_PL_SrcCal'}
for k in METHODS:
 target='Hard pseudo-labels' if k in ['FCDTM_HardPL','GBC_spherical_PL','LogME_PL','Hscore_PL','LEEP_PL','NLEEP_PL','OTCE_PL_SrcCal'] else 'Soft probabilities' if k=='FCDTM_SoftPL' else 'Predicted groups' if k=='DS_pixel' else 'None';rows.append([LABEL[k],'Yes' if k in required else 'No',target,'Full2048'])
table('reviewer_method_coverage.tex','Formal Comparator Register Under the Unified Full2048 Protocol','tab:reviewer_coverage',r'Method & Source labels & Target grouping & Input',rows,'llll','All target inputs are unlabeled. PL denotes hard pseudo-label adaptation; source-only calibration is disclosed for DoC and OTCE. OTCE-Q1024 uses a feasible approximate coupling constructed from all input pixels; MMD is exact. The study still evaluates binary U-Net checkpoints.')
# Plots: the same common points and color assignment are used by all methods.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':14,'axes.labelsize':15,'legend.fontsize':12,'pdf.fonttype':42,'ps.fonttype':42})
def color(task):return plt.get_cmap('tab20')(2*(task[0]-1)+int(DOM.index(task[1])>DOM.index(task[2])))
def draw(ax,k,groups,co,sc,scope='common',task=False):
 xx=[];yy=[];handles=[];labs=[]
 for key,rr in groups.items():
  x=[r[k]*(1 if k in BAD else -1) for r in rr];y=[r['true_AD'] for r in rr];xx+=x;yy+=y;h=ax.scatter(x,y,s=25,alpha=.8,color=color(key));handles.append(h);labs.append(name(key[1])+' → '+name(key[2])+' '+CL[key[0]])
 if task:
  sr=ranking(k,sc,co,scope);text=r'$\tau_b$'+' = '+fmt(sr['tau_b'])+'\n'+r'$\rho$'+' = '+fmt(sr['Spearman'])+'\n'+f"n = {len(xx)}"
 else:
  sr=summary(k,sc,co,scope);text=r'$r_{\mathrm F,5}$'+' = '+fmt(sr['Fisher_r'])+'\n'+r'$\bar\rho_5$'+' = '+fmt(sr['rho'])+'\n'+f"n = {len(xx)}";annotations.append(dict(cohort=co,scenario=sc,scope=scope,method=k,r=fmt(sr['Fisher_r']),rho=fmt(sr['rho']),points=len(xx)))
 ax.text(.985,.985,text,transform=ax.transAxes,ha='right',va='top',fontsize=14,bbox=dict(facecolor='white',edgecolor='#ddd',alpha=.95,pad=3));lab=LABEL[k] if k in BAD else '-'+LABEL[k];ax.set(xlabel=lab+' (AD direction)',ylabel='Accuracy drop');ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5));ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4),useOffset=False)
 if yy:
  span=max(max(yy)-min(yy),.1);ax.set_ylim(min(yy)-.05*span,max(yy)+.38*span)
 ax.grid(alpha=.15);return handles,labs

def storefig(fig,filename,**meta):
 for ext in ['png','pdf']:fig.savefig(F/(filename+'.'+ext),dpi=180,bbox_inches='tight',facecolor='white')
 index.append(dict(file=filename,**meta));plt.close(fig)
for co in ['D42','E50']:
 ck={tuple(k) for k in meta['cohorts'][co]}
 with PdfPages(F/(co+'_all_method_batch_book.pdf')) as book:
  for sc in SC:
   selected=[r for r in raw if r['kind']=='batch4' and r['seed']==290117 and r['scenario']==sc and (r['cls'],r['src'],r['tgt']) in ck and r['common']];groups={key:[r for r in selected if (r['cls'],r['src'],r['tgt'])==key] for key in sorted({(r['cls'],r['src'],r['tgt']) for r in selected})}
   for family,methods in MF.items():
    fig,axes=plt.subplots(3,2,figsize=(12.6,13.2));handles=labs=None
    for ax,k in zip(axes.flat,methods):handles,labs=draw(ax,k,groups,co,sc)
    for ax in list(axes.flat)[len(methods):]:ax.axis('off')
    axes.flat[-1].legend(handles,labs,loc='center left',frameon=True,fontsize=12 if len(labs)<=12 else 10.5);fig.suptitle(ST[sc]+' | Full2048 | '+co+'\n'+family.replace('_',' '),fontsize=17,y=.995);fig.tight_layout(rect=(0,0,1,.955),h_pad=1.8,w_pad=1.5);book.savefig(fig,bbox_inches='tight');storefig(fig,co+'_'+sc+'_'+family,kind='batch',cohort=co,scenario=sc,methods=methods)
 # Original Fig8 logic: 4 scenarios by signed / absolute / weighted aggregation.
 for fam in ['soft','hard']:
  methods=['FCDTM_'+('SoftSigned' if fam=='soft' else 'HardSigned'),'FCDTM_'+('SoftAbs' if fam=='soft' else 'HardAbs'),'FCDTM_'+('SoftPL' if fam=='soft' else 'HardPL')];fig,axes=plt.subplots(4,4,figsize=(15.5,17.5),gridspec_kw={'width_ratios':[1,1,1,.85]})
  for i,sc in enumerate(SC):
   selected=[r for r in raw if r['kind']=='batch4' and r['seed']==290117 and r['scenario']==sc and (r['cls'],r['src'],r['tgt']) in ck and r['common'] and r['common_'+fam]];groups={key:[r for r in selected if (r['cls'],r['src'],r['tgt'])==key] for key in sorted({(r['cls'],r['src'],r['tgt']) for r in selected})}
   for j,k in enumerate(methods):
    handles,labs=draw(axes[i,j],k,groups,co,sc,'ablation_'+fam);axes[i,j].set_xlabel(['Feature difference','Difference + abs','Difference + weight'][j]);axes[i,j].set_title(ST[sc] if j==1 else '',fontsize=16)
   axes[i,3].axis('off');axes[i,3].legend(handles,labs,loc='center left',frameon=False,fontsize=10.5,handletextpad=.3,labelspacing=.6)
  fig.suptitle('Matched '+fam+'-pseudo-label weighting ablation | Full2048 | '+co,fontsize=19,y=.995);fig.tight_layout(rect=(0,0,1,.97),h_pad=2.2,w_pad=1.2);storefig(fig,co+'_ablation_'+fam+'_fig8',kind='ablation',cohort=co,methods=methods)
 # Complete task scatter family book and confidence forests.
 taskrows=[r for r in raw if r['kind']=='pool' and r['budget']==40 and (r['cls'],r['src'],r['tgt']) in ck and r['common']];groups={(r['cls'],r['src'],r['tgt']):[r] for r in taskrows}
 with PdfPages(F/(co+'_all_method_task_book.pdf')) as book:
  for family,methods in MF.items():
   fig,axes=plt.subplots(3,2,figsize=(12.6,13.2))
   for ax,k in zip(axes.flat,methods):handles,labs=draw(ax,k,groups,co,'all',task=True)
   for ax in list(axes.flat)[len(methods):]:ax.axis('off')
   axes.flat[-1].text(.02,.8,'One point = one complete 40-image task\nColors encode class and transfer direction.\nNative score scales are not target-calibrated.',transform=axes.flat[-1].transAxes,fontsize=15,va='top');fig.suptitle('Across-task AD ordering | Full2048 | '+co+'\n'+family.replace('_',' '),fontsize=17);fig.tight_layout(rect=(0,0,1,.95),h_pad=2);book.savefig(fig,bbox_inches='tight');storefig(fig,co+'_tasks_'+family,kind='task',cohort=co,methods=methods)
 for sc in ['all']+SC:
  fig,axes=plt.subplots(1,2,figsize=(12.5,8.2),sharey=True);yy=np.arange(len(METHODS))
  for ax,metric in zip(axes,['tau','rho']):
   val=[];lo=[];hi=[]
   for k in METHODS:
    r=ranking(k,sc,co);val.append(float(r['tau_b'] if metric=='tau' else r['Spearman']));lo.append(float(r[metric+'_lo']));hi.append(float(r[metric+'_hi']))
   val=np.array(val);lo=np.array(lo);hi=np.array(hi);ax.errorbar(val,yy,xerr=np.array([val-lo,hi-val]),fmt='o',capsize=3,color='#285779');ax.axvline(0,color='#888',lw=1,ls='--');ax.set_xlim(-1,1);ax.set_xlabel('Kendall tau-b' if metric=='tau' else 'Spearman rho');ax.grid(axis='x',alpha=.15)
  axes[0].set_yticks(yy,[LABEL[k] for k in METHODS],fontsize=11);axes[0].invert_yaxis();fig.suptitle(ST[sc]+' | task ranking | Full2048 | '+co,fontsize=16);fig.tight_layout(rect=(0,.03,1,.95));fig.text(.5,.01,'95% class-block descriptive intervals; fixed historical domains, not independent-region uncertainty.',ha='center',fontsize=11);storefig(fig,co+'_'+sc+'_intervals',kind='interval',cohort=co,scenario=sc,methods=METHODS)
(F/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n');(F/'annotation_audit.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2)+'\n');dump(O/'table_bolding_audit.json',tableaudit)
if args.paper:
 import shutil
 paper=Path(args.paper);(paper/'data/U8').mkdir(exist_ok=True);(paper/'Figs/U8').mkdir(exist_ok=True)
 for p in O.glob('*'):
  if p.is_file() and p.suffix in ['.csv','.json']:shutil.copy2(p,paper/'data/U8'/p.name)
 for p in T.glob('*.tex'):shutil.copy2(p,paper/'data'/p.name)
 for p in F.glob('*'):
  if p.is_file():shutil.copy2(p,paper/'Figs/U8'/p.name)
print('U8_OUTPUTS',len(index),'figure pairs; complete full-method and ablation tables',flush=True)
