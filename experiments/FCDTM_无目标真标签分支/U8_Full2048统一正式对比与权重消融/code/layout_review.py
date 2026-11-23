# -*- coding: utf-8 -*-
"""PDF comments: paired coefficients, best/runner-up and directional legends."""
from pathlib import Path
import sys,csv,json,re,shutil,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'code'));from common import METHODS,LABEL,BAD,OLD
P=Path(sys.argv[1]);D=P/'data/U8';F=B/'figures';PF=P/'Figs/U8';read=lambda p:list(csv.DictReader(p.open()));C=read(B/'artifacts/batch_correlations.csv');E=read(B/'artifacts/evaluated.csv');S=read(B/'artifacts/summary.csv');m=json.loads((OLD/'artifacts/manifest_full.json').read_text());keys={tuple(x) for x in m['cohorts']['D42']};classes=['','Cropland','Forest','Grassland','Shrubland','Wetland','Water','Built-up','Bareland']
scens=['region_S2','region_L8','sensor_GBA','sensor_XJ'];titles=['Cross-region Sentinel-2','Cross-region Landsat-8','Cross-sensor GBA','Cross-sensor XJ'];audit=[];taudit=[]
def dn(s):return s.replace('dwq','GBA').replace('xj','XJ').replace('sentinel2','S2').replace('landsat8','L8').replace('_','-')
heads=[r'FCDTM\\HardPL',r'FCDTM\\SoftPL','FD','DS',r'GBC\\PL',r'AC /\\MSP','Entropy',r'ATC\\MC',r'DoC\\gap',r'DoC-Acc\\SrcCal',r'LogME\\ICML-PL',r'H-score\\PL',r'LEEP\\PL',r'NLEEP\\PL',r'OTCE-Q1024\\PL-SrcCal',r'MMD-RBF\\global']
def val(key,k,col):
 xs=[float(r[col]) for r in C if r['scope']=='common' and r['method']==k and (int(r['cls']),r['src'],r['tgt'])==key];assert len(xs)==5;return float(np.nanmean(xs))
def decorate(v):
 levels=[]
 for x in sorted([x for x in v if np.isfinite(x)],reverse=True):
  if not levels or abs(x-levels[-1])>1e-12:levels.append(x)
 out=[];status=[]
 for x in v:
  text=f'{x:.3f}' if np.isfinite(x) else 'N/A';rank=1 if levels and abs(x-levels[0])<=1e-12 else 2 if len(levels)>1 and abs(x-levels[1])<=1e-12 else 0;status.append(rank);out.append((r'\textbf{'+text+'}') if rank==1 else (r'\underline{'+text+'}') if rank==2 else text)
 return out,status
for kind in ['region','sensor']:
 kk=[k for k in keys if (k[1].split('_')[0]==k[2].split('_')[0])==(kind=='sensor')]
 kk.sort(key=lambda k:(0 if (k[1].startswith('dwq') if kind=='region' else 'sentinel2' in k[1]) else 1,0 if ('sentinel2' in k[1] if kind=='region' else k[1].startswith('dwq')) else 1,k[0]))
 header=r'\multirow{3}{*}{Source} & \multirow{3}{*}{Target} & \multirow{3}{*}{Class} & '+' & '.join(r'\multicolumn{2}{c}{\shortstack{'+h+'}}' for h in heads)+r'\\'+'\n'+r' & & & '+' & '.join([r'$r$ & $\rho$']*16)+r'\\\hline'+'\n'
 rows=[];previous=None
 for key in kk:
  cls,src,tgt=key;group=('GBA to XJ' if src.startswith('dwq') else 'XJ to GBA') if kind=='region' else ('S2 to L8' if 'sentinel2' in src else 'L8 to S2')
  if group!=previous:rows.append(r'\multicolumn{35}{l}{\textbf{'+group+r'}}\\\hline');previous=group
  rv=[val(key,k,'Pearson') for k in METHODS];sv=[val(key,k,'Spearman') for k in METHODS];rd,rr=decorate(rv);sd,sr=decorate(sv);cells=[]
  for i,k in enumerate(METHODS):cells.extend([rd[i],sd[i]]);taudit.append(dict(kind=kind,cls=cls,src=src,tgt=tgt,method=k,r=rv[i],rho=sv[i],r_rank=rr[i],rho_rank=sr[i]))
  rows.append(' & '.join([dn(src),dn(tgt),classes[cls]]+cells)+r'\\')
 text=r'''\begin{sidewaystable*}[p]
\color{revisioncolor}\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.25}
\caption{All-Method Directional Cross-'''+kind.title()+r''' Comparison (D42; Full2048)}
\label{tab:tasks_'''+kind+r'''}
\resizebox{\textheight}{!}{%
\begin{tabular}{lll'''+('rr'*16)+r'''}\hline
'''+header+'\n'.join(rows)+r'''
\hline\end{tabular}}
\par\smallskip\begin{minipage}{\textheight}\footnotesize Each method has adjacent Pearson $r$ and Spearman $\rho$ columns. Entries average five within-task seed correlations. Within each task and each coefficient, the best value is bold and the second distinct best value is underlined, using unrounded values and a $10^{-12}$ tie tolerance. Long method names are wrapped onto two lines. Negative associations are retained. OTCE-Q1024 uses the disclosed approximation.\end{minipage}
\end{sidewaystable*}
'''
 # Only two header rows, so span two rather than three.
 text=text.replace('multirow{3}','multirow{2}');(D/('tasks_'+kind+'.tex')).write_text(text)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def summary(k,sc):return next(r for r in S if r['cohort']=='D42' and r['scope']=='common' and r['scenario']==sc and r['method']==k)
def points(sc):return [r for r in E if r['kind']=='batch4' and r['seed']=='290117' and r['scenario']==sc and r['common']=='True' and (int(r['cls']),r['src'],r['tgt']) in keys]
def direction(src,tgt,sc):
 if sc.startswith('region'):return ('GBA' if src.startswith('dwq') else 'XJ')+' → '+('GBA' if tgt.startswith('dwq') else 'XJ')
 return ('S2' if 'sentinel2' in src else 'L8')+' → '+('S2' if 'sentinel2' in tgt else 'L8')
def draw(ax,k,sc,compact=False):
 q=points(sc);tasks=sorted({(int(r['cls']),r['src'],r['tgt']) for r in q})
 for i,key in enumerate(tasks):
  cls,src,tgt=key;t=[r for r in q if (int(r['cls']),r['src'],r['tgt'])==key];x=[float(r[k])*(1 if k in BAD else -1) for r in t];y=[float(r['true_AD']) for r in t];forward=(src.startswith('dwq') if sc.startswith('region') else 'sentinel2' in src);label=classes[cls]+' | '+direction(src,tgt,sc);ax.scatter(x,y,s=13,alpha=.68,marker='o' if forward else '^',color=plt.get_cmap('tab20')(i),label=label)
 s=summary(k,sc);ax.text(.98,.98,f"Fisher r={float(s['Fisher_r']):.3f}\nmean rho={float(s['rho']):.3f}\nn={len(q)}",transform=ax.transAxes,ha='right',va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'));ax.set_xlabel('AD-oriented metric');ax.set_ylabel('Actual F1 drop');ax.grid(alpha=.15)
 return tasks
for sc,title in zip(scens,titles):
 fig,axs=plt.subplots(4,4,figsize=(17,15))
 for i,(ax,k) in enumerate(zip(axs.flat,METHODS)):
  tasks=draw(ax,k,sc);ax.set_title('('+chr(97+i)+') '+LABEL[k].replace('OTCE-PL','OTCE-Q1024-PL'),fontsize=11);audit.append(dict(scenario=sc,method=k,points=len(points(sc)),legend_entries=[classes[c]+' | '+direction(s,t,sc) for c,s,t in tasks],directions_separately_encoded=True,layout='4x4'))
 hs,ls=axs.flat[0].get_legend_handles_labels();fig.legend(hs,ls,ncol=4,loc='lower center',fontsize=11);fig.suptitle(title+' | Full2048 | all 16 methods',fontsize=16);fig.tight_layout(rect=(0,.075,1,.97));stem='all_'+sc+'_4x4'
 for ext in ['png','pdf']:fig.savefig(F/(stem+'.'+ext),dpi=180,bbox_inches='tight');shutil.copy2(F/(stem+'.'+ext),PF/(stem+'.'+ext))
 plt.close(fig)
# Ablation keeps the scientifically meaningful three columns; encode direction
# by marker and include the exact row-specific direction key in each row title.
for family in ['Soft','Hard']:
 fig,axs=plt.subplots(4,3,figsize=(13,14))
 for i,(sc,title) in enumerate(zip(scens,titles)):
  for j,k in enumerate(['FCDTM_'+family+'Signed','FCDTM_'+family+'Abs','FCDTM_'+family+'PL']):
   q=points(sc)
   for cls in sorted({int(r['cls']) for r in q}):
    for forward in [True,False]:
     t=[r for r in q if int(r['cls'])==cls and (r['src'].startswith('dwq') if sc.startswith('region') else 'sentinel2' in r['src'])==forward]
     if not t:continue
     axs[i,j].scatter([float(r[k]) for r in t],[float(r['true_AD']) for r in t],s=14,alpha=.65,color=plt.get_cmap('tab10')(cls-1),marker='o' if forward else '^',label=classes[cls] if forward else None)
   s=summary(k,sc);axs[i,j].text(.98,.98,f"Fisher r={float(s['Fisher_r']):.3f}\nmean rho={float(s['rho']):.3f}",transform=axs[i,j].transAxes,ha='right',va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'));dirs='● GBA → XJ; ▲ XJ → GBA' if sc.startswith('region') else '● S2 → L8; ▲ L8 → S2';axs[i,j].set_title(title+' | '+['Signed','Abs','Weighted'][j]+'\n'+dirs,fontsize=9);axs[i,j].set_xlabel('AD-oriented metric');axs[i,j].set_ylabel('Actual F1 drop');axs[i,j].grid(alpha=.15)
 handles={}
 for ax in axs.flat:
  hs,ls=ax.get_legend_handles_labels()
  handles.update(zip(ls,hs))
 fig.legend(list(handles.values()),list(handles),ncol=7,loc='lower center');fig.tight_layout(rect=(0,.03,1,1))
 for ext in ['png','pdf']:fig.savefig(F/('ablation_'+family.lower()+'.'+ext),dpi=180,bbox_inches='tight');shutil.copy2(F/('ablation_'+family.lower()+'.'+ext),PF/('ablation_'+family.lower()+'.'+ext))
 plt.close(fig)
q=P/'main.tex';s=q.read_text()
if '\\usepackage{rotating}' not in s:s=s.replace('\\usepackage{multirow}','\\usepackage{multirow}\n\\usepackage{rotating}')
for sc,title in zip(scens,titles):
 lines=[l for l in s.splitlines() if 'Figs/U8/all_'+sc+'_part' in l]
 new=r'\begin{figure*}[!t]\centering\includegraphics[width=.99\textwidth,height=.90\textheight,keepaspectratio]{Figs/U8/all_'+sc+r'_4x4.pdf}\caption{'+title+r' comparison of all 16 Full2048 methods in a four-by-four layout. Each legend entry identifies a class and its transfer direction; opposite directions have different markers and colors. Points use the first fixed seed and the same finite records. Upper-right annotations give table-matched five-seed within-task summaries and the displayed point count.}\label{fig:'+sc+r'}\end{figure*}'
 for i,line in enumerate(lines):s=s.replace(line,new if i==0 else '')
s=s.replace('Pearson and Spearman panels are separated for readability. Bold marks the largest unrounded value across methods in each row, including ties;','Pearson and Spearman are adjacent under each two-line method heading. Rotated tables preserve width. Bold marks the largest unrounded value across methods for each task and coefficient; underlining marks the second distinct largest value, including ties;')
s=s.replace('each show all 16 formal methods over two continued pages','each show all 16 formal methods in a four-by-four layout')
s=s.replace('Colors denote foreground classes;','Colors and markers distinguish class--direction tasks;')
s=s.replace('The hard-pseudo-label counterpart is supplied in the supplement.','Markers distinguish GBA-to-XJ from XJ-to-GBA in the cross-region rows and S2-to-L8 from L8-to-S2 in the cross-sensor rows. The hard-pseudo-label counterpart is supplied in the supplement.')
s=s.replace('\\input{data/U8/ablation_region.tex}','\\clearpage\n\\input{data/U8/ablation_region.tex}')
s=s.replace('\\clearpage\n\\clearpage','\\clearpage')
q.write_text(s)
(B/'artifacts/layout_review_audit.json').write_text(json.dumps(dict(figure_panels=audit,table_cells=taudit,best_and_second_use_unrounded_values=True),indent=2)+'\n')
print('Layout review applied: paired columns, best/runner-up, 4x4 plots, directional keys.')
