# -*- coding: utf-8 -*-
from pathlib import Path
import sys,json,csv,shutil,re,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'code'));from common import METHODS,LABEL,BAD,OLD
P=Path(sys.argv[1]);F=B/'figures';PF=P/'Figs/U8';read=lambda p:list(csv.DictReader(p.open()));E=read(B/'artifacts/evaluated.csv');S=read(B/'artifacts/summary.csv');m=json.loads((OLD/'artifacts/manifest_full.json').read_text());keys={tuple(x) for x in m['cohorts']['D42']};scens=['region_S2','region_L8','sensor_GBA','sensor_XJ'];titles=['Cross-region Sentinel-2','Cross-region Landsat-8','Cross-sensor GBA','Cross-sensor XJ'];labels=['','Cropland','Forest','Grassland','Shrubland','Wetland','Water','Built-up','Bareland'];audit=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
for scen,title in zip(scens,titles):
 rr=[r for r in E if r['kind']=='batch4' and r['seed']=='290117' and r['scenario']==scen and (int(r['cls']),r['src'],r['tgt']) in keys];q=[r for r in rr if r['common']=='True'];assert len(q)==dict(region_S2=117,region_L8=79,sensor_GBA=100,sensor_XJ=104)[scen]
 for part in [0,1]:
  fig,axs=plt.subplots(4,2,figsize=(10,12));ks=METHODS[part*8:part*8+8]
  for j,(ax,k) in enumerate(zip(axs.flat,ks)):
   for cls in sorted({int(r['cls']) for r in q}):
    t=[r for r in q if int(r['cls'])==cls];x=[float(r[k])*(1 if k in BAD else -1) for r in t];y=[float(r['true_AD']) for r in t];ax.scatter(x,y,s=12,alpha=.6,label=labels[cls],color=plt.get_cmap('tab10')(cls-1))
   s=next(s for s in S if s['cohort']=='D42' and s['scope']=='common' and s['scenario']==scen and s['method']==k);ax.text(.98,.98,f"Fisher r={float(s['Fisher_r']):.3f}\nmean rho={float(s['rho']):.3f}\nn={len(q)}",transform=ax.transAxes,ha='right',va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'));name=LABEL[k].replace('OTCE-PL','OTCE-Q1024-PL');ax.set_title('('+chr(97+part*8+j)+') '+name,fontsize=11);ax.set_xlabel('AD-oriented metric');ax.set_ylabel('Actual F1 drop');ax.grid(alpha=.15);audit.append(dict(scenario=scen,method=k,part=part+1,points=len(q),nominal_points=len(rr),seed=290117,table_Fisher_r=float(s['Fisher_r']),table_rho=float(s['rho'])))
  hs,ls=axs.flat[0].get_legend_handles_labels();fig.legend(hs,ls,ncol=4,loc='lower center',fontsize=9);fig.suptitle(title+' | Full2048 | '+('methods 1–8' if part==0 else 'methods 9–16'),fontsize=13);fig.tight_layout(rect=(0,.045,1,.97));stem='all_'+scen+'_part'+str(part+1)
  for ext in ['png','pdf']:fig.savefig(F/(stem+'.'+ext),dpi=180,bbox_inches='tight');shutil.copy2(F/(stem+'.'+ext),PF/(stem+'.'+ext))
  plt.close(fig)
(B/'artifacts/main_figure_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
# Keep the deterministic manuscript builder aligned with current output.
for path in [P/'main.tex',P/'code/update_u8_manuscript.py']:
 s=path.read_text()
 if path.suffix=='.tex':
  for scen,title in zip(scens,titles):
   pattern=r'\\begin\{figure\*\}\[!t\].*?Figs/U8/batch_'+scen+r'_family1.pdf.*?\\end\{figure\*\}'
   # restrict start to single line (other figures must be preserved)
   line=next((l for l in s.splitlines() if 'Figs/U8/batch_'+scen+'_family1.pdf' in l),None)
   if line is None:continue
   new=''
   for part in [1,2]:
    new+=r'\begin{figure*}[!t]'+(r'\ContinuedFloat' if part==2 else '')+r'\centering\includegraphics[width=.98\textwidth,height=.86\textheight,keepaspectratio]{Figs/U8/all_'+scen+'_part'+str(part)+r'.pdf}\caption{'+title+r' Full2048 comparison of all 16 formal entries'+(' (continued), panels (i)--(p).' if part==2 else ', panels (a)--(h).')+r' Each point is a four-image batch from the first fixed seed. Annotations give table-matched five-seed within-task aggregates and the displayed point count. Colors identify classes; all methods share the same finite records.}'+(r'\label{fig:'+scen+'}' if part==1 else '')+r'\end{figure*}'+'\n'
   s=s.replace(line,new.rstrip())
  s=s.replace('All four scenarios and every formal method appear in the supplementary gallery.','Figures~\\ref{fig:region_S2}--\\ref{fig:sensor_XJ} each show all 16 formal methods over two continued pages; the supplementary gallery retains the method-family views.')
 else:
  a=s.index("for scen,title in zip(['region_S2'");z=s.index("res+=r'''\\subsection{Matched Ablation",a)
  block="for scen,title in zip(['region_S2','region_L8','sensor_GBA','sensor_XJ'],['Cross-region Sentinel-2','Cross-region Landsat-8','Cross-sensor GBA','Cross-sensor XJ']):\n for part in [1,2]:\n  res+=r'\\begin{figure*}[!t]'+(r'\\ContinuedFloat' if part==2 else '')+r'\\centering\\includegraphics[width=.98\\textwidth,height=.86\\textheight,keepaspectratio]{Figs/U8/all_'+scen+'_part'+str(part)+r'.pdf}\\caption{'+title+' Full2048 comparison of all 16 formal entries'+(' (continued), panels (i)--(p).' if part==2 else ', panels (a)--(h).')+r' Four-image batches use the first fixed seed; annotations report table-matched five-seed within-task summaries and point count. Colors identify classes; all methods share finite records.}'+(r'\\label{fig:'+scen+'}' if part==1 else '')+r'\\end{figure*}'+'\\n'\n"
  s=s[:a]+block+s[z:]
 path.write_text(s)
print('64 formal-method panels generated; original plotted point sets preserved.')

q=B/'artifacts/figure_index.json'
index=json.loads(q.read_text())
for sc in scens:
 for part in [1,2]:
  name='all_'+sc+'_part'+str(part)
  if name not in index:index.append(name)
q.write_text(json.dumps(index,indent=2)+'\n')
for name in ['main_figure_audit.json','figure_index.json']:
 shutil.copy2(B/'artifacts'/name,P/'data/U8'/name)
