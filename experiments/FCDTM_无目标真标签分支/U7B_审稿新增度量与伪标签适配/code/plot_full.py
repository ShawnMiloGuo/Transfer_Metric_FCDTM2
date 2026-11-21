"""Publication-style complete comparator scatter sets and vector figure books."""
from pathlib import Path
from collections import defaultdict
import json,csv,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
B=Path(__file__).resolve().parents[1];O=B/'artifacts';F=O/'figures';F.mkdir(exist_ok=True)
assert (O/'evaluation_full.json').exists(),'Plots only follow sealed independent evaluation.'
BAD={'FCDTM_HardPL','FCDTM_SoftPL','FD_global','Entropy','DoC','OTCE_PL_SrcCal','MMD_RBF_global'}
LABEL={'FCDTM_HardPL':'FCDTM-HardPL','FCDTM_SoftPL':'FCDTM-SoftPL','FD_global':'FD','DS_pixel':'DS','GBC_spherical_PL':'GBC-PL','AC':'AC / MSP','Entropy':'Entropy','ATC_MC':'ATC-MC','ATC_NE':'ATC-NE','DoC':'DoC-gap','DoC_Acc_SrcCal':'DoC-Acc-SrcCal','LogME_PL':'LogME-ICML-PL','Hscore_PL':'H-score-PL','LEEP_PL':'LEEP-PL','NCE_Pred_PL':'NCE-Pred-PL','NLEEP_PL':'NLEEP-PL','OTCE_PL_SrcCal':'OTCE-PL-SrcCal','MMD_RBF_global':'MMD-RBF-global'}
FAMILIES={'original':['DS_pixel','FD_global','GBC_spherical_PL','FCDTM_HardPL','FCDTM_SoftPL'],'output':['AC','Entropy','ATC_MC','ATC_NE','FCDTM_SoftPL'],'label_evidence':['LogME_PL','Hscore_PL','LEEP_PL','NCE_Pred_PL','FCDTM_SoftPL'],'confidence_clustering':['DoC','DoC_Acc_SrcCal','NLEEP_PL','FCDTM_SoftPL'],'transport':['OTCE_PL_SrcCal','MMD_RBF_global','FCDTM_SoftPL']}
DOM=['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8'];CL={1:'Cropland',2:'Forest',3:'Grassland',4:'Shrubland',5:'Wetland',6:'Water',7:'Built-up',8:'Bareland'}
def read(p):return list(csv.DictReader(p.open()))
def domain(s):return s.replace('dwq','GBA').replace('xj','XJ').replace('sentinel2','S2').replace('landsat8','L8').replace('_',' ')
def ranks(x):
 o=np.argsort(x,kind='stable');r=np.empty(len(x));i=0
 while i<len(o):
  j=i+1
  while j<len(o) and x[o[j]]==x[o[i]]:j+=1
  r[o[i:j]]=(i+j-1)/2;i=j
 return r

def corr(x,y):
 x=np.asarray(x);y=np.asarray(y)
 if len(x)<4 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan,np.nan
 c=lambda a,b:float(np.corrcoef((a-a.min())/np.ptp(a),(b-b.min())/np.ptp(b))[0,1])
 return c(x,y),c(ranks(x),ranks(y))
def fmt(v):return f'{float(v):.3f}' if np.isfinite(float(v)) else 'N/A'
raw=read(O/'逐项评价_full.csv');summary=read(O/'相关汇总_full.csv');manifest=json.loads((O/'manifest_full.json').read_text());index=[];audit=[]
for r in raw:
 for k in LABEL:r[k]=float(r[k])
 for k in ['cls','seed','budget','repeat']:r[k]=int(r[k])
 r['true_AD']=float(r['true_AD']);r['common']=r['common']=='True'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':14.5,'axes.labelsize':16,'legend.fontsize':14,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
for panel in ['full2048','shared256']:
 for cohort in ['D42','E50']:
  cohort_keys={tuple(k) for k in manifest['cohorts'][cohort]};dest=F/panel/cohort;dest.mkdir(parents=True,exist_ok=True)
  with PdfPages(F/f'{cohort}_{panel}_all_methods.pdf') as book:
   for scenario in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:
    selected=[r for r in raw if r['panel']==panel and r['kind']=='batch4' and r['seed']==290117 and (r['cls'],r['src'],r['tgt']) in cohort_keys and r['scenario']==scenario]
    tasks=sorted({(r['src'],r['tgt'],r['cls']) for r in selected});groups={key:[r for r in selected if (r['src'],r['tgt'],r['cls'])==key and r['common']] for key in tasks};nominal_splits=len(tasks)*5
    for family,keys in FAMILIES.items():
     if panel=='full2048' and family=='transport':continue
     n=len(keys);fig,axes=plt.subplots(3,2,figsize=(12.6,13.2));handles=[];labels=[]
     for j,(ax,method) in enumerate(zip(axes.ravel(),keys)):
      xx=[];yy=[]
      for task,rs in groups.items():
       direction=int(DOM.index(task[0])>DOM.index(task[1]));color=plt.get_cmap('tab20')(2*(task[2]-1)+direction);x=[r[method]*(1 if method in BAD else -1) for r in rs];y=[r['true_AD'] for r in rs];xx.extend(x);yy.extend(y);handle=ax.scatter(x,y,s=27,alpha=.8,color=color)
       if j==0:handles.append(handle);labels.append(f'{domain(task[0])} $\\rightarrow$ {domain(task[1])}  {CL[task[2]]}'+(' [n=0]' if not rs else ''))
      stats=next(r for r in summary if r['panel']==panel and r['cohort']==cohort and r['scenario']==scenario and r['scope']=='common' and r['method']==method);pr,sr=corr(xx,yy);text=r'$r_{\mathrm{F},5}$'+f" = {fmt(stats['Fisher_r'])}; "+r'$\bar\rho_5$'+f" = {fmt(stats['rho'])}\n"+r'$r_{\mathrm{pool}}$'+f' = {fmt(pr)}; '+r'$\rho_{\mathrm{pool}}$'+f' = {fmt(sr)}\n'+f'n = {len(xx)}; defined splits = {stats["finite_correlations"]}/{nominal_splits}'
      ax.text(.985,.985,text,transform=ax.transAxes,ha='right',va='top',size=14.5,linespacing=1.35,bbox=dict(boxstyle='round,pad=.3',facecolor='white',edgecolor='#ccc',linewidth=.6,alpha=.96));label=LABEL[method] if method in BAD else '-'+LABEL[method];ax.set(xlabel=label+' (AD direction)',ylabel='Accuracy Drop');ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5));ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4),useOffset=False)
      if yy:
       span=max(max(yy)-min(yy),.1);ax.set_ylim(min(yy)-.05*span,max(yy)+.42*span)
      else:ax.text(.5,.4,'No common finite points',transform=ax.transAxes,ha='center',color='grey')
      audit.append(dict(panel=panel,cohort=cohort,scenario=scenario,family=family,method=method,Fisher_r=fmt(stats['Fisher_r']),rho=fmt(stats['rho']),shown_pool_r=fmt(pr),shown_pool_rho=fmt(sr),shown_points=len(xx),finite_correlations=int(stats['finite_correlations'])))
     for ax in axes.ravel()[len(keys):]:ax.axis('off')
     axes.ravel()[-1].legend(handles,labels,loc='center left',frameon=True,handletextpad=.4,labelspacing=.35,fontsize=14 if len(labels)<=12 else 11.5)
     scenario_title={'region_S2':'Cross-region / Sentinel-2','region_L8':'Cross-region / Landsat-8','sensor_GBA':'Cross-sensor / GBA','sensor_XJ':'Cross-sensor / XJ'}[scenario];family_title={'original':'Original comparators','output':'Output-based estimators','label_evidence':'Pseudo-label evidence','confidence_clustering':'Confidence and mixture evidence','transport':'Transport and kernel distances'}[family];pixel_title='2048 pixels/image' if panel=='full2048' else '256 pixels/record';fig.suptitle(f'{scenario_title} | {pixel_title}\n{family_title} ({cohort})',fontsize=16,y=.998);fig.tight_layout(rect=(0,0,1,.955),h_pad=1.8,w_pad=1.8)
     name=f'{scenario}_{family}'
     for ext in ['pdf','png']:fig.savefig(dest/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
     book.savefig(fig,bbox_inches='tight',facecolor='white');plt.close(fig);index.append(dict(panel=panel,cohort=cohort,scenario=scenario,family=family,methods=[LABEL[k] for k in keys],file=str((dest/f'{name}.pdf').relative_to(B))))
(F/'figure_style.json').write_text(json.dumps(dict(figsize_inches=[12.6,13.2],annotation_points=14.5,axis_label_points=16,legend_points_D42=14,design='Minimum about 8 pt annotation at IEEE two-column width'),indent=2));(F/'figure_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2));(F/'annotation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));print('PLOTTED',len(index),'complete family/scenario figure pairs and four vector books',flush=True)
