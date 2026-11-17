import numpy as np
import common as c
import evaluate as ev

def draw(tasks,summary,paper_only=False):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import MaxNLocator
 out=c.OUT/'figures';out.mkdir(exist_ok=True)
 def plot(items,name,title,legend=False):
  fig,axes=plt.subplots(2,3,figsize=(15,10));handles=[];labels=[]
  for ax,k in zip(axes.ravel(),c.PANEL):
   for j,(key,rs) in enumerate(items):
    q=[r for r in rs if r['common']];sc=ax.scatter([c.oriented(r,k) for r in q],[r['true_AD'] for r in q],s=18,alpha=.7,color=plt.get_cmap('tab20')(j%20))
    if k==c.PANEL[0]:handles.append(sc);labels.append(str(key[0])+': '+key[1]+' -> '+key[2])
   values=[ev.corr([r for r in rs if r['common']],k) for _,rs in items];ax.set(xlabel=k+' (higher = worse)',ylabel='Actual F1 drop',title=f"mean task r={ev.avg(x[1] for x in values):.3f}; rho={ev.avg(x[2] for x in values):.3f}");ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(6))
  fig.suptitle(title+'; seed '+str(c.SEEDS[0]))
  if legend:fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=6)
  fig.tight_layout(rect=(0,.16 if legend else 0,1,.95))
  for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=150,bbox_inches='tight')
  plt.close(fig)
 first=[(key,rs) for key,rs in tasks.items() if key[3]==c.SEEDS[0]]
 declared={tuple(k) for k in c.check()['cohorts']['D42']}
 for scen in ['sensor_GBA','sensor_XJ','region_S2','region_L8']:plot([(key,rs) for key,rs in first if rs[0]['scenario']==scen and key[:3] in declared],'paper_'+scen,'U7A D42 '+scen,True)
 if paper_only:return
 for key,rs in first:plot([(key,rs)],f'task_{key[0]}_{key[1]}__{key[2]}','U7A class '+str(key[0])+': '+key[1]+' -> '+key[2])
