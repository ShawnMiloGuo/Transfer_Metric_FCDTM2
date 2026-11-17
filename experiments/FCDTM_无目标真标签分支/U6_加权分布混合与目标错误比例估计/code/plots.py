"""Static scientific plots from sealed predictions and post-seal evaluation."""
import json
import numpy as np
import common as c
import evaluate as ev

def draw(tasks,summary,errors,sels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    out=c.OUT/'figures';out.mkdir(exist_ok=True)
    def save(fig,name,legend=False):
        for ax in fig.axes:
            ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(6));ax.tick_params(labelsize=9)
        fig.tight_layout(rect=(0,.16 if legend else 0,1,.96))
        for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    def limits(vals):
        a=np.array([float(x) for x in vals if np.isfinite(float(x))]);lo,hi=(a.min(),a.max()) if len(a) else (0,1);pad=max((hi-lo)*.06,.01);return lo-pad,hi+pad
    def axis(ax,rs,method,ylim):
        good=[r for r in rs if np.isfinite(ev.metric(r,method)) and np.isfinite(r['true_AD'])];xx=[ev.metric(r,method) for r in good];xl=limits(xx);ax.plot(ylim,ylim,'k--',alpha=.4,lw=.8);ax.set(xlim=xl,ylim=ylim,xlabel=method+' predicted AD',ylabel='Actual AD')
    for scen in ['sensor_GBA','sensor_XJ','region_S2','region_L8']:
        selected=[(k,rs) for k,rs in tasks.items() if c.scenario(k[1],k[2])==scen];allr=[r for k,rs in selected for r in rs];ylim=limits([r['true_AD'] for r in allr]);fig,axes=plt.subplots(2,3,figsize=(15,10));handles=[];labels=[]
        for ax,method in zip(axes.ravel(),c.PANEL):
            for i,(key,rs) in enumerate(selected):
                good=[r for r in rs if r['common_valid']];sc=ax.scatter([ev.metric(r,method) for r in good],[r['true_AD'] for r in good],s=18,alpha=.75,color=plt.get_cmap('tab20')(i%20))
                if method==c.PANEL[0]:handles.append(sc);labels.append(key[1]+' -> '+key[2]+' '+c.CLASSES[key[0]-1])
            axis(ax,[r for r in allr if r['common_valid']],method,ylim);s=next(r for r in summary if r['scope']=='common_tasks' and r['scenario']==scen and r['method']==method);ax.set_title(f"mean task r={s['Pearson']:.3f}, rho={s['Spearman']:.3f}; tasks={s['defined_tasks']}",fontsize=10)
        fig.suptitle('U6 '+scen+' : four-image subdomains');fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7);save(fig,'paper_'+scen,True)
    for key,rs in tasks.items():
        fig,axes=plt.subplots(2,3,figsize=(13,8));ylim=limits([r['true_AD'] for r in rs]);good=[r for r in rs if r['common_valid']]
        for ax,method in zip(axes.ravel(),c.PANEL):
            ax.scatter([ev.metric(r,method) for r in good],[r['true_AD'] for r in good],s=24);axis(ax,good,method,ylim);n,pr,sr=ev.corr(good,method);ax.set_title(f'K={n}, r={pr:.3f}, rho={sr:.3f}',fontsize=10)
        fig.suptitle(f'{c.CLASSES[key[0]-1]}: {key[1]} -> {key[2]} (4 images per point)');save(fig,f'task_{key[0]}_{key[1]}__{key[2]}')
    methods=['U5_weighted','global_margin','global_weighted','mix_uniform','mix_weighted','mix_weighted_raw'];scens=['sensor_GBA','sensor_XJ','region_S2','region_L8'];xx=np.arange(4);fig,axes=plt.subplots(1,2,figsize=(14,5))
    for i,method in enumerate(methods):
        rr=[next(r for r in errors if r['kind']=='batch4' and r['scope']=='common_points' and r['scenario']==s and r['method']==method) for s in scens]
        for ax,col in zip(axes,['MAE','F1_bias']):ax.bar(xx+(i-2.5)*.12,[r[col] for r in rr],width=.12,label=method)
    for ax,col in zip(axes,['MAE','F1 prediction bias']):ax.set(xticks=xx,xticklabels=scens,ylabel=col);ax.axhline(0,color='k',lw=.5)
    axes[0].legend(fontsize=7);fig.suptitle('Four-image AD error and optimism; common finite points');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('error_bias.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method in methods+['ridge6']:
        rr=[next(r for r in sels if r['budget']==b and r['method']==method and r['scope']=='common_configs') for b in [8,16,24]]
        axes[0].plot([8,16,24],[r['regret'] for r in rr],marker='o',label=method);axes[1].plot([8,16,24],[r['Top1'] for r in rr],marker='o',label=method)
    axes[0].set(xlabel='Unlabeled images',ylabel='Selection regret');axes[1].set(xlabel='Unlabeled images',ylabel='Top-1');axes[0].legend(fontsize=7);fig.suptitle('Model selection; identical candidate sets / common eligible configurations');save(fig,'selection')
    fig,ax=plt.subplots(figsize=(9,4))
    for i,mode in enumerate(c.MODES):
        choices=[json.loads(p.read_text())['params'][mode]['lam'] for p in (c.OUT/'source').glob('*.json')];ax.bar(np.arange(len(c.LAMS))+(i-.5)*.32,[choices.count(x) for x in c.LAMS],width=.32,label=mode)
    ax.set(xticks=np.arange(len(c.LAMS)),xticklabels=[str(x) for x in c.LAMS],ylabel='Source models',xlabel='Lambda selected using source calibration only');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/('source_lambda.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)
    records=json.loads((c.OUT/'封存后混合诊断.json').read_text());fig,axes=plt.subplots(2,2,figsize=(10,8))
    for ax,scen in zip(axes.ravel(),scens):
        rr=[r for r in records if r['kind']=='batch4' and r['mode']=='weighted' and r['scenario']==scen]
        for side,label in [(0,'Miss among predicted background'),(1,'False alarm among predicted foreground')]:
            q=[r for r in rr if np.isfinite(r['true_e'+str(side)])];ax.scatter([r['true_e'+str(side)] for r in q],[r['side'+str(side)+'_error_rate'] for r in q],s=12,alpha=.6,label=label)
        ax.plot([0,1],[0,1],'k--',lw=.7);ax.set(title=scen,xlabel='Actual predicted-side error proportion',ylabel='Estimated proportion',xlim=(-.02,1.02),ylim=(-.02,1.02))
    axes[0,0].legend(fontsize=6);fig.suptitle('Post-seal mechanism diagnosis; no target truth used for estimation');save(fig,'error_proportions')
