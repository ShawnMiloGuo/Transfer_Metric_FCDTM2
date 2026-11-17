"""Paper Eq12-17 evaluation, target truth only after U5 forecast seal."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr,spearmanr
import common as c
CLASSES=['Crop','Forest','Grass','Shrub','Wetland','Water','Built','Bare']
def avg(x):
    a=[float(v) for v in x if np.isfinite(float(v))];return float(np.mean(a)) if a else np.nan
def macro(rs,col,select=False):
    groups=defaultdict(list)
    for r in rs:
        if np.isfinite(float(r[col])):
            k=(int(r['cls']),r['tgt']) if select else (int(r['cls']),r['src'],r['tgt'])
            groups[k+(int(r['budget']),)].append(float(r[col]))
    tasks=defaultdict(list)
    for k,v in groups.items():tasks[k[:-1]].append(avg(v))
    return avg(avg(v) for v in tasks.values())
def group(src,tgt):return ('region_S2' if 'sentinel2' in src else 'region_L8') if src.split('_')[0]!=tgt.split('_')[0] else ('sensor_GBA' if src.startswith('dwq') else 'sensor_XJ')
def metric(r,method):return r[method] if method in ['hard','soft'] else 1-r['confidence'] if method=='confidence' else r['source_F1']-r[method]
def corr(rs,method):
    valid=[r for r in rs if np.isfinite(metric(r,method)) and np.isfinite(r['true_AD'])];x=np.array([metric(r,method) for r in valid]);y=np.array([r['true_AD'] for r in valid]);defined=len(x)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12
    return len(x),float(pearsonr(x,y).statistic) if defined else np.nan,float(spearmanr(x,y).statistic) if defined else np.nan
def fmt(x):return f'{float(x):.5f}' if np.isfinite(float(x)) else '—'
def main():
    from osgeo import gdal
    gdal.UseExceptions();m=c.check();c.check_seal();assert json.loads((c.OUT/'最终核验.json').read_text())['status']=='passed';ys={d:np.array([gdal.Open(str(c.DATA/d/'train_val/annotations'/r['split']/r['name'])).ReadAsArray() for r in pool]) for d,pool in m['pools'].items()};points=[];task_points={};truth={}
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt'];sf=json.loads((c.OUT/f'source/{cls}_{src}.json').read_text())['source_F1'];rs=[]
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z:
            counts=[]
            for i,name in enumerate(z['names']):
                pred=np.unpackbits(z['bits'][i],count=65536).reshape(256,256).astype(bool);fg=ys[tgt][i]==cls;cc=np.array([(pred&fg).sum(),(pred&~fg).sum(),(~pred&fg).sum()]);counts.append(cc);tf=c.f1(cc);r=dict(cls=cls,src=src,tgt=tgt,scenario=group(src,tgt),name=str(name),source_F1=sf,true_F1=tf,true_AD=sf-tf,confidence=float(z['confidence'][i]),hard=float(z['hard'][i]),soft=float(z['soft'][i]),source_only=sf)
                for method in c.RISK:
                    if method!='source_only':r[method]=c.f1(z['risk_'+method][i])
                rs.append(r);points.append(r)
            for budget,repeat,ix in c.draws(tgt):truth[cls,src,tgt,budget,repeat]=c.f1(np.array(counts)[ix])
        task_points[cls,src,tgt]=rs
    c.save(c.OUT/'散点逐图数据.csv',points);correlations=[]
    for (cls,src,tgt),rs in task_points.items():
        common=[r for r in rs if np.isfinite(r['true_AD']) and all(np.isfinite(metric(r,method)) for method in c.PANEL)]
        for scope,data in [('own',rs),('panel_common',common)]:
            for method in c.RISK+c.SCORE:
                n,pr,sr=corr(data,method);correlations.append(dict(cls=cls,src=src,tgt=tgt,scenario=group(src,tgt),scope=scope,method=method,points=n,Pearson=pr,Spearman=sr))
    c.save(c.OUT/'论文式逐任务相关.csv',correlations);summary=[]
    # Same finite-correlation task set across panel methods.
    validtasks=set.intersection(*[{(r['cls'],r['src'],r['tgt']) for r in correlations if r['scope']=='panel_common' and r['method']==method and np.isfinite(r['Pearson']) and np.isfinite(r['Spearman'])} for method in c.PANEL])
    for scope in ['own','common_tasks']:
        for scenario in ['all','region_S2','region_L8','sensor_GBA','sensor_XJ']:
            for method in c.RISK+c.SCORE:
                rs=[r for r in correlations if r['scope']==('own' if scope=='own' else 'panel_common') and r['method']==method and (scenario=='all' or r['scenario']==scenario) and (scope=='own' or (r['cls'],r['src'],r['tgt']) in validtasks)]
                summary.append(dict(scope=scope,scenario=scenario,method=method,defined_tasks=sum(np.isfinite(r['Pearson']) for r in rs),Pearson=avg(r['Pearson'] for r in rs),Spearman=avg(r['Spearman'] for r in rs)))
    c.save(c.OUT/'相关汇总.csv',summary)
    rows=[]
    for raw in c.read(c.OUT/'predictions.csv'):
        r=dict(raw);k=(int(r['cls']),r['src'],r['tgt'],int(r['budget']),int(r['repeat']));r['true_F1']=truth[k];r['true_AD']=float(r['source_F1'])-r['true_F1']
        for method in c.RISK:r[method]=float(r[method]);r[method+'_error']=abs(r[method]-r['true_F1'])
        for method in c.SCORE:r[method]=float(r[method])
        rows.append(r)
    c.save(c.OUT/'逐池评价.csv',rows);ad=[]
    for budget in [0,8,16,24,40]:
        common=[r for r in rows if (int(r['budget'])<40 if budget==0 else int(r['budget'])==budget) and all(np.isfinite(r[method+'_error']) for method in c.RISK)]
        for method in c.RISK:ad.append(dict(budget=budget,method=method,rows=len(common),MAE=macro(common,method+'_error')))
    c.save(c.OUT/'AD误差.csv',ad);by=defaultdict(list)
    for r in rows:by[int(r['cls']),r['tgt'],int(r['budget']),int(r['repeat'])].append(r)
    choices=[]
    for (cls,tgt,budget,repeat),rs in by.items():
        if len(rs)<2:continue
        y=np.array([r['true_F1'] for r in rs])
        for method in c.RISK+c.SCORE:
            v=np.array([r[method] for r in rs]);v=-v if method in ['hard','soft'] else v;valid=np.isfinite(v).all() and np.isfinite(y).all()
            if valid:
                win=np.isclose(v,v.max(),atol=1e-12,rtol=0);regret=float(y.max()-y[win].mean());top=float(np.isclose(y,y.max(),atol=1e-12,rtol=0)[win].mean())
            else:regret=top=np.nan
            choices.append(dict(cls=cls,tgt=tgt,budget=budget,repeat=repeat,method=method,regret=regret,Top1=top))
    c.save(c.OUT/'模型选择.csv',choices);sels=[]
    for budget in [0,8,16,24,40]:
        for method in c.RISK+c.SCORE:
            rs=[r for r in choices if r['method']==method and (r['budget']<40 if budget==0 else r['budget']==budget) and np.isfinite(r['regret'])];sels.append(dict(budget=budget,method=method,configs=len(rs),regret=macro(rs,'regret',True),Top1=macro(rs,'Top1',True),max_loss=max((r['regret'] for r in rs),default=np.nan)))
    c.save(c.OUT/'选择汇总.csv',sels);plots(task_points,summary);report(summary,ad,sels,validtasks)
    (c.OUT/'evaluation_manifest.json').write_text(json.dumps(dict(script_sha=c.sha(__file__),forecast_seal=c.sha(c.OUT/'seal.json'),target_truth_read_after_seal=True),indent=1));print('EVALUATED U5; 2000 image points / 50 tasks / 800 pools',flush=True)
def plots(task_points,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=c.OUT/'figures';out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout(rect=(0,.12,1,.97))
        for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    for scenario in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:
        selected=[(k,rs) for k,rs in task_points.items() if group(k[1],k[2])==scenario];fig,axes=plt.subplots(2,3,figsize=(15,10));handles=[];labels=[]
        for ax,method in zip(axes.ravel(),c.PANEL):
            for i,(k,rs) in enumerate(selected):
                valid=[r for r in rs if np.isfinite(r['true_AD']) and np.isfinite(metric(r,method))];sc=ax.scatter([metric(r,method) for r in valid],[r['true_AD'] for r in valid],s=12,alpha=.6,color=plt.get_cmap('tab20')(i%20))
                if method==c.PANEL[0]:handles.append(sc);labels.append(k[1]+' -> '+k[2]+' '+CLASSES[k[0]-1])
            ss=next(r for r in summary if r['scope']=='common_tasks' and r['scenario']==scenario and r['method']==method);ax.set(xlabel=method+(' score' if method=='hard' else ' predicted AD'),ylabel='Accuracy drop (AD)',title=f"mean task r={fmt(ss['Pearson'])}, rho={fmt(ss['Spearman'])}; tasks={ss['defined_tasks']}")
            if method!='hard':ax.plot([-1,1],[-1,1],'k--',lw=.7,alpha=.4)
        fig.suptitle(scenario+' : paper-style metric vs AD');fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7);save(fig,'paper_'+scenario)
    for k,rs in task_points.items():
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for ax,method in zip(axes.ravel(),c.PANEL):
            valid=[r for r in rs if np.isfinite(metric(r,method)) and np.isfinite(r['true_AD'])];ax.scatter([metric(r,method) for r in valid],[r['true_AD'] for r in valid],s=14,alpha=.65);n,pr,sr=corr(rs,method);ax.set(xlabel=method+(' score' if method=='hard' else ' predicted AD'),ylabel='True AD',title=f'n={n}, r={fmt(pr)}, rho={fmt(sr)}')
            if method!='hard':ax.plot([-1,1],[-1,1],'k--',lw=.7,alpha=.4)
        fig.suptitle(f'{CLASSES[k[0]-1]}: {k[1]} -> {k[2]}');save(fig,f'task_{k[0]}_{k[1]}__{k[2]}')
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,method in zip(axes.ravel(),c.PANEL):
        xx=[];yy=[]
        for rs in task_points.values():
            valid=[r for r in rs if np.isfinite(metric(r,method)) and np.isfinite(r['true_AD'])]
            if not valid:continue
            x=np.array([metric(r,method) for r in valid]);y=np.array([r['true_AD'] for r in valid]);xx.extend(x-x.mean());yy.extend(y-y.mean())
        pr=float(pearsonr(xx,yy).statistic) if np.std(xx)>1e-12 else np.nan;ax.scatter(xx,yy,s=4,alpha=.25);ax.set(xlabel='Task-centered '+method,ylabel='Task-centered AD',title='Descriptive pooled residual r='+fmt(pr))
    save(fig,'task_centered_diagnostic')
def report(summary,ad,sels,validtasks):
    lines=['# U5 源域错误状态参考：实验结果分析','','## 目的与过程','', '仅源域建立TN/FN/FP/TP特征原型，固定512像素层级收缩，再按预测侧校准误差概率；目标无真实标签，无目标混合比例拟合。冻结25模型/50任务，访问审计通过后封存，独立评价。沿论文main.tex式12–17，AD=F1s−F1t，每个256×256图为目标子域；与原论文相同的误差下降量和相关指标，但当前是历史开发池。', '', '## 论文式相关比较','',f'六面板方法使用相同有限点和可定义相关的任务交集：{len(validtasks)}/50任务。常量任务和不足3点记N/A，不填0、取绝对值或看图翻号。Spearman对并列用平均秩。','', '|分组|方法|任务|Pearson r|Spearman rho|','|---|---|---:|---:|---:|']
    for r in summary:
        if r['scope']=='common_tasks' and r['method'] in c.PANEL:lines.append(f"|{r['scenario']}|{r['method']}|{r['defined_tasks']}|{fmt(r['Pearson'])}|{fmt(r['Spearman'])}|")
    lines+=['','## 论文式散点图','', '横轴原式hard为无标签FCDTM分数，其他面板为预测AD；纵轴均为真实AD。颜色表示类别与迁移方向。1:1线仅用于有AD单位的预测。图中相关为任务内相关的等权均值，不能从混类云团形状推断选模价值。']
    for name in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:lines+=['',f'![{name}](artifacts/figures/paper_{name}.png)']
    lines+=['','每个任务的独立六面板图见artifacts/figures/task_*.png/pdf；task_centered_diagnostic为任务去均值诊断，明确为描述性，不代替逐任务相关。','','## AD误差与模型选择（8/16/24预算等权）','','|方法|AD MAE|选择遗憾|Top-1|最大损失|','|---|---:|---:|---:|---:|']
    for method in c.RISK:
        a=next(r for r in ad if r['budget']==0 and r['method']==method);s=next(r for r in sels if r['budget']==0 and r['method']==method);lines.append(f"|{method}|{fmt(a['MAE'])}|{fmt(s['regret'])}|{fmt(s['Top1'])}|{fmt(s['max_loss'])}|")
    lines+=['','各方法有效点、相关可定义任务、子池覆盖、分预算与40图结果见CSV。源原式分数无AD单位，不汇报虚假数值预测误差。目标F1分母0的图不补0，缺失数量单列；候选不可算不缩小候选池。', '', '## 限制与结论要求','', '比较error_weighted、error_uniform、conditional_margin，以区分错误状态距离、预测侧拆分和权重作用；同时必须面对global_margin和ridge6强基线。原型误差样本少会回退，源条件错误分布未必跨域稳定。当前数据不独立，不按2000图像—模型记录或重复预算声称独立显著性。所有方法参数和图形方向固定，评价绘图不反馈校准。目标混合比例阶段本次未执行。']
    (c.BASE/'实验结果分析.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
