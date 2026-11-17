"""Exact four-image subdomain correction. Forecast and truth evaluation are separate."""
import json,sys,importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
import common as c
spec=importlib.util.spec_from_file_location('u5eval',c.BASE/'code/evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
OUT=c.OUT/'batch4';MAIN=c.PANEL+['conditional_margin']
def grouping(m):
    result={}
    for di,d in enumerate(c.DOM):
        names=[r['name'] for r in m['pools'][d]];order=np.array(sorted(range(40),key=lambda i:names[i]));order=order[np.random.default_rng(250117+di).permutation(40)];result[d]=order.reshape(10,4).tolist()
    return result
def forecast():
    m=c.check();c.check_seal();OUT.mkdir(exist_ok=True);assert not (OUT/'seal.json').exists();groups=grouping(m)
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,Path)):return
        p=str(args[0])
        if any(s in p for s in ['/annotations/','逐池评价','散点逐图数据','论文式逐任务相关','评价真值','/U3_']):raise PermissionError('Truth/evaluation input blocked in four-image aggregation')
    sys.addaudithook(guard);rows=[]
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt'];source=json.loads((c.OUT/f'source/{cls}_{src}.json').read_text());sf=source['source_F1'];mu=np.array(source['source_true_mu']);w=np.array(source['head_weight'])
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z:
            for k,ix in enumerate(groups[tgt]):
                r=dict(cls=cls,src=src,tgt=tgt,subdomain=k,indices=json.dumps(ix),names=json.dumps(z['names'][ix].tolist()),source_F1=sf,source_only=sf,confidence=float(z['confidence'][ix].mean()))
                for method in c.RISK:
                    if method!='source_only':r[method]=c.f1(z['risk_'+method][ix])
                    r[method+'_AD']=sf-r[method]
                n=z['hard_mass'][ix,1].sum();r['hard']=c.u0.score(z['hard_sum'][ix,1].sum(0)/n,mu,w) if n else np.nan;mass=z['soft_mass'][ix].sum();r['soft']=c.u0.score(z['soft_sum'][ix].sum(0)/mass,mu,w) if mass>1e-8 else np.nan;rows.append(r)
    assert len(rows)==500;c.save(OUT/'predictions.csv',rows);(OUT/'grouping.json').write_text(json.dumps(groups,indent=1));seal=dict(files={p.name:c.sha(p) for p in [OUT/'predictions.csv',OUT/'grouping.json']},code_sha=c.sha(__file__),protocol_sha=c.sha(c.BASE/'论文四图子域补充协议.md'),original_seal_sha=c.sha(c.OUT/'seal.json'),target_truth_read=False)
    (OUT/'seal.json').write_text(json.dumps(seal,indent=1));print('SEALED 500 four-image predictions; no target truth read')
def check():
    c.check();c.check_seal();s=json.loads((OUT/'seal.json').read_text());assert c.sha(__file__)==s['code_sha'] and c.sha(c.BASE/'论文四图子域补充协议.md')==s['protocol_sha'] and c.sha(c.OUT/'seal.json')==s['original_seal_sha']
    for p,h in s['files'].items():assert c.sha(OUT/p)==h,p
def evaluate():
    from osgeo import gdal
    gdal.UseExceptions();check();m=c.check();groups=json.loads((OUT/'grouping.json').read_text());assert groups==grouping(m)
    ys={d:np.array([gdal.Open(str(c.DATA/d/'train_val/annotations'/r['split']/r['name'])).ReadAsArray() for r in pool]) for d,pool in m['pools'].items()};truth={}
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt']
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z:
            pred=np.unpackbits(z['bits'],axis=1,count=65536).reshape(40,256,256).astype(bool);fg=ys[tgt]==cls;counts=np.array([[(p&y).sum(),(p&~y).sum(),(~p&y).sum()] for p,y in zip(pred,fg)])
            for k,ix in enumerate(groups[tgt]):truth[cls,src,tgt,k]=c.f1(counts[ix])
    points=[];tasks=defaultdict(list)
    for r in c.read(OUT/'predictions.csv'):
        for method in c.RISK+c.SCORE+['source_F1']:r[method]=float(r[method])
        key=int(r['cls']),r['src'],r['tgt'];r['true_F1']=truth[key+(int(r['subdomain']),)];r['true_AD']=r['source_F1']-r['true_F1'];r['scenario']=ev.group(key[1],key[2]);points.append(r);tasks[key].append(r)
    c.save(OUT/'散点数据.csv',points);corrs=[];common_tasks=set()
    for key,rs in tasks.items():
        shared=[r for r in rs if np.isfinite(r['true_AD']) and all(np.isfinite(ev.metric(r,k)) for k in MAIN)];values={k:ev.corr(shared,k) for k in MAIN}
        if all(np.isfinite(v[1]) and np.isfinite(v[2]) for v in values.values()):common_tasks.add(key)
        for scope,data in [('own',rs),('common_points',shared)]:
            for method in c.RISK+c.SCORE:
                n,pr,sr=ev.corr(data,method);corrs.append(dict(cls=key[0],src=key[1],tgt=key[2],scenario=ev.group(key[1],key[2]),scope=scope,method=method,points=n,Pearson=pr,Spearman=sr))
    c.save(OUT/'逐任务相关.csv',corrs);summary=[]
    for scope in ['own','common_tasks']:
        for scenario in ['all','region_S2','region_L8','sensor_GBA','sensor_XJ']:
            for method in c.RISK+c.SCORE:
                rs=[r for r in corrs if r['method']==method and r['scope']==('own' if scope=='own' else 'common_points') and (scenario=='all' or r['scenario']==scenario) and (scope=='own' or (r['cls'],r['src'],r['tgt']) in common_tasks)];summary.append(dict(scope=scope,scenario=scenario,method=method,defined_tasks=sum(np.isfinite(r['Pearson']) for r in rs),Pearson=ev.avg(r['Pearson'] for r in rs),Spearman=ev.avg(r['Spearman'] for r in rs)))
    c.save(OUT/'相关汇总.csv',summary);errors=[]
    for method in c.RISK:
        by=defaultdict(list)
        for r in points:
            if np.isfinite(r[method]) and np.isfinite(r['true_F1']):by[int(r['cls']),r['src'],r['tgt']].append(abs(r[method]-r['true_F1']))
        errors.append(dict(method=method,tasks=len(by),MAE=ev.avg(ev.avg(v) for v in by.values())))
    c.save(OUT/'四图AD误差.csv',errors);plot(tasks,summary)
    lines=['# U5 论文四图子域精度验证','','原文N_k=4；每域40图按固定种子分10个互不重叠四图子域，每个固定任务最多10散点，共500记录。真F1和预测F1均先汇总四图计数，不是逐图F1平均。初稿单图分析保留为敏感性；本补充纠正论文口径，并非独立预注册确认，模型和参数未改。','',f'主控制方法共同可定义相关任务{len(common_tasks)}/50；每任务最多K=10，相关不保证统计稳定。源F1来自固定64图。目标标签只在四图预测再封存后的独立评价读取。','', '|场景|方法|共同任务|r|rho|','|---|---|---:|---:|---:|']
    for r in summary:
        if r['scope']=='common_tasks' and r['method'] in MAIN:lines.append(f"|{r['scenario']}|{r['method']}|{r['defined_tasks']}|{ev.fmt(r['Pearson'])}|{ev.fmt(r['Spearman'])}|")
    lines+=['','## 四图AD数值误差','','|方法|任务|MAE|','|---|---:|---:|']
    for r in errors:lines.append(f"|{r['method']}|{r['tasks']}|{ev.fmt(r['MAE'])}|")
    lines+=['','## 论文式主散点图']
    for scenario in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:lines+=['',f'![{scenario}](artifacts/batch4/figures/paper_{scenario}.png)']
    lines+=['','50个单任务图见artifacts/batch4/figures/task_*.png/pdf。相关按共同点和共同任务计算，不以混类pooled相关替代任务内指标。原式hard轴为原生分数，其他为预测AD；负AD保留。常量或不足3点N/A，不记0。','', '预测与分组来自冻结无标签缓存和固定图名随机种子，未读真值参与聚合。论文图式沿用，但本轮不是FD/DS/GBC完整复现，历史池未变独立区域。']
    (c.BASE/'论文四图子域结果分析.md').write_text('\n'.join(lines)+'\n');(OUT/'核验.json').write_text(json.dumps(dict(status='passed',records=500,subdomain_images=4,task_subdomains=10,groups_disjoint_per_domain=True,all_40_images_covered=True,shared_all_models_classes=True,common_defined_tasks=len(common_tasks),predictions_sealed_before_batch4_truth_read=True,code_sha=c.sha(__file__)),indent=1));print('EVALUATED four-image paper protocol; common tasks',len(common_tasks))
def plot(tasks,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    def limits(values):
        a=[float(x) for x in values if np.isfinite(float(x))]
        if not a:return 0,1
        lo,hi=min(a),max(a);pad=max((hi-lo)*.06,.015 if hi-lo<1e-9 else 1e-8);return lo-pad,hi+pad
    def axes_set(ax,rs,method,ylim):
        valid=[r for r in rs if np.isfinite(r['true_AD']) and np.isfinite(ev.metric(r,method))];xlim=limits([ev.metric(r,method) for r in valid])
        if method!='hard':ax.plot(ylim,ylim,'k--',lw=.7,alpha=.4)
        ax.set(xlim=xlim,ylim=ylim,xlabel=method+(' score' if method=='hard' else ' predicted AD'),ylabel='Accuracy drop (AD)')
    def save(fig,name,legend=False):
        fig.tight_layout(rect=(0,.15 if legend else 0,1,.96))
        for ext in ['png','pdf']:fig.savefig(folder/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    for scenario in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:
        selected=[(k,rs) for k,rs in tasks.items() if ev.group(k[1],k[2])==scenario];allr=[r for _,rs in selected for r in rs];ylim=limits([r['true_AD'] for r in allr]);fig,axes=plt.subplots(2,3,figsize=(15,10));handles=[];labels=[]
        for ax,method in zip(axes.ravel(),c.PANEL):
            for i,(k,rs) in enumerate(selected):
                valid=[r for r in rs if np.isfinite(r['true_AD']) and np.isfinite(ev.metric(r,method))];sc=ax.scatter([ev.metric(r,method) for r in valid],[r['true_AD'] for r in valid],s=18,alpha=.75,color=plt.get_cmap('tab20')(i%20))
                if method=='hard':handles.append(sc);labels.append(k[1]+' -> '+k[2]+' '+ev.CLASSES[k[0]-1])
            axes_set(ax,allr,method,ylim);ss=next(r for r in summary if r['scope']=='common_tasks' and r['scenario']==scenario and r['method']==method);ax.set_title(f"mean task r={ev.fmt(ss['Pearson'])}, rho={ev.fmt(ss['Spearman'])}; tasks={ss['defined_tasks']}",fontsize=9)
        fig.suptitle(scenario+' : four-image subdomains (paper protocol)');fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7);save(fig,'paper_'+scenario,True)
    for k,rs in tasks.items():
        fig,axes=plt.subplots(2,3,figsize=(13,8));ylim=limits([r['true_AD'] for r in rs])
        for ax,method in zip(axes.ravel(),c.PANEL):
            good=[r for r in rs if np.isfinite(r['true_AD']) and np.isfinite(ev.metric(r,method))];ax.scatter([ev.metric(r,method) for r in good],[r['true_AD'] for r in good],s=24);axes_set(ax,rs,method,ylim);n,pr,sr=ev.corr(rs,method);ax.set_title(f'K={n}, r={ev.fmt(pr)}, rho={ev.fmt(sr)}',fontsize=9)
        fig.suptitle(f'{ev.CLASSES[k[0]-1]}: {k[1]} -> {k[2]} (4 images per point)');save(fig,f'task_{k[0]}_{k[1]}__{k[2]}')
if __name__=='__main__':forecast() if sys.argv[1]=='forecast' else evaluate()
