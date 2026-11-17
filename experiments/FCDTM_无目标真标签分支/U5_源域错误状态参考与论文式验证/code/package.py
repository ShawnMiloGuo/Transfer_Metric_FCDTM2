"""Post-evaluation integrity, matched control correlations and paper figure layout."""
import json,importlib.util
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
import common as c
spec=importlib.util.spec_from_file_location('u5evaluation',c.BASE/'code/evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
def main():
    c.check();c.check_seal();assert json.loads((c.OUT/'最终核验.json').read_text())['status']=='passed'
    rows=c.read(c.OUT/'散点逐图数据.csv');points=[];tasks=defaultdict(list)
    for r in rows:
        for method in c.RISK+c.SCORE+['true_F1','true_AD','source_F1']:r[method]=float(r[method])
        assert np.isclose(r['true_AD'],r['source_F1']-r['true_F1'],equal_nan=True,atol=1e-12)
        points.append(r);tasks[int(r['cls']),r['src'],r['tgt']].append(r)
    assert len(points)==2000 and len(tasks)==50
    # Independent recomputation from saved point records, including ties.
    checks=0
    for r in c.read(c.OUT/'论文式逐任务相关.csv'):
        key=int(r['cls']),r['src'],r['tgt'];rs=tasks[key]
        if r['scope']=='panel_common':rs=[x for x in rs if np.isfinite(x['true_AD']) and all(np.isfinite(ev.metric(x,k)) for k in c.PANEL)]
        good=[x for x in rs if np.isfinite(x['true_AD']) and np.isfinite(ev.metric(x,r['method']))];x=np.array([ev.metric(v,r['method']) for v in good]);y=np.array([v['true_AD'] for v in good]);assert len(x)==int(r['points'])
        if len(x)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12:
            pr=np.corrcoef(x,y)[0,1];sr=np.corrcoef(rankdata(x),rankdata(y))[0,1];assert np.isclose(pr,float(r['Pearson']),atol=1e-10) and np.isclose(sr,float(r['Spearman']),atol=1e-10)
        else:assert np.isnan(float(r['Pearson'])) and np.isnan(float(r['Spearman']))
        checks+=1
    controls=['error_weighted','error_uniform','conditional_margin','global_margin'];cr=[];valid=[]
    for key,rs in tasks.items():
        part=[r for r in rs if np.isfinite(r['true_AD']) and all(np.isfinite(ev.metric(r,method)) for method in controls)];computed={method:ev.corr(part,method) for method in controls};defined=all(np.isfinite(v[1]) and np.isfinite(v[2]) for v in computed.values())
        if defined:valid.append(key)
        for method,(n,pr,sr) in computed.items():cr.append(dict(cls=key[0],src=key[1],tgt=key[2],scenario=ev.group(key[1],key[2]),method=method,points=n,Pearson=pr,Spearman=sr,all_controls_defined=defined))
    c.save(c.OUT/'条件拆分控制相关.csv',cr);cs=[]
    for scenario in ['all','region_S2','region_L8','sensor_GBA','sensor_XJ']:
        for method in controls:
            rs=[r for r in cr if r['method']==method and r['all_controls_defined'] and (scenario=='all' or r['scenario']==scenario)];cs.append(dict(scenario=scenario,method=method,tasks=len(rs),Pearson=ev.avg(r['Pearson'] for r in rs),Spearman=ev.avg(r['Spearman'] for r in rs)))
    c.save(c.OUT/'条件拆分公平相关汇总.csv',cs)
    batch=c.OUT/'batch4';bs=json.loads((batch/'seal.json').read_text());assert c.sha(c.BASE/'code/batch4.py')==bs['code_sha'] and c.sha(c.BASE/'论文四图子域补充协议.md')==bs['protocol_sha']
    for p,h in bs['files'].items():assert c.sha(batch/p)==h,p
    grouping=json.loads((batch/'grouping.json').read_text())
    for groups in grouping.values():assert len(groups)==10 and all(len(g)==4 for g in groups) and sorted(i for g in groups for i in g)==list(range(40))
    bp=c.read(batch/'散点数据.csv');bt=defaultdict(list)
    for r in bp:
        for method in c.RISK+c.SCORE+['source_F1','true_F1','true_AD']:r[method]=float(r[method])
        assert np.isclose(r['true_AD'],r['source_F1']-r['true_F1'],equal_nan=True,atol=1e-12);bt[int(r['cls']),r['src'],r['tgt']].append(r)
    assert len(bp)==500
    for r in c.read(batch/'逐任务相关.csv'):
        data=bt[int(r['cls']),r['src'],r['tgt']]
        if r['scope']=='common_points':data=[x for x in data if np.isfinite(x['true_AD']) and all(np.isfinite(ev.metric(x,k)) for k in c.PANEL+['conditional_margin'])]
        good=[x for x in data if np.isfinite(x['true_AD']) and np.isfinite(ev.metric(x,r['method']))];x=np.array([ev.metric(v,r['method']) for v in good]);y=np.array([v['true_AD'] for v in good]);assert len(x)==int(r['points'])
        if len(x)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12:
            assert np.isclose(np.corrcoef(x,y)[0,1],float(r['Pearson']),atol=1e-10) and np.isclose(np.corrcoef(rankdata(x),rankdata(y))[0,1],float(r['Spearman']),atol=1e-10)
        else:assert np.isnan(float(r['Pearson'])) and np.isnan(float(r['Spearman']))
    # Axis-only presentation: show actual AD range rather than let a long 1:1 line expand it.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=c.OUT/'figures'
    def span(values):
        a=[float(v) for v in values if np.isfinite(float(v))]
        if not a:return (0,1)
        lo,hi=min(a),max(a);pad=max((hi-lo)*.06,.015 if hi-lo<1e-9 else 1e-8);return lo-pad,hi+pad
    def axis(ax,rs,method,ylim):
        good=[r for r in rs if np.isfinite(r['true_AD']) and np.isfinite(ev.metric(r,method))];xx=[ev.metric(r,method) for r in good];xlim=span(xx)
        if method!='hard':ax.plot(ylim,ylim,'k--',lw=.7,alpha=.35)
        ax.set(xlim=xlim,ylim=ylim,xlabel=method+(' score' if method=='hard' else ' predicted AD'),ylabel='True accuracy drop (AD)')
    def save(fig,name,legend=False):
        fig.tight_layout(rect=(0,.15 if legend else 0,1,.96))
        for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    correlations=c.read(c.OUT/'相关汇总.csv')
    for scenario in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:
        selected=[(k,rs) for k,rs in tasks.items() if ev.group(k[1],k[2])==scenario];allpoints=[r for _,rs in selected for r in rs];ylim=span([r['true_AD'] for r in allpoints]);fig,axes=plt.subplots(2,3,figsize=(15,10));handles=[];labels=[]
        for ax,method in zip(axes.ravel(),c.PANEL):
            for i,(key,rs) in enumerate(selected):
                good=[r for r in rs if np.isfinite(r['true_AD']) and np.isfinite(ev.metric(r,method))];sc=ax.scatter([ev.metric(r,method) for r in good],[r['true_AD'] for r in good],s=12,alpha=.6,color=plt.get_cmap('tab20')(i%20))
                if method=='hard':handles.append(sc);labels.append(key[1]+' -> '+key[2]+' '+ev.CLASSES[key[0]-1])
            axis(ax,allpoints,method,ylim);ss=next(r for r in correlations if r['scope']=='common_tasks' and r['scenario']==scenario and r['method']==method);ax.set_title(f"mean task r={ev.fmt(ss['Pearson'])}, rho={ev.fmt(ss['Spearman'])}; tasks={ss['defined_tasks']}",fontsize=9)
        fig.suptitle(scenario+' : fixed-source-model subdomains');fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=7);save(fig,'paper_'+scenario,True)
    for key,rs in tasks.items():
        ylim=span([r['true_AD'] for r in rs]);fig,axes=plt.subplots(2,3,figsize=(13,8))
        for ax,method in zip(axes.ravel(),c.PANEL):
            good=[r for r in rs if np.isfinite(r['true_AD']) and np.isfinite(ev.metric(r,method))];ax.scatter([ev.metric(r,method) for r in good],[r['true_AD'] for r in good],s=14,alpha=.65);axis(ax,rs,method,ylim);n,pr,sr=ev.corr(rs,method);ax.set_title(f'n={n}, r={ev.fmt(pr)}, rho={ev.fmt(sr)}',fontsize=9)
        fig.suptitle(f'{ev.CLASSES[key[0]-1]}: {key[1]} -> {key[2]}');save(fig,f'task_{key[0]}_{key[1]}__{key[2]}')
    ad={r['method']:r for r in c.read(c.OUT/'AD误差.csv') if r['budget']=='0'};sel={r['method']:r for r in c.read(c.OUT/'选择汇总.csv') if r['budget']=='0'}
    report=c.BASE/'实验结果分析.md';text=report.read_text()
    if text.startswith('> 论文主验证使用每子域4图'):text=text.split('\n\n',1)[1]
    marker='\n## 首轮结果判断与核验补充\n';text=text.split(marker)[0];text+=marker+'\n'
    for control in ['error_uniform','conditional_margin','global_margin','ridge6']:
        text+=f"- error_weighted相对{control}：MAE变化（正=退化）{float(ad['error_weighted']['MAE'])-float(ad[control]['MAE']):.6f}；选择遗憾变化{float(sel['error_weighted']['regret'])-float(sel[control]['regret']):.6f}。\n"
    text+='\n### 与条件margin的同任务公平相关\n\n|场景|方法|共同任务|r|rho|\n|---|---|---:|---:|---:|\n'
    for r in cs:text+=f"|{r['scenario']}|{r['method']}|{r['tasks']}|{ev.fmt(r['Pearson'])}|{ev.fmt(r['Spearman'])}|\n"
    text+='\n这张补充表对四种误差风险控制使用共同有限点和共同可定义任务；主六面板表另含原式hard，因空伪前景/常量不同，交集可能更小。不得跨不同交集的均值直接宣称增益。高r/rho不保证AD校准或低选择遗憾。\n\n图形仅重绘轴域使各面板显示相同真实AD范围；原预测、散点、r/rho均不变，1:1线若在可见区域之外不强制扩轴。原图代码与独立重绘代码都保留。源64图曾用于历史选优，源fit/cal/test只是度量器内部划分，不是分割模型的全新OOF源验证。\n\n误差概率是P(真实背景|预测前景)和P(真实前景|预测背景)，分别对应预测侧错误后验，不等于以真实类别为分母的FPR/FNR。此区别不影响通过概率构造TP/FP/FN，但应避免混称。\n'
    report.write_text(text)
    primary=c.BASE/'论文四图子域结果分析.md';ptext=primary.read_text();mark='\n## 与原假设的对照判断\n';ptext=ptext.split(mark)[0];bmae={r['method']:r for r in c.read(batch/'四图AD误差.csv')};bcorr={r['method']:r for r in c.read(batch/'相关汇总.csv') if r['scope']=='common_tasks' and r['scenario']=='all'}
    ptext+=mark+'\n|方法|共同任务r|共同任务rho|四图AD MAE|\n|---|---:|---:|---:|\n'
    for method in ['hard','error_weighted','error_uniform','conditional_margin','global_margin','global_weighted','ridge6']:
        rr=bcorr[method];ptext+=f"|{method}|{ev.fmt(rr['Pearson'])}|{ev.fmt(rr['Spearman'])}|{ev.fmt(bmae[method]['MAE']) if method in bmae else '无AD单位'}|\n"
    ptext+='\n源错误状态加权相对条件margin的误差下降，不等于超过原U2或证明头权重必需。对照同容量无权重、全局margin和ridge6后综合判断，不能以单图更高的相关替代原论文四图口径。当前缺乏稳定超过强基线的整体证据，混合比例估计阶段不在本次运行范围。\n\n四图分组和概率汇总已再次封存；相关统计独立重算通过，所有模型共用同一固定分组。先前单图程序自动完成后才发现N_k=4的核对遗漏，补充协议透明记录时序；没有改变任何模型或概率参数，不声称独立盲法或完全预注册确认。\n'
    primary.write_text(ptext)
    report.write_text('> 论文主验证使用每子域4图，见[四图结果分析](论文四图子域结果分析.md)。本文件保留单图敏感性及8/16/24图池选择结果。\n\n'+report.read_text())
    info=dict(points=2000,undefined_true_F1=sum(not np.isfinite(r['true_F1']) for r in points),negative_AD_points=sum(np.isfinite(r['true_AD']) and r['true_AD']<0 for r in points),correlation_rows_recomputed=checks,four_controls_common_tasks=len(valid),forecasts_unchanged=True,all_values_same_as_frozen_predictor=True)
    (c.OUT/'散点与相关复算核验.json').write_text(json.dumps(info,ensure_ascii=False,indent=1,default=c.u0.serialize))
    index=['# U5 图表索引','','论文每子域4图；主验证见[四图结果](论文四图子域结果分析.md)。单图结果仅作补充。原式hard不是AD单位。','','## 四图子域主总览']
    for name in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:index.append(f'- {name}：[PNG](artifacts/batch4/figures/paper_{name}.png) / [PDF](artifacts/batch4/figures/paper_{name}.pdf)')
    index+=['','## 四图具体任务']
    for key in tasks:
        name=f'task_{key[0]}_{key[1]}__{key[2]}';index.append(f'- {ev.CLASSES[key[0]-1]} {key[1]} → {key[2]}：[PNG](artifacts/batch4/figures/{name}.png) / [PDF](artifacts/batch4/figures/{name}.pdf)')
    index+=['','## 单图补充总览']
    for name in ['region_S2','region_L8','sensor_GBA','sensor_XJ']:index.append(f'- {name}：[PNG](artifacts/figures/paper_{name}.png) / [PDF](artifacts/figures/paper_{name}.pdf)')
    index+=['','## 具体任务']
    for key in tasks:
        name=f'task_{key[0]}_{key[1]}__{key[2]}';index.append(f'- {ev.CLASSES[key[0]-1]} {key[1]} → {key[2]}：[PNG](artifacts/figures/{name}.png) / [PDF](artifacts/figures/{name}.pdf)')
    index+=['','[结果分析](实验结果分析.md)；统计CSV位于artifacts，散点与相关已独立复算。'];(c.BASE/'图表索引.md').write_text('\n'.join(index)+'\n')
    (c.BASE/'TODO.md').write_text((c.BASE/'TODO.md').read_text().replace('- [ ]','- [x]'))
    p=c.BR/'TODO.md';text=p.read_text();entry='- [x] U5完成：25模型/50任务，按论文每子域4图的500预测与主散点；单图2000点仅敏感性，800池选模；25访问审计通过，4总览+50任务的PNG/PDF，r/rho及AD误差分别报告。原单图口径核对遗漏及补充时序已明示，未改模型参数。[主结果](U5_源域错误状态参考与论文式验证/论文四图子域结果分析.md)。'
    lines=text.splitlines();hits=[i for i,line in enumerate(lines) if line.startswith('- [') and 'U5' in line]
    if hits:lines[hits[0]]=entry
    else:lines.insert(2,entry)
    p.write_text('\n'.join(lines)+'\n')
    c.check_seal();print(json.dumps(info,ensure_ascii=False,default=c.u0.serialize));print('PACKAGED U5 without forecast changes')
if __name__=='__main__':main()
