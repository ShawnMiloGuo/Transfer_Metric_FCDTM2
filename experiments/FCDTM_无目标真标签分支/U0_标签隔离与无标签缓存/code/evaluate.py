"""Separate truth reader; only runs after native access audits and global seal."""
import json
from collections import defaultdict
import numpy as np
import common as c
def key(r):return (int(r['cls']),r['src'],r['tgt'],int(r['budget']),int(r['repeat']))
def avg(x):
    a=[float(v) for v in x if np.isfinite(float(v))];return float(np.mean(a)) if a else np.nan
def macro(rows,col,select=False):
    by=defaultdict(list)
    for r in rows:
        if np.isfinite(float(r[col])):
            k=(int(r['cls']),r['tgt']) if select else (int(r['cls']),r['src'],r['tgt'])
            by[k+(int(r['budget']),)].append(float(r[col]))
    groups=defaultdict(list)
    for k,v in by.items():groups[k[:-1]].append(avg(v))
    return avg(avg(v) for v in groups.values())
def fmt(x):return f'{float(x):.5f}' if np.isfinite(float(x)) else '—'
def table(headers,rows):return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]
def main():
    from osgeo import gdal
    gdal.UseExceptions();m=c.check();assert json.loads((c.OUT/'最终核验.json').read_text())['status']=='passed'
    for p,h in json.loads((c.OUT/'seal.json').read_text()).items():assert c.sha(c.BR/p)==h,p
    ys={d:np.array([gdal.Open(str(c.DATA/d/'train_val/annotations'/r['split']/r['name'])).ReadAsArray() for r in pool]) for d,pool in m['pools'].items()}
    risk=c.read(c.U2/'artifacts/predictions.csv');scores={key(r):r for r in c.read(c.U1/'artifacts/predictions.csv')};truth={};truthrows=[]
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt']
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z:
            pred=np.unpackbits(z['bits'],axis=1,count=65536).reshape(40,256,256).astype(bool);fg=ys[tgt]==cls;cc=np.array([[(p&y).sum(),(p&~y).sum(),(~p&y).sum()] for p,y in zip(pred,fg)])
            for budget,repeat,ix in c.draws(tgt):
                tf=c.f1(cc[ix]);k=(cls,src,tgt,budget,repeat);truth[k]=tf;truthrows.append(dict(cls=cls,src=src,tgt=tgt,budget=budget,repeat=repeat,target_F1=tf,foreground=int(fg[ix].sum()),predicted_fg=int(pred[ix].sum())))
    c.save(c.OUT/'评价真值.csv',truthrows);allrows=[]
    for r in risk:
        rr=dict(r);rr['true_F1']=truth[key(r)];rr['true_AD']=float(r['source_F1'])-rr['true_F1'];rr['scenario']='sensor' if r['src'].split('_')[0]==r['tgt'].split('_')[0] else 'region'
        for method in c.RISK:rr[method]=float(r[method]);rr[method+'_error']=abs(rr[method]-rr['true_F1'])
        for method in c.SCORE:rr[method]=float(scores[key(r)][method])
        allrows.append(rr)
    c.save(c.U2/'artifacts/逐次评价.csv',allrows);summary=[]
    for group,values in [('budget',[8,16,24,40]),('scenario',['sensor','region']),('cls',range(1,9)),('overall',['overall'])]:
        for value in values:
            rs=[r for r in allrows if (group=='budget' or int(r['budget'])<40) and (group=='overall' or (int(r[group]) if group in ['budget','cls'] else r[group])==value)]
            common=[r for r in rs if all(np.isfinite(r[k+'_error']) for k in c.RISK)]
            for method in c.RISK:summary.append(dict(partition=group,value=value,method=method,own_rows=sum(np.isfinite(r[method+'_error']) for r in rs),common_rows=len(common),MAE=macro(common,method+'_error'),own_MAE=macro(rs,method+'_error')))
    c.save(c.U2/'artifacts/AD汇总.csv',summary)
    groups=defaultdict(list)
    for r in allrows:groups[(int(r['cls']),r['tgt'],int(r['budget']),int(r['repeat']))].append(r)
    choices=[]
    for (cls,tgt,budget,repeat),rs in groups.items():
        if len(rs)<2:continue
        y=np.array([r['true_F1'] for r in rs])
        for method in c.RISK+c.SCORE:
            p=np.array([r[method] for r in rs]);p=-p if method in c.SCORE and method not in ['confidence','neg_entropy'] else p
            valid=np.isfinite(p).all() and np.isfinite(y).all()
            if valid:
                sel=np.isclose(p,p.max(),rtol=0,atol=1e-12);best=np.isclose(y,y.max(),rtol=0,atol=1e-12);selected=y[sel].mean();regret=y.max()-selected;top=best[sel].mean()
            else:selected=regret=top=np.nan
            choices.append(dict(cls=cls,tgt=tgt,budget=budget,repeat=repeat,method=method,candidates=len(rs),selected_F1=selected,regret=regret,Top1=top))
    sels=[]
    for budget in [0,8,16,24,40]:
        part=[r for r in choices if (int(r['budget'])<40 if budget==0 else int(r['budget'])==budget)]
        for method in c.RISK+c.SCORE:
            rs=[r for r in part if r['method']==method and np.isfinite(r['regret'])]
            sels.append(dict(budget=budget,method=method,configs=len(rs),regret=macro(rs,'regret',True),Top1=macro(rs,'Top1',True),max_loss=max((r['regret'] for r in rs),default=np.nan)))
    # Common-pool comparison for every U1 score against source, confidence and margin.
    comparison=c.RISK+c.SCORE;keys={method:{(r['cls'],r['tgt'],r['budget'],r['repeat']) for r in choices if r['method']==method and np.isfinite(r['regret'])} for method in comparison};commonkeys=set.intersection(*keys.values());cs=[]
    for method in comparison:
        rs=[r for r in choices if r['method']==method and int(r['budget'])<40 and (r['cls'],r['tgt'],r['budget'],r['repeat']) in commonkeys];cs.append(dict(method=method,configs=len(rs),regret=macro(rs,'regret',True),Top1=macro(rs,'Top1',True)))
    for folder in [c.U1,c.U2]:c.save(folder/'artifacts/模型选择.csv',choices);c.save(folder/'artifacts/选择汇总.csv',sels);c.save(folder/'artifacts/全部方法共同选择.csv',cs)
    (c.OUT/'evaluation_manifest.json').write_text(json.dumps(dict(script_sha=c.sha(__file__),seal_sha=c.sha(c.OUT/'seal.json'),truth_read_after_global_seal=True),indent=1))
    report(summary,sels,cs,truthrows);plots(summary,sels)
    print('EVALUATED',len(allrows),'paired subpools; truth read only by separate evaluator',flush=True)
def report(summary,sels,common,truth):
    for folder,label,methods in [(c.U1,'U1 硬软伪标签FCDTM',c.SCORE+['source_only','ridge6','margin']),(c.U2,'U2 权重引导双侧源风险',c.RISK+['confidence','neg_entropy'])]:
        lines=['# '+label,'','本轮为历史池上的开发实验；预测没有访问目标真实标签。25模型/50任务，8/16/24张各五次加40张一次，共800子池。8/16/24是无标签图像预算；每次评价同一子池性能，不能与旧少标签外推口径直接比较。','', '## 预测与标签访问','', '25worker的Python/GDAL访问阻断探针及strace系统调用审计均需通过，完整预测封存后独立评价才打开目标标签。仅源域64图标签允许进入代理和原型；未读取旧目标真值缓存。审计不等同于OS强沙箱。源参考曾用于选优，来源不独立。','', '## 选择结果（8/16/24预算等权）','']
        lines+=table(['方法','自身有效配置','遗憾','Top-1','最大损失'],[[r['method'],r['configs'],fmt(r['regret']),fmt(r['Top1']),fmt(r['max_loss'])] for r in sels if r['budget']==0 and r['method'] in methods])
        lines+=['','各法自身覆盖可能不同；全部方法共同候选配置见下表，不能通过删除不可算任务宣称更优。','']+table(['方法','共同配置','遗憾','Top-1'],[[r['method'],r['configs'],fmt(r['regret']),fmt(r['Top1'])] for r in common if r['method'] in methods])
        lines+=['','## 全池40张（一次）','']+table(['方法','有效配置','遗憾','Top-1'],[[r['method'],r['configs'],fmt(r['regret']),fmt(r['Top1'])] for r in sels if r['budget']==40 and r['method'] in methods])
        if folder==c.U2:
            lines+=['','## AD预测MAE','']+table(['分组','值','方法','共同配置','MAE'],[[r['partition'],r['value'],r['method'],r['common_rows'],fmt(r['MAE'])] for r in summary if r['partition'] in ['overall','budget','scenario']])
            lines+=['','![风险估计](artifacts/AD_budget.png)']
        else:lines+=['','U1原生分数没有AD单位，没有用目标标签拟合比例，也未将beta1换算当作AD预测。hard/soft使用真源原型与伪目标分组，matched使用源目标一致预测分组，uniform_hard去掉头方向；它们不是原式可无标签计算的证明。']
        lines+=['','![选择遗憾](artifacts/selection_budget.png)','','## 解释要求','', '必须分别比较权重与无权重、margin-only及置信度，不将仅优于源F1先验当作权重贡献。没有事后调公式、翻转方向、挑类别。空硬前景、相对均值分母非有限、F1分母为零分开记录为缺失，不补0；相关CSV包含覆盖率。单候选组不计选择增益。', '', '8类和两个迁移场景均保留；当前未做新的独立区域确认。标签零访问不保证可识别所有错误，源概率在目标上可能失准，高置信预测也可能错。伪标签不是准确率证据。']
        (folder/'实验结果分析.md').write_text('\n'.join(lines)+'\n')
    (c.BASE/'实验结果分析.md').write_text('# U0 标签隔离审计\n\n25个worker的实际访问审计及退出码见artifacts/execution.json、audit_*.json。源annotation白名单仅为各模型源参考64图；目标标注访问0，旧目标缓存访问0须由最终核验确认。全部U1/U2预测封存后才独立评价。\n\n专用目标缓存只含模型输出、伪分组统计和源代理预测，不含目标真实计数。过程是代码层阻断加strace审计，非OS强沙箱。开发图像来自已公开结果的历史池，不宣称盲法独立验证。\n')
def plots(summary,sels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for folder,methods in [(c.U1,['source_only','confidence','hard','soft','uniform_hard','margin']),(c.U2,c.RISK)]:
        fig,ax=plt.subplots(figsize=(9,4.5))
        for method in methods:ax.plot([8,16,24,40],[next(r['regret'] for r in sels if r['budget']==b and r['method']==method) for b in [8,16,24,40]],'o-',label=method)
        ax.set(xlabel='Unlabeled target images',ylabel='Selection regret',xticks=[8,16,24,40]);ax.legend(fontsize=8);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(folder/('artifacts/selection_budget.'+ext),dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for method in c.RISK:ax.plot([8,16,24,40],[next(r['MAE'] for r in summary if r['partition']=='budget' and r['value']==b and r['method']==method) for b in [8,16,24,40]],'o-',label=method)
    ax.set(xlabel='Unlabeled target images',ylabel='AD MAE',xticks=[8,16,24,40]);ax.legend(fontsize=8);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(c.U2/('artifacts/AD_budget.'+ext),dpi=180)
    plt.close(fig)
if __name__=='__main__':main()
