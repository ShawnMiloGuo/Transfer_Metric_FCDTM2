import csv,json,sys
from pathlib import Path
import numpy as np
BR=Path(__file__).resolve().parents[2];sys.path.insert(0,str(BR/'U0_标签隔离与无标签缓存/code'));import common as c
BASE=Path(__file__).resolve().parents[1];OUT=BASE/'artifacts'
def main():
    from osgeo import gdal
    gdal.UseExceptions();OUT.mkdir(exist_ok=True);m=c.check();assert json.loads((c.OUT/'最终核验.json').read_text())['status']=='passed'
    for p,h in json.loads((c.OUT/'seal.json').read_text()).items():assert c.sha(c.BR/p)==h,p
    ys={d:np.array([gdal.Open(str(c.DATA/d/'train_val/annotations'/r['split']/r['name'])).ReadAsArray() for r in pool]) for d,pool in m['pools'].items()};rows=[];sources=[]
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt']
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z:
            for i,name in enumerate(z['names']):
                pred=np.unpackbits(z['bits'][i],count=65536).reshape(256,256).astype(bool);fg=ys[tgt][i]==cls;cc=np.array([(pred&fg).mean(),(pred&~fg).mean(),(~pred&fg).mean()]);tf=c.f1(cc);confidence=float(z['confidence'][i]);r=dict(cls=cls,src=src,tgt=tgt,name=str(name),true_F1=tf,true_foreground=float(fg.mean()),predicted_foreground=float(pred.mean()),confidence=confidence,hard_empty=bool(not pred.any()),true_empty=bool(not fg.any()),high_conf_low_F1=bool(confidence>=.9 and np.isfinite(tf) and tf<=.2))
                for method in ['weighted','uniform','margin','ridge6','ridge2']:
                    counts=z['risk_'+method][i];r[method+'_F1']=c.f1(counts);r[method+'_error']=abs(c.f1(counts)-tf)
                    for k,label in enumerate(['TP','FP','FN']):r[method+'_'+label+'_bias']=float(counts[k]-cc[k])
                rows.append(r)
    for t in m['models']:
        source=json.loads((c.OUT/f"source/{t['cls']}_{t['src']}.json").read_text())
        for r in source['validation']:sources.append(dict(cls=t['cls'],src=t['src'],**r))
    c.save(OUT/'逐图诊断.csv',rows);c.save(OUT/'源验证.csv',sources)
    finite=[r for r in rows if np.isfinite(r['true_F1'])];high=[r for r in finite if r['high_conf_low_F1']]
    report=dict(images=len(rows),defined_F1=len(finite),undefined_F1=len(rows)-len(finite),hard_empty=sum(r['hard_empty'] for r in rows),true_empty=sum(r['true_empty'] for r in rows),high_conf_low_F1=len(high),high_conf_low_F1_with_true_foreground=sum(not r['true_empty'] for r in high),diagnosis_only=True)
    (OUT/'诊断摘要.json').write_text(json.dumps(report,ensure_ascii=False,indent=1))
    lines=['# U3 首轮伪标签与概率失配诊断','','这是封存后的机制诊断，不是新方法测试或适用条件确认。完整的增强一致性/伪标签筛选尚未执行。','',json.dumps(report,ensure_ascii=False,indent=1),'','逐图数共用模型/影像，不能当独立样本显著性证据。F1未定义不补0；高置信低F1既可来自类别缺失也可来自真实前景错误，两种计数分开。源验证Brier好也不能保证目标条件关系稳定。','', '![置信度与真实F1](artifacts/confidence_F1.png)']
    (BASE/'实验结果分析.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5));present=[r for r in finite if not r['true_empty']];absent=[r for r in finite if r['true_empty']]
    for rs,label in [(absent,'Class absent'),(present,'Class present')]:ax.scatter([r['confidence'] for r in rs],[r['true_F1'] for r in rs],s=8,alpha=.25,label=label)
    ax.set(xlabel='Mean max-softmax confidence',ylabel='True image F1');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/('confidence_F1.'+ext),dpi=180)
    plt.close(fig);print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
