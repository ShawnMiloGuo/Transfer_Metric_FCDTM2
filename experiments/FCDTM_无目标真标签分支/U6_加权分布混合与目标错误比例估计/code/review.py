"""Independent arithmetic, grouping and correlation checks after evaluation."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
import common as c

def close(x,y):
    if np.isnan(x) and np.isnan(y):return
    assert abs(x-y)<1e-10,(x,y)
def mean(values):
    a=[v for v in values if np.isfinite(v)];return float(np.mean(a)) if a else np.nan

def main():
    c.check();c.check_seal();assert (c.OUT/'评价核验.json').exists();rows=c.read(c.OUT/'逐项评价.csv');lookup={};batches=defaultdict(list)
    for r in rows:
        k=(int(r['cls']),r['src'],r['tgt'],r['kind'],int(r['budget']),int(r['repeat']));lookup[k]=r
        if r['kind']=='batch4':batches[k[:3]].append(r)
        for method in c.RISK:
            close(float(r[method+'_error']),abs(float(r[method])-float(r['true_F1'])));close(float(r[method+'_bias']),float(r[method])-float(r['true_F1']))
    old=c.read(c.U5/'artifacts/batch4/散点数据.csv');checks=0
    for r in old:
        n=lookup[int(r['cls']),r['src'],r['tgt'],'batch4',4,int(r['subdomain'])];assert json.loads(n['indices'])==json.loads(r['indices']);close(float(n['true_F1']),float(r['true_F1']))
        for a,b in [('U5_weighted','error_weighted'),('global_margin','global_margin'),('global_weighted','global_weighted'),('ridge6','ridge6')]:close(float(n[a]),float(r[b]));checks+=1
    for r in c.read(c.OUT/'误差与偏差.csv'):
        items=[q for q in rows if q['kind']==r['kind'] and (int(q['budget'])<40 if int(r['budget'])==0 else int(q['budget'])==int(r['budget'])) and (r['scenario']=='all' or q['scenario']==r['scenario']) and (r['scope']=='own' or q['common_valid']=='True')];method=r['method'];items=[q for q in items if np.isfinite(float(q[method])) and np.isfinite(float(q['true_F1']))];assert len(items)==int(r['valid_rows'])
        for col,fn in [('MAE',abs),('F1_bias',lambda x:x)]:
            gb=defaultdict(list);gt=defaultdict(list)
            for q in items:gb[int(q['cls']),q['src'],q['tgt'],int(q['budget'])].append(fn(float(q[method])-float(q['true_F1'])))
            for k,v in gb.items():gt[k[:-1]].append(mean(v))
            close(mean(mean(v) for v in gt.values()),float(r[col]))
    correlations=c.read(c.OUT/'逐任务相关.csv')
    for r in correlations:
        items=batches[int(r['cls']),r['src'],r['tgt']];items=[q for q in items if r['scope']=='own' or q['common_valid']=='True'];method=r['method']
        def metric(q):return float(q[method]) if method in ['hard','soft'] else 1-float(q['confidence']) if method=='confidence' else float(q['source_F1'])-float(q[method])
        items=[q for q in items if np.isfinite(metric(q)) and np.isfinite(float(q['true_AD']))];assert len(items)==int(r['points']);x=np.array([metric(q) for q in items]);y=np.array([float(q['true_AD']) for q in items]);ok=len(x)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12
        pr=float(np.corrcoef(x,y)[0,1]) if ok else np.nan;sr=float(np.corrcoef(rankdata(x),rankdata(y))[0,1]) if ok else np.nan;close(pr,float(r['Pearson']));close(sr,float(r['Spearman']))
    frozen=json.loads((c.BASE/'评价代码冻结.json').read_text())
    for name,h in frozen.items():assert c.sha(c.BASE/'code'/name)==h,name
    for p in c.OUT.glob('audit_*.json'):assert json.loads(p.read_text())['status']=='passed'
    plots=list((c.OUT/'figures').glob('*.png'));assert len(plots)==58 and len(list((c.OUT/'figures').glob('*.pdf')))==58
    result=dict(status='passed',evaluation_rows=len(rows),U5_baseline_scalars_identical=checks,correlation_rows_recomputed=len(correlations),error_summaries_recomputed=len(c.read(c.OUT/'误差与偏差.csv')),plots_png_pdf_pairs=len(plots),forecast_hashes_intact=True,evaluation_code_unchanged_since_freeze=True)
    (c.OUT/'评价独立复核.json').write_text(json.dumps(result,indent=1));print(json.dumps(result,indent=1))
if __name__=='__main__':main()
