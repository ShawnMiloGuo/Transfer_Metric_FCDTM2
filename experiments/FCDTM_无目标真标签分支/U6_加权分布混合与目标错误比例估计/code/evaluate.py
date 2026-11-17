"""Post-seal evaluation. This is the only U6 module that reads target truth."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr,spearmanr
import common as c

def avg(values):
    a=[float(x) for x in values if np.isfinite(float(x))];return float(np.mean(a)) if a else np.nan

def macro(rows,col,selection=False):
    a=defaultdict(list)
    for r in rows:
        if np.isfinite(float(r[col])):
            key=(int(r['cls']),r['tgt']) if selection else (int(r['cls']),r['src'],r['tgt'])
            a[key+(int(r['budget']),)].append(float(r[col]))
    b=defaultdict(list)
    for k,v in a.items():b[k[:-1]].append(avg(v))
    return avg(avg(v) for v in b.values())

def metric(r,method):return r[method] if method in ['hard','soft'] else 1-r['confidence'] if method=='confidence' else r['source_F1']-r[method]
def corr(rs,method):
    a=[r for r in rs if np.isfinite(r['true_AD']) and np.isfinite(metric(r,method))];x=np.array([metric(r,method) for r in a]);y=np.array([r['true_AD'] for r in a]);good=len(a)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12
    return len(a),float(pearsonr(x,y).statistic) if good else np.nan,float(spearmanr(x,y).statistic) if good else np.nan

def main():
    from osgeo import gdal
    gdal.UseExceptions();m=c.check();c.check_seal();assert json.loads((c.OUT/'独立复算核验.json').read_text())['status']=='passed'
    ys={d:np.array([gdal.Open(str(c.DATA/d/'train_val/annotations'/r['split']/r['name'])).ReadAsArray() for r in pool]) for d,pool in m['pools'].items()};truth={};shifts=[]
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt']
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z,np.load(c.OUT/f'source/{cls}_{src}.npz') as srcz:
            pred=np.unpackbits(z['bits'],axis=1,count=65536).reshape(40,256,256).astype(bool);fg=ys[tgt]==cls
            truth[cls,src,tgt]=np.array([[(p&y).sum(),(p&~y).sum(),(~p&y).sum()] for p,y in zip(pred,fg)])
            samplepred=np.take_along_axis(pred.reshape(40,-1),z['sample_indices'],1);sampletrue=np.take_along_axis(fg.reshape(40,-1),z['sample_indices'],1)
            for mode in c.MODES:
                ref=srcz[mode][srcz['fit']].sum(0);target=sum(c.hist(ids,p,p!=y) for ids,p,y in zip(z['sample_bins_'+mode],samplepred,sampletrue))
                for side in range(2):
                    for error in range(2):
                        n=int(target[side,error,0].sum());ns=int(ref[side,error,0].sum());tv=float(np.abs(c.densities(target)[side,error]-c.densities(ref)[side,error]).sum()/6) if n and ns else np.nan
                        shifts.append(dict(cls=cls,src=src,tgt=tgt,scenario=c.scenario(src,tgt),mode=mode,side=side,error=error,target_support=n,source_support=ns,reference_TV=tv,diagnosis_only=True))
    c.save(c.OUT/'封存后条件分布漂移.csv',shifts);rows=[];taskpoints=defaultdict(list)
    for raw in c.read(c.OUT/'predictions.csv'):
        r=dict(raw);r.update(cls=int(r['cls']),budget=int(r['budget']),repeat=int(r['repeat']));ix=np.array(json.loads(r['indices']));cc=truth[r['cls'],r['src'],r['tgt']][ix].sum(0);total=len(ix)*65536;tp,fp,fn=cc
        for k in c.RISK+c.SCORE+['source_F1']:r[k]=float(r[k])
        r.update(true_F1=c.f1(cc),scenario=c.scenario(r['src'],r['tgt']),true_e1=float(fp/(tp+fp)) if tp+fp else np.nan,true_e0=float(fn/(total-tp-fp)) if total-tp-fp else np.nan)
        r['true_AD']=r['source_F1']-r['true_F1'];r['common_valid']=bool(np.isfinite(r['true_F1']) and all(np.isfinite(r[k]) for k in c.PANEL))
        for method in c.RISK:r[method+'_error']=abs(r[method]-r['true_F1']);r[method+'_bias']=r[method]-r['true_F1']
        rows.append(r)
        if r['kind']=='batch4':taskpoints[r['cls'],r['src'],r['tgt']].append(r)
    c.save(c.OUT/'逐项评价.csv',rows);correlations=[];validtasks=set()
    for key,rs in taskpoints.items():
        shared=[r for r in rs if r['common_valid']]
        if all(np.isfinite(corr(shared,k)[1]) and np.isfinite(corr(shared,k)[2]) for k in c.PANEL):validtasks.add(key)
        for scope,items in [('own',rs),('common_points',shared)]:
            for method in c.RISK+c.SCORE:
                n,pr,sr=corr(items,method);correlations.append(dict(cls=key[0],src=key[1],tgt=key[2],scenario=c.scenario(key[1],key[2]),scope=scope,method=method,points=n,Pearson=pr,Spearman=sr))
    c.save(c.OUT/'逐任务相关.csv',correlations);summary=[]
    for scope in ['own','common_tasks']:
        for scen in ['all','sensor_GBA','sensor_XJ','region_S2','region_L8']:
            for method in c.RISK+c.SCORE:
                q=[r for r in correlations if r['scope']==('own' if scope=='own' else 'common_points') and r['method']==method and (scen=='all' or r['scenario']==scen) and (scope=='own' or (r['cls'],r['src'],r['tgt']) in validtasks)]
                summary.append(dict(scope=scope,scenario=scen,method=method,defined_tasks=sum(np.isfinite(r['Pearson']) for r in q),Pearson=avg(r['Pearson'] for r in q),Spearman=avg(r['Spearman'] for r in q)))
    c.save(c.OUT/'相关汇总.csv',summary);errors=[]
    for kind,budgets in [('batch4',[4]),('pool',[0,8,16,24,40])]:
        for budget in budgets:
            for scen in ['all','sensor_GBA','sensor_XJ','region_S2','region_L8']:
                for scope in ['own','common_points']:
                    raw=[r for r in rows if r['kind']==kind and (r['budget']<40 if budget==0 else r['budget']==budget) and (scen=='all' or r['scenario']==scen)]
                    base=[r for r in raw if scope=='own' or r['common_valid']]
                    for method in c.RISK:
                        q=[r for r in base if np.isfinite(r[method+'_error'])];errors.append(dict(kind=kind,budget=budget,scenario=scen,scope=scope,method=method,total_rows=len(raw),valid_rows=len(q),MAE=macro(q,method+'_error'),F1_bias=macro(q,method+'_bias')))
    c.save(c.OUT/'误差与偏差.csv',errors)
    by=defaultdict(list)
    for r in rows:
        if r['kind']=='pool':by[r['cls'],r['tgt'],r['budget'],r['repeat']].append(r)
    choices=[]
    for (cls,tgt,budget,repeat),rr in by.items():
        if len(rr)<2:continue
        y=np.array([r['true_F1'] for r in rr]);scen='mixed_candidates'
        for method in c.RISK+c.SCORE:
            v=np.array([r[method] for r in rr]);v=-v if method in ['hard','soft'] else v;good=np.isfinite(v).all() and np.isfinite(y).all()
            if good:
                win=np.isclose(v,v.max(),atol=1e-12,rtol=0);regret=float(y.max()-y[win].mean());top=float(np.isclose(y,y.max(),atol=1e-12,rtol=0)[win].mean())
            else:regret=top=np.nan
            choices.append(dict(cls=cls,tgt=tgt,budget=budget,repeat=repeat,method=method,candidates=len(rr),common_eligible=bool(np.isfinite(y).all() and all(np.isfinite(r[k]) for r in rr for k in c.PANEL)),regret=regret,Top1=top))
    c.save(c.OUT/'模型选择.csv',choices);sels=[]
    for budget in [0,8,16,24,40]:
        for scope in ['own','common_configs']:
            for method in c.RISK+c.SCORE:
                q=[r for r in choices if r['method']==method and (r['budget']<40 if budget==0 else r['budget']==budget)]
                good=[r for r in q if np.isfinite(r['regret']) and (scope=='own' or r['common_eligible'])]
                sels.append(dict(budget=budget,scope=scope,method=method,total_configs=len(q),valid_configs=len(good),regret=macro(good,'regret',True),Top1=macro(good,'Top1',True),max_loss=max((r['regret'] for r in good),default=np.nan)))
    c.save(c.OUT/'选择汇总.csv',sels)
    # Per-class paired metrics; task counts differ by class and domain availability.
    classes=[]
    for cls in range(1,9):
        rr=[r for r in rows if r['kind']=='batch4' and r['cls']==cls and r['common_valid']]
        for method in c.RISK:classes.append(dict(cls=cls,class_name=c.CLASSES[cls-1],method=method,points=sum(np.isfinite(r[method+'_error']) for r in rr),MAE=macro(rr,method+'_error'),F1_bias=macro(rr,method+'_bias')))
    c.save(c.OUT/'分类别误差.csv',classes)
    diagnose(rows,shifts)
    import plots
    plots.draw(taskpoints,summary,errors,sels)
    import report
    report.write(summary,errors,sels,validtasks)
    (c.OUT/'评价核验.json').write_text(json.dumps(dict(status='passed',records=len(rows),four_image_records=500,pool_records=800,common_correlation_tasks=len(validtasks),target_truth_read_after_seal=True,seal_sha=c.sha(c.OUT/'seal.json'),script_sha=c.sha(__file__)),indent=1));print('EVALUATED U6; truth read only after sealed replay passed',flush=True)

def diagnose(rows,shifts):
    lookup={(r['cls'],r['src'],r['tgt'],r['kind'],r['budget'],r['repeat']):r for r in rows};records=[]
    for d in json.loads((c.OUT/'diagnostics.json').read_text()):
        r=lookup[d['cls'],d['src'],d['tgt'],d['kind'],d['budget'],d['repeat']];source=json.loads((c.OUT/f"source/{d['cls']}_{d['src']}.json").read_text())
        d.update(scenario=r['scenario'],true_F1=r['true_F1'],predicted_F1=r['mix_'+d['mode']],F1_error=abs(r['mix_'+d['mode']]-r['true_F1']))
        for side in range(2):
            actual=r['true_e'+str(side)];estimate=d['side'+str(side)+'_error_rate'];d['true_e'+str(side)]=actual;d['e'+str(side)+'_absolute_error']=abs(actual-estimate);d['side'+str(side)+'_residual_above_source95']=bool(d['side'+str(side)+'_residual_TV']>source['params'][d['mode']]['source_cal_residual95'][side])
        if d['kind']=='batch4':d['F1_bootstrap_covered']=bool(d['F1_lo']<=r['true_F1']<=d['F1_hi']) if np.isfinite(r['true_F1']) else None
        records.append(d)
    (c.OUT/'封存后混合诊断.json').write_text(json.dumps(records,default=c.u0.serialize,indent=1));summary=[]
    for scen in ['all','sensor_GBA','sensor_XJ','region_S2','region_L8']:
        for mode in c.MODES:
            q=[r for r in records if r['kind']=='batch4' and r['mode']==mode and (scen=='all' or r['scenario']==scen)];drift=[r['reference_TV'] for r in shifts if r['mode']==mode and (scen=='all' or r['scenario']==scen)]
            out=dict(scenario=scen,mode=mode,points=len(q),target_conditional_reference_TV=avg(drift),F1_bootstrap_width=avg(r['F1_hi']-r['F1_lo'] for r in q),F1_bootstrap_empirical_coverage=avg(float(r['F1_bootstrap_covered']) for r in q if r['F1_bootstrap_covered'] is not None))
            for side in range(2):
                for key in ['residual_TV','separation','boundary','residual_above_source95']:out['side'+str(side)+'_'+key]=avg(float(r['side'+str(side)+'_'+key]) for r in q if not r['side'+str(side)+'_empty_target_side'])
                out['e'+str(side)+'_MAE']=avg(r['e'+str(side)+'_absolute_error'] for r in q)
            summary.append(out)
    c.save(c.OUT/'机制诊断汇总.csv',summary)
if __name__=='__main__':main()
