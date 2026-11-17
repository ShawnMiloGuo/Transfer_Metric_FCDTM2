"""Replay four-image predictions from sealed unlabeled statistics, no target labels."""
import json,importlib.util
import numpy as np
import common as c
spec=importlib.util.spec_from_file_location('batch4check',c.BASE/'code/batch4.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def main():
    b.check();rows=c.read(b.OUT/'predictions.csv');assert len(rows)==500;maximum=0.;count=0;by={}
    for r in rows:by.setdefault((int(r['cls']),r['src'],r['tgt']),[]).append(r)
    for (cls,src,tgt),rs in by.items():
        sf=json.loads((c.OUT/f'source/{cls}_{src}.json').read_text())['source_F1']
        with np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as z:
            for r in rs:
                ix=json.loads(r['indices']);assert len(ix)==4 and len(set(ix))==4 and json.loads(r['names'])==z['names'][ix].tolist()
                for method in c.RISK:
                    value=sf if method=='source_only' else c.f1(z['risk_'+method][ix]);assert np.isclose(value,float(r[method]),equal_nan=True,atol=1e-12);assert np.isclose(sf-value,float(r[method+'_AD']),equal_nan=True,atol=1e-12)
                    if np.isfinite(value):maximum=max(maximum,abs(value-float(r[method])))
                    count+=1
    out=dict(status='passed',records=len(rows),scalar_predictions_replayed=count,max_replay_difference=maximum,images_per_point=4,target_labels_read=False)
    (b.OUT/'预测独立复算.json').write_text(json.dumps(out,indent=1));print(json.dumps(out))
if __name__=='__main__':main()
