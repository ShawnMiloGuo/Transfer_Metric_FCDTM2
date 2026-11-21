"""Pre-label numerical canonicalization of a proved LEEP/NLEEP boundary.
One pseudo-label class => conditional target-label probability is exactly 1,
so log expectation is exactly zero. Keep failed/nonfinite scores as failures.
This file never opens source/target annotations or any evaluation cache.
"""
from pathlib import Path
import sys,csv,json,hashlib,shutil
import numpy as np
B=Path(__file__).resolve().parents[1];O=B/'artifacts'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (O/'seal_full.json').exists()
 if (O/'singleton_canonicalization.json').exists():return
 for p,h in json.loads((O/'seal_full.json').read_text()).items():assert sha(B/p)==h
 # Synthetic identity, fixed without inspecting target correctness.
 q=np.array([[.2,.3,.5],[.7,.2,.1]],dtype=np.float64);joint=q.sum(0)/2;cond=np.divide(joint,joint,out=np.zeros_like(joint),where=joint>0);assert np.allclose(q@cond,1.)
 def guard(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes)):
   p=str(a[0])
   if '/annotations/' in p or any(v in p for v in ['逐项评价','评价真值','true_F1']):raise PermissionError('Numerical finalizer may not read reference/evaluation data')
 sys.addaudithook(guard)
 with (O/'predictions_full.csv').open() as f:rows=list(csv.DictReader(f))
 archive=B/'archive/pre_singleton_correction';archive.mkdir(parents=True,exist_ok=True);shutil.copy2(O/'seal_full.json',archive/'seal_full.json');shutil.copy2(O/'predictions_full.csv',archive/'predictions_full.csv')
 domains=['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8'];cache={};changes=[];singletons=0
 for i,r in enumerate(rows):
  key=f"{r['cls']}_{r['src']}__{r['tgt']}"
  if key not in cache:
   with np.load(O/'features'/f'{key}.npz') as z:
    assert set(z.files)=={'z','p','pred','offsets'}
    cache[key]=(z['pred'],z['offsets'])
  yp,off=cache[key];ix=json.loads(r['indices']);pred=np.concatenate([yp[off[j]:off[j+1]] for j in ix])
  if r['panel']=='shared256':
   seed=420121+int(r['cls'])*100000+domains.index(r['src'])*10000+domains.index(r['tgt'])*1000+int(r['seed'])+int(r['budget'])*7+int(r['repeat']);cap=np.random.default_rng(seed).choice(len(pred),256,False);pred=pred[cap]
  singleton=bool(np.unique(pred).size==1);r['pseudo_foreground_count']=int(pred.sum());r['pseudo_pixel_count']=len(pred);r['singleton_PL']=singleton
  for method in ['LEEP_PL','NLEEP_PL']:
   raw=float(r[method]);r[method+'_raw_float']=raw
   if singleton and np.isfinite(raw):
    assert abs(raw)<=1e-6,(i,method,raw,'Unexpected non-roundoff error on a singleton label task')
    if raw!=0:changes.append(dict(record=i,method=method,raw=raw,exact=0.0))
    r[method]=0.
  singletons+=int(singleton)
  assert float(r['NCE_Pred_PL'])==0.
 with (O/'predictions_full.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report=dict(reason='One self-pseudo-label class implies exact LEEP/NLEEP score zero; remove roundoff-only variation.',records=len(rows),singleton_records=singletons,changed_values=len(changes),changes=changes,failed_scores_not_filled=True,reference_reads=0,finalizer_sha256=sha(Path(__file__)),initial_seal_sha256=sha(archive/'seal_full.json'))
 (O/'singleton_canonicalization.json').write_text(json.dumps(report,indent=1))
 files=[p for p in O.rglob('*') if p.is_file() and p.name!='seal_full.json' and not any(x in p.name for x in ['评价','相关','选择']) and not p.name.startswith('evaluation_')]
 (O/'seal_full.json').write_text(json.dumps({str(p.relative_to(B)):sha(p) for p in files},indent=1));print('CANONICALIZED AND RESEALED before target references:',len(changes),'roundoff-only values',flush=True)
if __name__=='__main__':main()
