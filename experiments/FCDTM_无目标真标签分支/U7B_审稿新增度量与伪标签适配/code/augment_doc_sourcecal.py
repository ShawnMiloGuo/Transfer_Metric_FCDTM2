"""Complete DoC regression adaptation using source-only auxiliary accuracy.
Separate pre-reference stage. Raw DoC gap is retained, not mislabeled as the
complete supervised regression system. No target annotation/evaluation access.
"""
from pathlib import Path
import sys,json,csv,hashlib,shutil
import numpy as np
B=Path(__file__).resolve().parents[1];O=B/'artifacts';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
DOM=['dwq_sentinel2','dwq_landsat8','xj_sentinel2','xj_landsat8']
def main():
 if (O/'doc_sourcecal_added.json').exists():return
 assert (O/'seal_full.json').exists()
 for p,h in json.loads((O/'seal_full.json').read_text()).items():assert sha(B/p)==h
 def guard(e,a):
  if e=='open' and a and isinstance(a[0],(str,bytes)):
   name=str(a[0])
   if '/annotations/' in name or any(s in name for s in ['逐项评价','评价真值','true_F1']):raise PermissionError('Source-only DoC fitting cannot read target reference/evaluation data')
 sys.addaudithook(guard)
 with (O/'predictions_full.csv').open() as f:rows=list(csv.DictReader(f))
 assert all('true_F1' not in r for r in rows)
 archive=B/'archive/pre_doc_sourcecal';archive.mkdir(parents=True,exist_ok=True);shutil.copy2(O/'seal_full.json',archive/'seal_full.json');shutil.copy2(O/'predictions_full.csv',archive/'predictions_full.csv');models={};diagnostics=[]
 for cls,src in sorted({(int(r['cls']),r['src']) for r in rows}):
  stem=f'{cls}_{src}';meta=json.loads((O/'source'/f'{stem}.json').read_text());plan=meta['calibration'];fits={}
  with np.load(O/'features'/f'{stem}_source.npz') as z:
   assert set(z.files)=={'z','p','pred','source_y','counts','offsets'}
   p=z['p'];yp=z['pred'];ys=z['source_y'];off=z['offsets']
  if 'rows' in plan:
   cap=np.random.default_rng(301121+cls*1000+DOM.index(src)*100).choice(len(p),256,False)
   references={'full2048':float(np.maximum(p,1-p).mean()),'shared256':float(np.maximum(p[cap],1-p[cap]).mean())}
   for panel in ['full2048','shared256']:
    auxiliary=[]
    for role in ['fit','check']:
     rr=[r for r in plan['rows'] if r['role']==role];pool=np.array(plan['fit_images'] if role=='fit' else plan['check_images'])
     for j,r in enumerate(rr):
      rng=np.random.default_rng(399121+cls*1000+DOM.index(src)*100+j+(0 if role=='fit' else 10000));ix=rng.choice(pool,4,False);assert list(ix)==r['indices']
      pp=np.concatenate([p[off[i]:off[i+1]] for i in ix]);pred=np.concatenate([yp[off[i]:off[i+1]] for i in ix]);truth=np.concatenate([ys[off[i]:off[i+1]] for i in ix]);c=rng.choice(len(pp),256,False)
      if panel=='shared256':pp=pp[c];pred=pred[c];truth=truth[c]
      auxiliary.append(dict(role=role,indices=list(map(int,ix)),gap=references[panel]-float(np.maximum(pp,1-pp).mean()),accuracy=float((pred==truth).mean())))
    train=[r for r in auxiliary if r['role']=='fit'];check=[r for r in auxiliary if r['role']=='check'];x=np.array([r['gap'] for r in train]);y=np.array([r['accuracy'] for r in train]);center=float(x.mean());scale=max(float(x.std()),1e-8);xx=(x-center)/scale;inter=float(y.mean());coef=float(xx@(y-inter)/(xx@xx+.001));error=[abs(inter+(r['gap']-center)/scale*coef-r['accuracy']) for r in check];fit=dict(mean=center,scale=scale,intercept=inter,coef=coef);fits[panel]=fit;diagnostics.append(dict(cls=cls,src=src,panel=panel,**fit,source_check_accuracy_MAE=float(np.mean(error)),ridge_lambda=.001,auxiliary=auxiliary))
  models[cls,src]=fits
 for r in rows:
  fit=models[int(r['cls']),r['src']].get(r['panel']);r['DoC_Acc_SrcCal']=float(fit['intercept']+(float(r['DoC'])-fit['mean'])/fit['scale']*fit['coef']) if fit else float('nan')
 with (O/'predictions_full.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report=dict(method='DoC_Acc_SrcCal',target_unit='pixel_accuracy_not_F1',target_reference_reads=0,source_labels='per-candidate cached source_y only',predictions_clipped=False,source_models=diagnostics,code_sha256=sha(Path(__file__)))
 (O/'doc_sourcecal_added.json').write_text(json.dumps(report,indent=1))
 files=[p for p in O.rglob('*') if p.is_file() and p.name!='seal_full.json' and not any(s in p.name for s in ['评价','相关','选择']) and not p.name.startswith('evaluation_')];(O/'seal_full.json').write_text(json.dumps({str(p.relative_to(B)):sha(p) for p in files},indent=1));print('ADDED SOURCE-ONLY DoC accuracy regression and resealed; target reference reads=0',flush=True)
if __name__=='__main__':main()
