"""U7B phase 0: exact cache-only H-score-PL and self-PL NCE.
Only unlabelled sufficient statistics and sealed predictor indices are readable.
No evaluation cache or target reference masks may be opened by this entry point.
"""
from pathlib import Path
import sys,csv,json,hashlib,time
import numpy as np
B=Path(__file__).resolve().parents[1];A=B.parent/'U7A_原论文方法无标签正式对比/artifacts';O=B/'artifacts';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def hscore(s):
 n=float(s['n']);mu=s['sum']/n
 cov=(s['outer']-np.outer(s['sum'],s['sum'])/n)/(n-1)
 cov=(cov+cov.T)/2
 between=np.zeros_like(cov)
 for j in range(2):
  if s['cn'][j]>0:
   d=s['cs'][j]/s['cn'][j]-mu;between+=s['cn'][j]*np.outer(d,d)/(n-1)
 return float(np.trace(np.linalg.pinv(cov,rcond=1e-15)@between))
def synthetic():
 rng=np.random.default_rng(17);x=rng.normal(size=(40,6));y=(x[:,0]>.2).astype(int)
 def summary(x,y):return dict(n=len(x),sum=x.sum(0),outer=x.T@x,cn=np.bincount(y,minlength=2),cs=np.array([x[y==j].sum(0) for j in range(2)]))
 g=np.array([x[y==j].mean(0) for j in y]);expected=float(np.trace(np.linalg.pinv(np.cov(x,rowvar=False),rcond=1e-15)@np.cov(g,rowvar=False)))
 assert abs(hscore(summary(x,y))-expected)<1e-10
 assert abs(hscore(summary(x,np.zeros(len(x),dtype=int))))<1e-10
 same=np.c_[x,x[:,0]]; yy=y;gg=np.array([same[yy==j].mean(0) for j in yy]);e=float(np.trace(np.linalg.pinv(np.cov(same,rowvar=False),rcond=1e-15)@np.cov(gg,rowvar=False)))
 assert abs(hscore(summary(same,yy))-e)<1e-9
 # Independent empirical conditional entropy when both label sequences are identical.
 p=np.bincount(y,minlength=2)/len(y);joint=np.diag(p);entropy=-sum(joint[t,s]*np.log(joint[t,s]/p[s]) for t in range(2) for s in range(2) if joint[t,s]>0)
 assert entropy==0.
 return dict(Hscore_direct_covariance_difference=abs(hscore(summary(x,y))-expected),singular_covariance=True,one_predicted_class_Hscore_zero=True,NCE_self_PL_constant_zero=True)
def main():
 assert not (O/'seal_phase0.json').exists(),'Outputs already sealed; use an explicit new version to rerun.'
 unit=synthetic();whitelist={str((A/'predictions.csv').resolve()),str((A/'manifest.json').resolve())}|{str(p.resolve()) for p in (A/'cache').glob('*.npz')}
 denied=[]
 def guard(event,args):
  if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
  p=Path(args[0]).resolve();name=str(p)
  if '/annotations/' in name or '逐项评价' in name or ('/artifacts/' in name and name not in whitelist and not p.is_relative_to(O.resolve())):
   denied.append(name);raise PermissionError('Predictor cannot access target truth or evaluation cache: '+name)
 sys.addaudithook(guard)
 try:open(A/'逐项评价.csv');raise AssertionError('Guard failed')
 except PermissionError:pass
 inputs={p:sha(Path(p)) for p in sorted(whitelist)}
 with (A/'predictions.csv').open() as f:old=list(csv.DictReader(f))
 result=[];start=time.time();current=None;cached=None
 for r in old:
  stem=f"{r['cls']}_{r['src']}__{r['tgt']}";p=A/'cache'/f'{stem}.npz'
  if stem!=current:
   if cached is not None:cached.close()
   cached=np.load(p);current=stem
   assert set(cached.files)=={'names','bits','valid_bits','n','sum','outer','cn','cs','cq','soft_sum','soft_mass'}
  ix=json.loads(r['indices']);s={k:cached[k][ix].sum(0) for k in ['n','sum','outer','cn','cs']}
  hs=hscore(s);assert np.isfinite(hs)
  # Both sequences are precisely the same model prediction. Nonzero entropy would be a bug.
  nce=0.0
  result.append({**{k:r[k] for k in ['cls','src','tgt','kind','budget','seed','repeat','indices','source_F1']},'Hscore_PL':hs,'NCE_Pred_PL':nce,'NCE_status':'degenerate_self_pseudo_labels','target_truth_read':False})
 assert len(result)==3300
 with (O/'predictions_phase0.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 audit=dict(unit=unit,records=len(result),seconds=time.time()-start,readable_input_sha256=inputs,guard_probe_blocked=bool(denied),target_truth_reads=0,methods=['Hscore_PL','NCE_Pred_PL'],NCE_interpretation='Mathematically constant; correlation undefined, not a failed implementation.')
 (O/'audit_phase0.json').write_text(json.dumps(audit,indent=2))
 outputs=[O/'predictions_phase0.csv',O/'audit_phase0.json'];(O/'seal_phase0.json').write_text(json.dumps({str(p.relative_to(B)):sha(p) for p in outputs},indent=2))
 print('SEALED',len(result),'H-score and NCE records; zero target reference/evaluation reads; seconds=',audit['seconds'],flush=True)
if __name__=='__main__':main()
