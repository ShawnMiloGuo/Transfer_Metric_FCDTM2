from pathlib import Path
import sys,os,json,hashlib,subprocess,queue,time,importlib.util
from concurrent.futures import ThreadPoolExecutor
B=Path(__file__).resolve().parents[1];O=B/'artifacts';A=B.parent/'U7A_原论文方法无标签正式对比';sys.path.insert(0,str(A/'code'));import common as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=True);assert not (O/'manifest_full.json').exists(),'Full protocol already frozen; do not overwrite.'
 env=dict(os.environ,PYTHONPATH=str(B/'.deps'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');subprocess.check_call([sys.executable,str(B/'code/unit_full.py')],env=env)
 m=json.loads((A/'artifacts/manifest.json').read_text());m.update(experiment='U7B',panels=['full2048','shared256'],shared_pixel_cap=256,target_reference_for_predictors=False,code_sha256={str(p.relative_to(B)):sha(p) for p in [B/'code/metrics.py',B/'code/worker_full.py',B/'code/logme_exact_moments.py',B/'code/score_moments.py',B/'协议.md']})
 (O/'manifest_full.json').write_text(json.dumps(m,ensure_ascii=False,indent=1));logs=B/'logs';logs.mkdir(exist_ok=True);spec=importlib.util.spec_from_file_location('audit_source',old.U0/'code/launch.py');aud=importlib.util.module_from_spec(spec);spec.loader.exec_module(aud)
 devices=queue.Queue()
 for k in range(4):devices.put(k)
 start=time.time()
 def one(t):
  device=devices.get();name=f"{t['cls']}_{t['src']}";trace=logs/(name+'.strace')
  try:
   with (logs/(name+'.log')).open('w') as f:r=subprocess.run(['strace','-f','-s','4096','-e','trace=open,openat','-o',str(trace),sys.executable,str(B/'code/worker_full.py'),str(t['cls']),t['src'],str(device)],stdout=f,stderr=subprocess.STDOUT,env=env)
   status=aud.audit(trace,t);(O/f'audit_{name}.json').write_text(json.dumps(status,indent=1));print('WORKER',name,r.returncode,status['status'],flush=True);return dict(model=name,exit=r.returncode,audit=status['status'])
  finally:devices.put(device)
 with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,m['models']))
 (O/'execution_full.json').write_text(json.dumps(dict(seconds=time.time()-start,workers=results),indent=1));assert all(r['exit']==0 and r['audit']=='passed' for r in results),results
 rows=[]
 for t in m['models']:rows+=old.read(O/f"worker_{t['cls']}_{t['src']}.csv")
 assert len(rows)==6600;old.save(O/'predictions_full.csv',rows)
 files=[p for p in O.rglob('*') if p.is_file() and p.name not in ['seal_full.json'] and not any(k in p.name for k in ['评价','相关','选择']) and not p.name.startswith('evaluation_')]
 (O/'seal_full.json').write_text(json.dumps({str(p.relative_to(B)):sha(p) for p in files},indent=1));print('SEALED ALL 6600 PREDICTIONS before new target evaluation',flush=True)
 # Evaluator is a distinct post-seal step; a missing evaluator does not authorize reading labels here.
 if (B/'code/evaluate_full.py').exists():subprocess.check_call([sys.executable,str(B/'code/evaluate_full.py')],env=env)
if __name__=='__main__':main()
