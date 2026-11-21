"""Explicitly versioned CPU retry after pre-feature CUDA failures; no method tuning."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,time,importlib.util,queue,shutil
from concurrent.futures import ThreadPoolExecutor
B=Path(__file__).resolve().parents[1];O=B/'artifacts';A=B.parent/'U7A_原论文方法无标签正式对比';sys.path.insert(0,str(A/'code'));import common as old
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (O/'seal_full.json').exists()

 archive=B/'archive/cpu_parallel_stall';archive.mkdir(parents=True,exist_ok=True);previous=json.loads((O/'manifest_full.json').read_text());shutil.copy2(O/'manifest_full.json',archive/'manifest_full.json')
 if (O/'execution_full.json').exists():shutil.copy2(O/'execution_full.json',archive/'execution_full.json')
 for p in O.glob('audit_*.json'):shutil.copy2(str(p),str(archive/p.name))
 if (B/'logs').exists():shutil.move(str(B/'logs'),str(B/'logs_cpu_parallel_stall'))
 logs=B/'logs';logs.mkdir()
 env=dict(os.environ,PYTHONPATH=str(B/'.deps'),CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');subprocess.check_call([sys.executable,str(B/'code/unit_full.py')],env=env)
 m=previous;m.update(execution_device='cpu',worker_entry='code/worker_cached_retry.py',gpu_failure_archive='archive/cpu_parallel_stall',fresh_all_model_predictions=True,code_sha256={str(p.relative_to(B)):sha(p) for p in [B/'code/metrics.py',B/'code/worker_cached_retry.py',B/'code/logme_exact_moments.py',B/'code/score_moments.py',B/'协议.md']});(O/'manifest_full.json').write_text(json.dumps(m,ensure_ascii=False,indent=1))
 spec=importlib.util.spec_from_file_location('audit_source',old.U0/'code/launch.py');aud=importlib.util.module_from_spec(spec);spec.loader.exec_module(aud);start=time.time()
 def one(t):
  name=f"{t['cls']}_{t['src']}";trace=logs/(name+'.strace')
  prior=O/f'audit_{name}.json'
  if (O/f'worker_{name}.csv').exists() and prior.exists() and json.loads(prior.read_text())['status']=='passed':
   print('REUSE_COMPLETED_CPU_WORKER',name,flush=True);return dict(model=name,exit=0,audit='passed',reused=True)
  with (logs/(name+'.log')).open('w') as f:r=subprocess.run(['strace','-f','-s','4096','-e','trace=open,openat','-o',str(trace),sys.executable,str(B/'code/worker_cached_retry.py'),str(t['cls']),t['src'],'-1'],stdout=f,stderr=subprocess.STDOUT,env=env)
  status=aud.audit(trace,t);(O/f'audit_{name}.json').write_text(json.dumps(status,indent=1));print('CPU_WORKER',name,r.returncode,status['status'],flush=True);return dict(model=name,exit=r.returncode,audit=status['status'])
 with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,m['models']))
 (O/'execution_full.json').write_text(json.dumps(dict(device='cpu',seconds=time.time()-start,workers=results),indent=1));assert all(r['exit']==0 and r['audit']=='passed' for r in results),results
 rows=[]
 for t in m['models']:rows+=old.read(O/f"worker_{t['cls']}_{t['src']}.csv")
 assert len(rows)==6600;old.save(O/'predictions_full.csv',rows)
 files=[p for p in O.rglob('*') if p.is_file() and p.name!='seal_full.json' and not any(k in p.name for k in ['评价','相关','选择']) and not p.name.startswith('evaluation_')]
 (O/'seal_full.json').write_text(json.dumps({str(p.relative_to(B)):sha(p) for p in files},indent=1));print('SEALED ALL 6600 CPU predictions before target evaluation',flush=True)
 if (B/'code/evaluate_full.py').exists():subprocess.check_call([sys.executable,str(B/'code/evaluate_full.py')],env=env)
if __name__=='__main__':main()
