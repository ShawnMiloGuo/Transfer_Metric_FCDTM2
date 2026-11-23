# -*- coding: utf-8 -*-
from common import *
import subprocess,time,concurrent.futures
m=json.loads((OLD/'artifacts/manifest_full.json').read_text());(O/'ot_logs').mkdir(exist_ok=True);start=time.time()
def one(x):
 stem=f'{x["cls"]}_{x["src"]}';out=O/'ot_workers'/f'{stem}.csv';audit=O/'ot_workers'/f'{stem}_audit.json'
 if out.exists() and audit.exists():return dict(model=stem,reused=True,exit=0)
 with (O/'ot_logs'/f'{stem}.log').open('w') as log:
  p=subprocess.run([sys.executable,str(B/'code/ot_worker.py'),str(x['cls']),x['src']],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 return dict(model=stem,exit=p.returncode)
workers=int(sys.argv[1]) if len(sys.argv)>1 else 4
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
 rows=[]
 for r in pool.map(one,m['models']):rows.append(r);print(r,flush=True)
dump(O/'execution_ot.json',dict(seconds=time.time()-start,workers=workers,models=rows));assert all(r['exit']==0 for r in rows),rows
print('OT_APPROX_COMPLETE',flush=True)
