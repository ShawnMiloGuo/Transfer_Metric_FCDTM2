# -*- coding: utf-8 -*-
"""Resume/monitor own workers with bounded 20-core parallelism."""
from common import *
import time,subprocess,concurrent.futures,signal
m=json.loads((OLD/'artifacts/manifest_full.json').read_text());start=time.time();active={};transition=[]
# Stop only our obsolete 8-worker supervisor; leave its active children running.
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().split(b'\0')
  if str(B/'code/launch.py').encode() in argv:
   pid=int(p.name);os.kill(pid,signal.SIGTERM);transition.append(pid)
 except (OSError,PermissionError):pass
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes().split(b'\0');marker=str(B/'code/worker.py').encode()
  if marker in argv:
   i=argv.index(marker);active[(int(argv[i+1]),argv[i+2].decode())]=int(p.name)
 except (OSError,PermissionError,ValueError):pass
dump(O/'supervisor_transition.json',dict(old_supervisors_stopped=transition,active_workers_retained={f'{a}_{b}':p for (a,b),p in active.items()},max_mmd_workers=20,ot_workers=4,no_worker_killed=True))
def one(x):
 stem=f'{x["cls"]}_{x["src"]}';out=O/'workers'/f'{stem}.csv';audit=O/'workers'/f'{stem}_audit.json';pid=active.get((x['cls'],x['src']))
 while pid and not (out.exists() and audit.exists()):
  try:os.kill(pid,0)
  except ProcessLookupError:break
  time.sleep(2)
 if out.exists() and audit.exists():return dict(model=stem,reused_or_monitored=True,exit=0)
 with (O/'logs'/f'{stem}.log').open('w') as log:
  p=subprocess.run([sys.executable,str(B/'code/worker.py'),str(x['cls']),x['src']],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 return dict(model=stem,exit=p.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
 rows=[]
 for r in pool.map(one,m['models']):rows.append(r);print(r,flush=True)
dump(O/'execution_mmd_ablation.json',dict(seconds=time.time()-start,workers=20,models=rows,source='continued first supervisor workers without discarding computation'));assert all(r['exit']==0 for r in rows),rows;print('MMD_AND_ABLATION_COMPLETE',flush=True)
