import sys,json,time,subprocess,queue,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import common as c

def unit():
 from scipy.integrate import quad
 from scipy.stats import norm
 a=c.summary(np.array([[-1.],[1.]]),np.array([False,True]),np.array([.2,.8]));b=c.summary(np.array([[1.],[3.]]),np.array([False,True]),np.array([.2,.8]));assert abs(c.frechet(a,b)-4)<1e-10
 z=np.array([[-2.],[-1.],[1.],[2.]]);s=c.summary(z,z[:,0]>0,np.full(4,.5));r=c.scores(s,s,np.array([1.]),np.array([1.]));assert abs(r['DS_pixel']-np.log(9))<1e-12
 integral=quad(lambda x:np.sqrt(norm.pdf(x,-1.5,.5)*norm.pdf(x,1.5,.5)),-10,10)[0];assert abs(c.gbc(s)+2*integral)<1e-10
 assert np.isnan(c.gbc(c.summary(z,np.zeros(4,bool),np.zeros(4))))
 same=c.summary(np.zeros((4,2)),np.zeros(4,bool),np.zeros(4));assert c.frechet(same,same)==0
 (c.OUT/'unit.json').write_text(json.dumps(dict(status='passed',FD_1d_translation=True,DS_squared_log=True,GBC_independent_density_integral=True,missing_class_NA=True,singular_covariance=True),indent=1));print('UNIT PASSED',flush=True)

def main():
 c.OUT.mkdir(exist_ok=True);assert not (c.OUT/'manifest.json').exists();unit();old=json.loads((c.U0/'artifacts/manifest.json').read_text());taskcsv=c.read(c.BR/'U7_原论文与审稿新增方法正式对比/artifacts/论文与当前任务对照.csv');cohorts={k:[(int(r['cls']),r['src'],r['tgt']) for r in taskcsv if r[col]=='True'] for k,col in [('P37','published_main'),('D42','declared_III_C'),('E50','current_manifest')]};files=[p for p in (c.BASE/'code').glob('*.py') if p.name not in ['evaluate.py','plots.py','report.py']]+[c.BASE/'协议.md'];m=dict(models=old['models'],tasks=old['tasks'],pools=old['pools'],cohorts=cohorts,files={str(p.relative_to(c.BASE)):c.sha(p) for p in files},samples_per_image=2048,seeds=c.SEEDS,target_truth_for_predictors=False)
 (c.OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=1));logs=c.BASE/'logs';logs.mkdir(exist_ok=True);q=queue.Queue()
 for device in range(4):q.put(device)
 audit=c.u0
 spec=c.u0.__spec__;import importlib.util;s=importlib.util.spec_from_file_location('nativeaudit',c.U0/'code/launch.py');audit=importlib.util.module_from_spec(s);s.loader.exec_module(audit);start=time.time()
 def one(t):
  device=q.get();stem=f"{t['cls']}_{t['src']}";trace=logs/(stem+'.strace')
  try:
   with (logs/(stem+'.log')).open('w') as f:p=subprocess.run(['strace','-f','-s','4096','-e','trace=open,openat','-o',str(trace),sys.executable,str(c.BASE/'code/worker.py'),str(t['cls']),t['src'],str(device)],stdout=f,stderr=subprocess.STDOUT)
   a=audit.audit(trace,t);(c.OUT/f'audit_{stem}.json').write_text(json.dumps(a,indent=1));print('WORKER',stem,p.returncode,a['status'],flush=True);return dict(model=stem,exit=p.returncode,audit=a['status'])
  finally:q.put(device)
 with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,m['models']))
 (c.OUT/'execution.json').write_text(json.dumps(dict(seconds=time.time()-start,workers=results),indent=1));assert all(r['exit']==0 and r['audit']=='passed' for r in results)
 rows=[]
 for t in m['models']:rows+=c.read(c.OUT/f"worker_{t['cls']}_{t['src']}.csv")
 assert len(rows)==3300;c.save(c.OUT/'predictions.csv',rows);(c.OUT/'seal.json').write_text(json.dumps({str(p.relative_to(c.BASE)):c.sha(p) for p in c.OUT.rglob('*') if p.is_file() and p.name!='seal.json'},indent=1));print('SEALED 3300 score records before target truth',flush=True)
 subprocess.check_call([sys.executable,str(c.BASE/'code/evaluate.py')])
if __name__=='__main__':main()
