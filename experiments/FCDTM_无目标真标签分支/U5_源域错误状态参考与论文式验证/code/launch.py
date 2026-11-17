import json,subprocess,sys,time,importlib.util
from concurrent.futures import ThreadPoolExecutor
import common as c
spec=importlib.util.spec_from_file_location('u0audit',c.SRC/'code/launch.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def main():
    m=c.check();logs=c.BASE/'logs';logs.mkdir(exist_ok=True);start=time.time();assert not (c.OUT/'seal.json').exists()
    def one(t):
        stem=f"{t['cls']}_{t['src']}";trace=logs/(stem+'.strace')
        with (logs/(stem+'.log')).open('w') as f:p=subprocess.run(['strace','-f','-s','4096','-e','trace=open,openat','-o',str(trace),sys.executable,str(c.BASE/'code/worker.py'),str(t['cls']),t['src']],stdout=f,stderr=subprocess.STDOUT)
        audit=old.audit(trace,t);(c.OUT/f'audit_{stem}.json').write_text(json.dumps(audit,indent=1));print('WORKER',stem,p.returncode,audit['status'],flush=True);return dict(cls=t['cls'],src=t['src'],code=p.returncode,audit=audit['status'])
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(one,m['models']))
    (c.OUT/'execution.json').write_text(json.dumps(dict(seconds=time.time()-start,workers=results),indent=1));assert all(r['code']==0 and r['audit']=='passed' for r in results)
    rows=[]
    for t in m['models']:
        rs=c.read(c.OUT/f"worker_{t['cls']}_{t['src']}.csv");assert len(rs)==32;rows+=rs
    assert len(rows)==800;c.save(c.OUT/'predictions.csv',rows);assert len(list((c.OUT/'unlabeled_cache').glob('*.npz')))==50
    files=[p for p in c.OUT.rglob('*') if p.is_file() and p.name!='seal.json'];(c.OUT/'seal.json').write_text(json.dumps({str(p.relative_to(c.BASE)):c.sha(p) for p in files},indent=1));print('SEALED U5 before target label evaluation',flush=True)
if __name__=='__main__':main()
