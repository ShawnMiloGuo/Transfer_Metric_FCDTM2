import sys,json,time,subprocess
from concurrent.futures import ThreadPoolExecutor
import common as c
old=c.module('u6nativeaudit',c.U0/'code/launch.py')
def main():
    m=c.check();assert not (c.OUT/'seal.json').exists();logs=c.BASE/'logs';logs.mkdir(exist_ok=True);start=time.time()
    def one(t):
        stem=f"{t['cls']}_{t['src']}";trace=logs/(stem+'.strace')
        with (logs/(stem+'.log')).open('w') as f:p=subprocess.run(['strace','-f','-s','4096','-e','trace=open,openat','-o',str(trace),sys.executable,str(c.BASE/'code/worker.py'),str(t['cls']),t['src']],stdout=f,stderr=subprocess.STDOUT)
        audit=old.audit(trace,t);(c.OUT/f'audit_{stem}.json').write_text(json.dumps(audit,indent=1));print('WORKER',stem,p.returncode,audit['status'],flush=True);return dict(cls=t['cls'],src=t['src'],code=p.returncode,audit=audit['status'])
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(one,m['models']))
    (c.OUT/'execution.json').write_text(json.dumps(dict(seconds=time.time()-start,workers=results),indent=1));assert all(r['code']==0 and r['audit']=='passed' for r in results),'No seal on failed worker/audit'
    rows=[];diag=[]
    for t in m['models']:
        stem=f"{t['cls']}_{t['src']}";rs=c.read(c.OUT/f'worker_{stem}.csv');assert len(rs)==52;rows+=rs;diag+=json.loads((c.OUT/f'diagnostics_{stem}.json').read_text())
    assert len(rows)==1300 and len(diag)==2600;assert len(list((c.OUT/'unlabeled_cache').glob('*.npz')))==50;c.save(c.OUT/'predictions.csv',rows);(c.OUT/'diagnostics.json').write_text(json.dumps(diag,indent=1))
    files=[p for p in c.OUT.rglob('*') if p.is_file() and p.name!='seal.json'];seal=dict(files={str(p.relative_to(c.BASE)):c.sha(p) for p in files},target_truth_read=False)
    (c.OUT/'seal.json').write_text(json.dumps(seal,indent=1));print('SEALED U6 500 four-image and 800 pool predictions before truth',flush=True)
if __name__=='__main__':main()
