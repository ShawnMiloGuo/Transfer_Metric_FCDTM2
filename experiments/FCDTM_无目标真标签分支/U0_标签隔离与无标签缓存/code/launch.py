import json,re,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
import common as c
def audit(trace,t):
    allowed={str(c.DATA/t['src']/'train_val/annotations/val'/n) for n in t['ref']};observed=set();bad=[];historical=[]
    for line in trace.read_text(errors='replace').splitlines():
        for path in re.findall(r'"([^"\n]*)"',line):
            path=re.sub(r'\\([0-7]{3})',lambda match:chr(int(match[1],8)),path)
            try:path=path.encode('latin1').decode('utf8')
            except (UnicodeEncodeError,UnicodeDecodeError):pass
            if '/annotations/' in path:
                # Native GDAL may probe sidecars for permitted source rasters.
                if not any(path==p or path.startswith(p+'.') for p in allowed):bad.append(path)
                observed.add(path)
            if '/experiments/' in path and str(c.BR) not in path and not path.endswith(('.py','.pyc')):historical.append(path)
    complete=allowed.issubset(observed)
    return dict(status='passed' if not bad and not historical and complete else 'failed',source_reads_complete=complete,annotation_paths=sorted(observed),unauthorized_annotations=sorted(set(bad)),historical_data_paths=sorted(set(historical)),trace_sha=c.sha(trace),source=t['src'],cls=t['cls'])
def main():
    m=c.check();logs=c.BASE/'logs';logs.mkdir(exist_ok=True);start=time.time()
    def one(t):
        stem=f"{t['cls']}_{t['src']}";trace=logs/(stem+'.strace')
        with (logs/(stem+'.log')).open('w') as f:
            p=subprocess.run(['strace','-f','-s','4096','-e','trace=open,openat','-o',str(trace),sys.executable,str(c.BASE/'code/worker.py'),str(t['cls']),t['src']],stdout=f,stderr=subprocess.STDOUT)
        result=audit(trace,t);(c.OUT/f'audit_{stem}.json').write_text(json.dumps(result,indent=1));print('WORKER',stem,p.returncode,result['status'],flush=True);return dict(cls=t['cls'],src=t['src'],code=p.returncode,audit=result['status'])
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(one,m['models']))
    (c.OUT/'execution.json').write_text(json.dumps(dict(seconds=time.time()-start,workers=results),indent=1));assert all(r['code']==0 and r['audit']=='passed' for r in results),'Worker/audit failure; no seal'
    assert len(list((c.OUT/'unlabeled_cache').glob('*.npz')))==50
    for folder in [c.U1,c.U2]:
        rows=[]
        for t in m['models']:
            data=c.read(folder/f"artifacts/worker_{t['cls']}_{t['src']}.csv");assert len(data)==32;rows+=data
        assert len(rows)==800;c.save(folder/'artifacts/predictions.csv',rows)
    files=[p for folder in [c.BASE,c.U1,c.U2] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.log','.strace'] and p.name!='seal.json']
    (c.OUT/'seal.json').write_text(json.dumps({str(p.relative_to(c.BR)):c.sha(p) for p in sorted(files)},ensure_ascii=False,indent=1));print('SEALED U0/U1/U2 before target truth evaluation',flush=True)
if __name__=='__main__':main()
