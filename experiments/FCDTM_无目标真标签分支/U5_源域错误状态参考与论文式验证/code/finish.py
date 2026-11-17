import json,subprocess,sys,time
import common as c
def main():
    start=time.monotonic()
    while not (c.OUT/'seal.json').exists():
        p=c.OUT/'execution.json'
        if p.exists() and any(r['code'] or r['audit']!='passed' for r in json.loads(p.read_text())['workers']):raise RuntimeError('Worker/audit failed: evaluation blocked')
        if time.monotonic()-start>10800:raise TimeoutError('No seal after 3h')
        time.sleep(10)
    for name,args in [('verify.py',['final']),('evaluate.py',[])]:print('STAGE',name,*args,flush=True);subprocess.run([sys.executable,str(c.BASE/'code'/name)]+args,check=True)
    print('COMPLETE U5 and paper-style scatter figures',flush=True)
if __name__=='__main__':main()
