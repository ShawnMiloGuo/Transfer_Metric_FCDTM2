import json,subprocess,sys,time
import common as c
def main():
    start=time.monotonic()
    while not (c.OUT/'seal.json').exists():
        p=c.OUT/'execution.json'
        if p.exists() and any(r['code'] or r['audit']!='passed' for r in json.loads(p.read_text())['workers']):raise RuntimeError('Worker/audit failure, evaluation forbidden')
        if time.monotonic()-start>10800:raise TimeoutError('No seal after 3h')
        time.sleep(10)
    for script,args in [('verify.py',['final']),('evaluate.py',[])]:
        print('STAGE',script,*args,flush=True);subprocess.run([sys.executable,str(c.BASE/'code'/script)]+args,check=True)
    print('COMPLETE U0/U1/U2 frozen first-round development',flush=True)
if __name__=='__main__':main()
