import json
import common as c

def main():
    c.OUT.mkdir(exist_ok=True);assert not (c.OUT/'manifest.json').exists()
    old=json.loads((c.U0/'artifacts/manifest.json').read_text())
    deps=[c.U0/'code/common.py',c.U0/'code/launch.py',c.U0/'artifacts/manifest.json',c.U5/'code/common.py']
    deps+=list((c.U0/'artifacts/source').glob('*.json'))+list((c.U5/'artifacts/source').glob('*.json'))+list((c.U5/'artifacts/unlabeled_cache').glob('*.npz'))
    deps+=[c.BASE/'code'/name for name in ['common.py','worker.py','prepare.py','launch.py','verify.py']]+[c.BASE/'协议.md']
    m=dict(models=old['models'],tasks=old['tasks'],pools=old['pools'],files={str(p.relative_to(c.BR)):c.sha(p) for p in deps},development_reuse=True,bins=c.BINS,lambda_grid=c.LAMS,bootstrap=c.BOOT,workers=8)
    (c.OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=1));print('FROZEN U6; 25 models / 50 tasks / source-only lambda calibration')
if __name__=='__main__':main()
