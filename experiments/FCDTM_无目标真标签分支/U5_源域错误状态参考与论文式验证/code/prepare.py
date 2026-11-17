import json
import common as c
def main():
    c.OUT.mkdir(exist_ok=True);assert not (c.OUT/'manifest.json').exists()
    for p,h in json.loads((c.SRC/'artifacts/seal.json').read_text()).items():assert c.sha(c.BR/p)==h,p
    old=json.loads((c.SRC/'artifacts/manifest.json').read_text());deps=[c.SRC/'artifacts/manifest.json',c.SRC/'code/common.py',c.SRC/'code/launch.py']
    deps+=list((c.SRC/'artifacts/source').glob('*.json'))+list((c.SRC/'artifacts/unlabeled_cache').glob('*.npz'))
    deps+=list((c.BASE/'code').glob('*.py'))+[c.BASE/'协议.md']
    files={str(p.relative_to(c.BR)):c.sha(p) for p in deps if p.name!='package.py'}
    (c.OUT/'manifest.json').write_text(json.dumps(dict(models=old['models'],tasks=old['tasks'],pools=old['pools'],files=files,shrink_pixels=512,development_reuse=True),ensure_ascii=False,indent=1));print('FROZEN U5 first-stage error-state reference; 25 models / 50 tasks')
if __name__=='__main__':main()
