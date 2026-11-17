import json
import common as c
def main():
    c.OUT.mkdir(exist_ok=True);assert not (c.OUT/'manifest.json').exists()
    prior=c.REPO/'experiments/E18_E13融合模型权重分布差异/artifacts/manifest.json';m=json.loads(prior.read_text())
    models=[{k:t[k] for k in ['cls','src','ck','sha','ref']} for t in m['models']]
    files={str(p.relative_to(c.BR)):c.sha(p) for p in c.BR.rglob('*.py') if p.name not in ['package.py']}
    for folder in [c.BASE,c.U1,c.U2]:files[str((folder/'协议.md').relative_to(c.BR))]=c.sha(folder/'协议.md')
    (c.OUT/'manifest.json').write_text(json.dumps(dict(models=models,tasks=[{k:t[k] for k in ['cls','src','tgt']} for t in m['tasks']],pools=m['pools'],historical_inventory_sha=c.sha(prior),files=files,development=True,source_pixel_sample=512),ensure_ascii=False,indent=1));print('FROZEN 25 models / 50 tasks; no historical target cache input')
if __name__=='__main__':main()
