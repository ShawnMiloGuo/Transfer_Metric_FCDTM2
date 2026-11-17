import hashlib,json
from pathlib import Path
BR=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in json.loads((BR/'U0_标签隔离与无标签缓存/artifacts/seal.json').read_text()).items():assert sha(BR/p)==h,p
files=sorted(p for p in BR.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='文件校验.json')
(BR/'文件校验.json').write_text(json.dumps({str(p.relative_to(BR)):sha(p) for p in files},ensure_ascii=False,indent=1));print('PASS: frozen predictions intact, indexed',len(files),'files')
