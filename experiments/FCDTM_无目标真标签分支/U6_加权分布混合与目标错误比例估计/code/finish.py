import time,json,subprocess,hashlib
from pathlib import Path
b=Path('/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/experiments/FCDTM_无目标真标签分支/U6_加权分布混合与目标错误比例估计');py='/home/Shanxin.Guo/anaconda3/envs/FCDTM/bin/python'
start=time.time()
while not (b/'artifacts/seal.json').exists():
 if (b/'artifacts/execution.json').exists():raise RuntimeError('Workers ended without seal; review audits before evaluation')
 if time.time()-start>7200:raise RuntimeError('No seal after two hours; evaluation not run')
 time.sleep(20)
p=b/'TODO.md';s=p.read_text().replace('- [ ] 完成25模型','- [x] 完成25模型');p.write_text(s)
subprocess.check_call([py,str(b/'code/verify.py'),'final'])
s=p.read_text().replace('- [ ] 封存1300','- [x] 封存1300');p.write_text(s)
subprocess.check_call([py,str(b/'code/evaluate.py')])
s=p.read_text().replace('- [ ] 独立读取','- [x] 独立读取');p.write_text(s)
print('U6 inference, independent replay and post-seal evaluation completed',flush=True)
