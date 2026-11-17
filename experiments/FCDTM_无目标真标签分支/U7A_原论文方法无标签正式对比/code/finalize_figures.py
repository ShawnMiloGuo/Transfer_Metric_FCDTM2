from pathlib import Path
import sys,json,subprocess,hashlib
from collections import defaultdict
u=Path('/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/experiments/FCDTM_无目标真标签分支/U7A_原论文方法无标签正式对比');sys.path.insert(0,str(u/'code'));import common as c;import plots
items=defaultdict(list)
for r in c.read(u/'artifacts/逐项评价.csv'):
 if r['kind']!='batch4':continue
 for k in c.METHODS+['source_F1','true_F1','true_AD']:r[k]=float(r[k])
 for k in ['cls','seed','repeat','budget']:r[k]=int(r[k])
 r['common']=r['common']=='True';items[r['cls'],r['src'],r['tgt'],r['seed']].append(r)
# Independent synthetic checks for both FCDTM aggregation paths and unequal-variance GBC.
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
zz=np.array([[0.,1.],[2.,5.]]);pp=np.array([False,True]);qq=np.array([.2,.8]);st=c.summary(zz,pp,qq);values=c.scores(st,st,np.array([1.,2.]),np.array([-3.,1.]))
assert abs(values['FCDTM_HardPL']+1.5)<1e-12 and abs(values['FCDTM_SoftPL']+.7)<1e-12
zero=c.scores(st,st,np.array([0.,2.]),np.array([-3.,1.]));assert np.isfinite(zero['FCDTM_HardPL']) and np.isnan(zero['FCDTM_HardPL_unprotected'])
z=np.array([[-2.,-3.],[-1.,-1.],[0.,0.],[1.,2.],[2.,4.],[4.,8.] ]);labels=np.array([False]*3+[True]*3);ss=c.summary(z,labels,np.full(6,.5));mu=np.array([z[~labels].mean(0),z[labels].mean(0)]);var=np.array([z[~labels].var(0),z[labels].var(0)])
for spherical in [True,False]:
 vv=np.broadcast_to(var.mean(1)[:,None],var.shape) if spherical else var
 bc=1.
 for j in range(2):bc*=quad(lambda x:np.sqrt(norm.pdf(x,mu[0,j],np.sqrt(vv[0,j]))*norm.pdf(x,mu[1,j],np.sqrt(vv[1,j]))),-100,100,epsabs=1e-12)[0]
 assert abs(c.gbc(ss,spherical)+2*bc)<1e-9
(u/'artifacts/额外公式核验.json').write_text(json.dumps(dict(status='passed',FCDTM_hard_hand_calculation=True,FCDTM_soft_hand_calculation=True,zero_denominator_protected_vs_unprotected=True,GBC_spherical_and_diagonal_independent_unequal_variance_integrals=True,target_labels_read=False),indent=1))
plots.draw(items,[],paper_only=True)
p=u/'实验结果分析.md';s=p.read_text().replace('## 论文式总览图','## 论文式总览图 D42主清单');s+='\n四张总览图仅含正文声明补齐的D42任务，与主结果表一致；50张逐任务图另外保留全类别扩展。类别编号及复现步骤见README。\n';p.write_text(s)
p=u.parent/'U7_原论文与审稿新增方法正式对比/TODO.md';s=p.read_text().replace('- [ ] 完成公式与作者实现数值一致性、空类/常量/协方差/符号核验。','- [x] 完成公式、独立参考积分/矩阵表达式、空类/常量/协方差/符号核验；未执行作者TF整流程逐位对照，限制另记。');p.write_text(s)
for name,src in [('finalize_analysis.py','/tmp/u7a_complete.py'),('finalize_delivery.py','/tmp/u7a_publish_finish.py'),('finalize_figures.py','/tmp/u7a_qafinal.py')]:
 (u/'code'/name).write_text(Path(src).read_text())
(u/'最终评价代码SHA.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (u/'code').glob('*.py')},indent=1))
subprocess.check_call(['/home/Shanxin.Guo/anaconda3/envs/FCDTM/bin/python',str(u.parent/'归档核验.py')]);subprocess.check_call(['tar','--exclude=__pycache__','-czf','/tmp/u7a_delivery_final.tar.gz','-C',str(u.parent.parent),u.parent.name]);print('FINAL QA PACKAGE READY')
