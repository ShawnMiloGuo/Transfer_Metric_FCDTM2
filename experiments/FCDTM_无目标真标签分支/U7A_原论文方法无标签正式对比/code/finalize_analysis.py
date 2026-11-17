from pathlib import Path
import subprocess,time,tarfile,json,os,csv
u=Path('/home/Shanxin.Guo/ZhangtuosCode/Transfer_Metric_FCDTM2/experiments/FCDTM_无目标真标签分支/U7A_原论文方法无标签正式对比');py='/home/Shanxin.Guo/anaconda3/envs/FCDTM/bin/python'
while True:
 s=(u/'logs/finish.log').read_text()
 if 'U7A results ready' in s:break
 if 'Traceback' in s:raise RuntimeError(s[-2000:])
 time.sleep(10)
with tarfile.open('/tmp/u7a_final_eval.tar.gz') as t:t.extractall(u)
subprocess.check_call([py,str(u/'code/evaluate.py')])
(u/'统计核对说明.md').write_text('''# 统计数值核对说明\n\n首次封存后评价完成后，发现原通用相关/并列代码的绝对1e-12阈值不适合可能接近0的GBC原生分数。已将相关计算先按自身范围无量纲化、仅在真实常量时记不可定义；选模并列用相对尺度。随后从同一封存分数完整重算表图，未改预测、像素、参数或方向。主要五方法相关采用共同可定义任务与seed交集，对角GBC敏感性覆盖另列。原始每法覆盖保留。此为数值正确性修正，不是按目标结果挑选度量。\n''')
# Add the plain-language interpretation using the final paired summary.
a=list(csv.DictReader((u/'artifacts/相关汇总.csv').open()));p=u/'实验结果分析.md';s=p.read_text();intro='''\n## 如何理解本轮结果\n\n主表比较的是同一个冻结模型在不同目标图组上的精度下降变化，不是直接预测F1的具体数值。FCDTM硬/软版用模型自己的目标预测代替真实类别掩膜，保留原式的特征差异与头权重；因此是伪标签适配，不是原有监督版本的直接复现。\n\n以下取正文声明补齐的42任务，在主要五方法共同可定义任务/种子上比较。越接近1的正相关，通常表示越能跟上真实下降变化；负值可能表示方向相反。源域/目标标签机会、采样和符号一致，但原生特征尺度不同，不能把这些数直接当F1下降量。\n\n|方法中文名|Fisher汇总r|rho|\n|---|---:|---:|\n'''
names={'FCDTM_HardPL':'FCDTM硬伪标签','FCDTM_SoftPL':'FCDTM软伪标签','FD_global':'全局FD距离','DS_pixel':'平方距离修正的DS','GBC_spherical_PL':'球形GBC伪标签','GBC_diagonal_PL':'对角GBC伪标签（敏感性）'}
for r in a:
 if r['cohort']=='D42' and r['scenario']=='all' and r['method'] in names:intro+=f"|{names[r['method']]}|{float(r['Fisher_r']):.4f}|{float(r['rho']):.4f}|\n"
intro+='''\n这张表只能支持本轮方法之间的比较。具体跨传感器与跨地域结果见主表，不把总体较好写成每个类别都更好。GBC常量或缺类导致相关不可定义，覆盖详见CSV；不拿各方法不同有效任务的均值直接争胜。主表Fisher r与图上第一预设种子的原始r均值口径不同，数字不必相同。\n\n还不能据此证明完整的返修创新性：本轮尚未运行审稿提出的ATC、DoC、LogME等新增对照（属于U7B），尚未用新架构或独立网格确认。软版优于硬版如有出现，只表示使用概率加权可能更有帮助，不证明伪标签真实正确，也不直接证明权重相对无权重的贡献；后者需U7C。原式保护/不保护结果、源均值接近0维度均留档。\n\n模型选择是另一项检验：CSV保留同类同目标候选的遗憾、Top-1和不可算配置；它采用原生分数的预定方向作启发式选择，不能把它说成已经校准的目标F1预测。\n'''
s=s.replace('## 先说明本轮验证什么',intro+'\n## 先说明本轮验证什么',1);s+='\n额外独立FD复核实际时序见[评价时序说明](评价时序说明.md)，相关尺度修正见[统计核对说明](统计核对说明.md)。所有预测哈希仍一致。\n';p.write_text(s)
assert len(list((u/'artifacts/figures').glob('*.png')))==54
(u/'图表索引.md').write_text('# U7A图表索引\n\n'+'\n'.join(f'- [{p.stem} PNG](artifacts/figures/{p.name}) / [PDF](artifacts/figures/{p.stem}.pdf)' for p in sorted((u/'artifacts/figures').glob('*.png')))+'\n')
p=u/'TODO.md';s=p.read_text().replace('- [ ] 通俗分析、图形检查、同步本地、更新TODO、提交独立分支。','- [ ] 通俗分析已完成；待最终图形检查、本地同步及独立分支提交。');p.write_text(s)
subprocess.check_call(['tar','--exclude=__pycache__','-czf','/tmp/u7a_final_results.tar.gz','-C',str(u.parent),u.name]);print('FINAL RESULTS READY',flush=True)
