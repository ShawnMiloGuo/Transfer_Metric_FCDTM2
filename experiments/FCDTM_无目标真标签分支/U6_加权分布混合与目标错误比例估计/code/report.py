"""Write reviewable results; never select new parameters from target outcomes."""
import json
from collections import Counter
import numpy as np
import common as c
import evaluate as ev

def f(x):return f'{float(x):.5f}' if np.isfinite(float(x)) else 'N/A'
def write(summary,errors,sels,validtasks):
    def err(method,scen='all',kind='batch4',scope='common_points',budget=None):
        b=(4 if kind=='batch4' else 0) if budget is None else budget
        return next(r for r in errors if r['kind']==kind and r['budget']==b and r['scope']==scope and r['scenario']==scen and r['method']==method)
    lines=['# U6 实验结果分析','','## 要验证什么','','U5将源域错误后验直接用于目标，跨地域出现乐观偏差。U6保留模型权重引导特征距离和源域错误状态参考，在目标无标签分布上拟合正确/错误的混合比例。检验这种比例更新能否减少精度下降预测误差，同时保持跨传感器表现和选模价值。','',
    '本轮25个冻结UNet、8类、50迁移任务，源64图/目标40图，与U5配对。主验证每4图一个子域、每任务10点，共500记录。附加800个8/16/24/40图无标签池预测。不是目标标注预算，也不是未见网格确认。','',
    '预测进程25个原生文件访问审计全部通过后封存；独立最小二乘重算5200个混合预测通过，才读取目标真标签评价。没有目标微调、目标真标签校准或事后调lambda。','',
    '## 主要四图结果','','下表在主六面板共同有限点上，按任务等权，相关再使用共同可定义任务。MAE=|预测AD-真实AD|=|预测F1-真实F1|；F1正偏差表示高估目标精度。源域F1固定，AD=源F1-目标F1。','',
    '|方法|有效点/500|AD MAE↓|F1偏差|平均r↑|平均rho↑|','|---|---:|---:|---:|---:|---:|']
    for method in c.RISK:
        a=err(method);r=next(r for r in summary if r['scope']=='common_tasks' and r['scenario']=='all' and r['method']==method);lines.append(f"|{method}|{a['valid_rows']}|{f(a['MAE'])}|{f(a['F1_bias'])}|{f(r['Pearson'])}|{f(r['Spearman'])}|")
    lines+=['',f'六面板共同可定义相关任务：{len(validtasks)}/50。各方法自身有效点与共同点结果分别保存在误差与偏差.csv，N/A不补0。source_only为常量AD，相关不可定义。硬/软伪标签原式和confidence的原生分数相关保存在CSV，不给它们虚构AD数值误差。','',
    '## 分场景判断','','|场景|方法|点数|MAE↓|F1偏差|','|---|---|---:|---:|---:|']
    for scen in ['sensor_GBA','sensor_XJ','region_S2','region_L8']:
        for method in ['U5_weighted','mix_weighted','mix_uniform','mix_weighted_raw','mix_uniform_raw','global_margin','global_weighted','ridge6']:
            a=err(method,scen);lines.append(f"|{scen}|{method}|{a['valid_rows']}|{f(a['MAE'])}|{f(a['F1_bias'])}|")
    a=err('mix_weighted');b=err('U5_weighted');un=err('mix_uniform');base=err('global_margin')
    lines+=['','### 预设假设的检验','',f"混合加权相对U5的总体四图MAE变化（负为改善）：{f(a['MAE']-b['MAE'])}；相对同容量无权重：{f(a['MAE']-un['MAE'])}；相对全局间隔：{f(a['MAE']-base['MAE'])}。"]
    for scen in ['sensor_GBA','sensor_XJ','region_S2','region_L8']:
        a=err('mix_weighted',scen);b=err('U5_weighted',scen);lines.append(f"- {scen}：相对U5 MAE变化{f(a['MAE']-b['MAE'])}，F1偏差由{f(b['F1_bias'])}变为{f(a['F1_bias'])}。")
    lines+=['','不能由局部相关或一个类别的改善推断整体胜出；lambda=0版本用于诊断源域校准是否过度收缩，不据其目标结果重新选lambda。不同方法若有效点不一致，优先查看共同点和覆盖率。','',
    '## 8/16/24无标签影像预算与模型选择','','误差按源模型—迁移任务等权，选择按类别—目标域等权、预算等权。重复子池共用影像，不是独立实验。候选模型与U5完全相同，方法不可算时整次配置记不可算，不删候选。选择可能同时涉及跨传感器和跨地域候选，不将其硬拆为单一场景。','',
    '|方法|AD MAE↓|共同可选配置/全部|平均遗憾↓|Top-1↑|最大损失↓|','|---|---:|---:|---:|---:|---:|']
    for method in c.RISK:
        a=err(method,kind='pool');r=next(r for r in sels if r['scope']=='common_configs' and r['budget']==0 and r['method']==method);lines.append(f"|{method}|{f(a['MAE'])}|{r['valid_configs']}/{r['total_configs']}|{f(r['regret'])}|{f(r['Top1'])}|{f(r['max_loss'])}|")
    lines+=['','原生分数对照、自身覆盖、分预算见选择汇总.csv；分类别误差见分类别误差.csv。相同F1预测并不保证相同选模表现，不能用任务内散点相关代替选模实验。','',
    '## 源域校准与参考支持','','lambda仅从[0,0.001,0.01,0.1,1]按源cal伪目标F1 MAE选择；原型、尺度、8箱分位点、0.5伪计数、100次空间块bootstrap均在目标评价前固定。','']
    for mode in c.MODES:
        params=[json.loads(p.read_text())['params'][mode] for p in (c.OUT/'source').glob('*.json')];lines.append(f"- {mode}的25模型lambda计数：{dict(sorted(Counter(p['lam'] for p in params).items()))}。")
    lines+=['','源fit/cal/test名单、直方图及每模型lambda候选损失见artifacts/source。内部test只是度量器检查；源64图历史上参与过模型选优，不能称分割网络独立OOF。','',
    '## 机制诊断：比例更新是否足够','','TV残差是目标直方图与拟合混合参考的差异；参考分离度小表示比例难估。超过源cal残差95分位仅表示参考解释能力不足，不直接等同于高错误率。','',
    '|场景|加权参考条件漂移TV|背景侧比例MAE|前景侧比例MAE|F1 bootstrap宽度|经验覆盖率|','|---|---:|---:|---:|---:|---:|']
    for r in c.read(c.OUT/'机制诊断汇总.csv'):
        if r['mode']=='weighted':lines.append(f"|{r['scenario']}|{f(float(r['target_conditional_reference_TV']))}|{f(float(r['e0_MAE']))}|{f(float(r['e1_MAE']))}|{f(float(r['F1_bootstrap_width']))}|{f(float(r['F1_bootstrap_empirical_coverage']))}|")
    lines+=['','条件漂移诊断使用封存后才读取的目标真值和每图固定512个采样像素，只用于解释失败，绝不作为部署输入。不同场景可用错误状态支持不同，缺失状态不记0。漂移TV包含采样误差，不能单独证明因果机制。bootstrap按空间行块重采样；四图子域可能只有少量块，区间是描述性的，不保证真值覆盖。','',
    '## 图表与读图','','主六面板横轴为预测AD，纵轴为真实AD，虚线是1:1；各任务颜色一致，相关按任务内计算后平均。每个点汇总4图计数。各方法同一有效点，保留负AD。']
    for scen in ['sensor_GBA','sensor_XJ','region_S2','region_L8']:lines+=['',f'![{scen}](artifacts/figures/paper_{scen}.png)']
    for name in ['error_bias','selection','source_lambda','error_proportions']:lines+=['',f'![{name}](artifacts/figures/{name}.png)']
    lines+=['','50组逐任务六面板见[图表索引](图表索引.md)，全部PNG/PDF。','',
    '## 范围与限制','','本轮沿用已见开发池，不能证明对独立区域有效；U4需先做训练角色/空间重叠审计。直方图采用边缘分布，未建模完整联合分布；单个正确/错误参考仍可能多峰、稀缺，跨地域参考状态本身会漂移。分布混合假设被破坏时，仅调整比例无法保证识别真实错误。所有结果包括失败项一并保留；没有看目标真值再调参数。保留模型权重并不自动证明其贡献，必须通过同容量无权重消融。']
    (c.BASE/'实验结果分析.md').write_text('\n'.join(lines)+'\n')
    idx=['# U6 图表索引','','主验证：每点4图，预测先封存，真值仅用于独立评价。']
    for scen in ['sensor_GBA','sensor_XJ','region_S2','region_L8']:idx+=['',f'- [{scen} PNG](artifacts/figures/paper_{scen}.png) / [PDF](artifacts/figures/paper_{scen}.pdf)']
    for name in ['error_bias','selection','source_lambda','error_proportions']:idx+=['',f'- [{name} PNG](artifacts/figures/{name}.png) / [PDF](artifacts/figures/{name}.pdf)']
    idx+=['','## 全部50任务']
    for p in sorted((c.OUT/'figures').glob('task_*.png')):idx+=['',f'- [{p.stem} PNG](artifacts/figures/{p.name}) / [PDF](artifacts/figures/{p.stem}.pdf)']
    (c.BASE/'图表索引.md').write_text('\n'.join(idx)+'\n')
