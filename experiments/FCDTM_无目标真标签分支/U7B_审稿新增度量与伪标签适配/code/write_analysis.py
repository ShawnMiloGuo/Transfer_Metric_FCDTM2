from pathlib import Path
import csv,json,math,statistics,collections
B=Path(__file__).resolve().parents[1];O=B/'artifacts'
def read(n):return list(csv.DictReader((O/n).open()))
S=read('相关汇总_full.csv');C=read('模型选择宏汇总_full.csv');T=read('逐任务相关_full.csv');R=read('逐项评价_full.csv');A=read('ATC原生accuracy误差.csv');E=json.loads((O/'evaluation_full.json').read_text());M=json.loads((O/'manifest_full.json').read_text());assert E['seal_verified']
LABEL={'FCDTM_HardPL':'FCDTM硬伪标签','FCDTM_SoftPL':'FCDTM软伪标签','FD_global':'FD','DS_pixel':'DS','GBC_spherical_PL':'GBC-PL','AC':'AC／MSP','Entropy':'平均熵','ATC_MC':'ATC-MC','ATC_NE':'ATC-NE','DoC':'DoC-gap','DoC_Acc_SrcCal':'DoC-Acc-SrcCal','LogME_PL':'LogME-ICML-PL','Hscore_PL':'H-score-PL','LEEP_PL':'LEEP-PL','NCE_Pred_PL':'NCE-Pred-PL','NLEEP_PL':'NLEEP-PL','OTCE_PL_SrcCal':'OTCE-PL-SrcCal','MMD_RBF_global':'MMD-RBF-global'}
def f(x,n=4):return f'{float(x):.{n}f}' if math.isfinite(float(x)) else 'N/A'
def s(k,p='full2048',scen='all',cohort='D42'):return next(r for r in S if r['method']==k and r['panel']==p and r['cohort']==cohort and r['scenario']==scen and r['scope']=='common')
def c(k,p='full2048',cohort='D42'):return next(r for r in C if r['method']==k and r['panel']==p and r['cohort']==cohort)
soft=s('FCDTM_SoftPL');sel=c('FCDTM_SoftPL');fd=c('FD_global')
text=f'''# U7B 实验结果分析：不看目标真标签时，与正式方法相比怎么样

## 这次实验要验证什么

不是再发明一个容易被FCDTM超过的基线，而是把原论文FD、DS、GBC和审稿人建议的已发表方法都放到同一套无目标真标签条件下比较。要回答两件事：①固定一个模型，度量能否判断它在哪些目标图组上精度下降更多；②有几个候选模型时，能否选到目标F1更高的模型。两者必须分开看。

本轮完成25模型、8类别、50方向任务、两种像素面板共6600条记录；D42是原论文声明的42任务主口径，P37是原主表37项，E50是全部可运行任务。4图批次×10组×5预定种子；另有8／16／24张**无标签**图的候选选择。它们不是少量目标标注预算。历史图池不是新独立地区。

## 先读最关键的数值

Full2048主口径中，软FCDTM的批次Fisher相关为 **{f(soft['Fisher_r'])}**，Spearman为 **{f(soft['rho'])}**。在同一选择配置上，它的平均遗憾为 **{f(sel['regret'])}**，Top-1为 **{f(100*float(sel['Top1']),1)}%**；FD对应 **{f(fd['regret'])}**、**{f(100*float(fd['Top1']),1)}%**。不能只用相关性高就宣称候选筛选更好，更不能把相关系数当成AD预测误差。

所有正式方法的完整结果如下；应先看同一面板内的结果，再看可计算覆盖，不能把两种像素预算混为一个榜单。

'''
for p in ['full2048','shared256']:
 q=c('FCDTM_SoftPL',p);v=s('FCDTM_SoftPL',p)
 text+='## '+p+'：全部方法\n\n'
 text+=('每图先取2048有效像素；此面板16项，OTCE／MMD不伪造全量结果。' if p=='full2048' else '源参考和每条目标记录各统一取256点，18项都使用同样的像素。256是计算上限，不是图数。')+'\n\n'
 text+=f"D42共同有限批次点：{v['common_points']}/2100；共同选模配置：{q['common_configs']}/{q['total_configs']}，覆盖{q['target_class_groups']}个类别—目标组。N/A和常量会使相关分母不同，以下同时列出定义成功的任务—种子数。选择配置不是独立区域。\n\n"
 text+='|方法|Fisher r↑|Spearman↑|有定义相关／210|平均遗憾↓|Top-1↑|最大预期损失↓|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for k in E['methods'][p]:
  a=s(k,p);b=c(k,p)
  text+=f"|{LABEL[k]}|{f(a['Fisher_r'])}|{f(a['rho'])}|{a['finite_correlations']}/210|{f(b['regret'])}|{f(100*float(b['Top1']),1)}%|{f(b['max_loss'])}|\n"
 text+='\n该表的r按任务做Fisher变换汇总后，再对5种子平均；Spearman按任务平均。相关未定义的任务不被写成0，完整own/common覆盖见CSV。遗憾先平均重复抽样，再对预算及类别—目标组等权；最大预期损失在配置级取最大。恒分数并列按等概率选择，另存最差并列损失。\n\n'
ss=s('FCDTM_SoftPL','shared256');cc=c('FCDTM_SoftPL','shared256');mm=c('MMD_RBF_global','shared256')
text+=f'''## 最值得保留的结论与容易误读之处

本轮Full2048中，软FCDTM的平均批次关联高于新增伪标签对照；Shared256中也有正关联，但r降至{f(ss['Fisher_r'])}，且与DS的差距较小。只有256个像素时稀有前景容易不受支持，覆盖也下降；不能把它当成标签完全无关的稳健性已经解决。

选模方面，在Shared256同一共同配置内，MMD的平均遗憾为{f(mm['regret'])}、Top-1为{f(100*float(mm['Top1']),1)}%，软FCDTM为{f(cc['regret'])}、{f(100*float(cc['Top1']),1)}%。因此即使补齐新增方法，结论仍是：FCDTM较擅长同一模型的批次变化指示，尚未证明更擅长跨候选选择。优势终点与失败终点都必须进入返修。

AC和DoC-gap在固定模型内只差一个源置信度常量，方向统一后相关相同是预期的；跨模型的常量不同，因此候选选择可以不同。二分类下最大置信度与负熵单调对应，本轮ATC-MC和ATC-NE得到相同结果，也不是两份独立成功证据。保留这些条目是为了完整对应已发表方法及审稿建议，而不是扩大样本数量。

'''
text+='## 跨传感器与跨地域分别看\n\n|面板／场景|软FCDTM r|硬FCDTM r|H-score-PL r|NLEEP-PL r|AC r|FD r|\n|---|---:|---:|---:|---:|---:|---:|\n'
for p in ['full2048','shared256']:
 for scen in ['sensor_GBA','sensor_XJ','region_S2','region_L8']:
  text+='|'+p+'／'+scen+'|'+'|'.join(f(s(k,p,scen)['Fisher_r']) for k in ['FCDTM_SoftPL','FCDTM_HardPL','Hscore_PL','NLEEP_PL','AC','FD_global'])+'|\n'
text+='\n这些是分组描述，不是传感器差异的因果证明。传感器／地域、前景占比、伪标签错误、候选来源及特征尺度可能同时变化，不能凭一张相关表断言某个机制已经被证实。所有场景都保留，不能事后只选有利组回应审稿。\n\n'
text+='## 全部8类是否都有测试\n\n有。以下按E50汇总每类任务，方法、种子和图组不变；每类的源候选数量不同。r仍是先按种子聚合可定义任务，再平均种子；它不是单个多分类网络的整体精度。\n\n|类别|面板|软FCDTM r|H-score-PL r|NLEEP-PL r|FD r|\n|---|---|---:|---:|---:|---:|\n'
classes=['','耕地','森林','草地','灌木','湿地','水体','建设用地','裸地'];classrows=[]
for cl in range(1,9):
 for p in ['full2048','shared256']:
  vals=[]
  for k in ['FCDTM_SoftPL','Hscore_PL','NLEEP_PL','FD_global']:
   by=collections.defaultdict(list)
   for r in T:
    if r['panel']==p and r['scope']=='common' and int(r['cls'])==cl and r['method']==k and math.isfinite(float(r['Pearson'])):by[r['seed']].append(float(r['Pearson']))
   z=[math.tanh(statistics.mean(math.atanh(max(-1+1e-7,min(1-1e-7,x))) for x in vv)) for vv in by.values()];value=statistics.mean(z) if z else math.nan;vals.append(value);classrows.append(dict(cls=cl,category=classes[cl],panel=p,method=k,Fisher_r=value,finite_task_seeds=sum(map(len,by.values()))))
  text+='|'+classes[cl]+'|'+p+'|'+'|'.join(f(x) for x in vals)+'|\n'
with (O/'全类别相关摘要.csv').open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=list(classrows[0]));w.writeheader();w.writerows(classrows)
text+='''
## 各方法到底用了什么

- FD、MMD：看整体源—目标特征分布差异，不用目标类别标签；本轮不是完整AutoEval训练回归系统。
- DS：看模型预测出的两组特征是否分开；它本身就是预测分组。
- GBC-PL、LogME-ICML-PL、H-score-PL、LEEP-PL、NLEEP-PL：原版需要目标标签，这里统一用当前候选自己的硬预测替代，因此名称必须保留PL，不能宣称原版无需标签。
- AC：平均最大置信度；Entropy：平均预测不确定性。DoC-gap：源、目标平均置信度之差；DoC-Acc-SrcCal：仅源辅助任务拟合出的置信差到像素accuracy回归。两者分开保留。
- ATC-MC／NE：用源标签定置信／负熵阈值，再在目标只数超过阈值的比例。它们估计像素accuracy，不能当成前景F1。
- OTCE-PL-SrcCal：最优运输距离与耦合条件熵同时进入仅源域的组合校准，不再用只有距离的占位实现。源派生四图任务拟合的是F1下降；没有目标真标签调系数。
- FCDTM硬／软：保留原稿的前景均值相对差和分类头权重，只把目标真掩膜换成预测／概率。源前景均值仍使用允许访问的源标签。

## 为什么表里会有0或N/A

NCE-Pred-PL比较的是同一套自伪标签，条件熵严格为0，所以得分恒0、相关N/A、选模全并列。这是该适配的数学退化，不是没运行，也不是精度下降预测正确。LEEP／NLEEP在只有一个预测类别时也严格得0；只消除浮点行和造成的微小假变化，并保留原始浮点分数。GBC接近0也可能是很小的有效类重叠，不能一概当失败。

缺少预测前景、源前景不受样本支持、Gaussian组样本不足，或者算法异常，才保留不可计算标志。没有按目标真值挑像素、补前景或把异常补0。共同配置上的好坏是有条件结论，必须同时看覆盖；各方法自身覆盖也在CSV中。DS的无穷极限在选择时保留其定义上的顺序，相关分析仍要求有限点。

## 图怎么看，与表格怎么对应

横轴是预先规定方向的原生分数，越右表示预期AD越大；纵轴是真实F1下降，只在封存后用真值算。每个点是四张图，不是一个像素，也不是一个独立新地区。颜色表示方向与地物类别。

右上角第一行r_F,5和rho_5与相应场景表格相同；第二行pool是图上第一个固定种子全部散点合起来算的相关，两者不同不代表图表错误。另列点数及有定义的任务—种子数。不能为图好看而挑最佳种子、翻相关符号或隐藏不利方法。全部18项按五组方法族展示，软FCDTM重复作为参照；各组不是不同试验样本。

artifacts/figures提供D42／E50、Full2048／Shared256、四场景全部图。Full2048不含OTCE／MMD，Shared256全部18项；两个面板分图分表。与论文配套的图册包含每一项，不只挑三种方法。

## 原生accuracy误差，不能当F1误差

|面板|方法|像素accuracy MAE↓|偏差|记录数|\n|---|---|---:|---:|---:|\n'''
for r in A:text+=f"|{r['panel']}|{r['method']}|{f(r['MAE'])}|{f(r['bias'])}|{r['records']}|\n"
non=fail=0
for r in R:
 d=json.loads(r['diagnostics']).get('NLEEP_PL',{})
 if d.get('converged') is False:non+=1
 if 'error' in d:fail+=1
text+=f'''\nNLEEP全部6600条中，达到最大迭代仍未收敛标志的记录为{non}，异常记录为{fail}。未收敛但有限的末次结果按固定协议保留；异常不补0。DoC允许源校准外推，不根据目标结果裁剪。

## 能回答哪些审稿问题，不能回答哪些

本轮完成了“加入已发表正式对照、核清标签要求、禁止目标真值进入评分、明确像素适配、全类别并列展示”的证据。它不能把原本需目标标签的方法宣称成原版无标签算法，也不能用自伪标签退化说明原有标签方法没有价值。

保留原FCDTM思想的事实由公式说明，但**分类头加权本身是否产生独立增益，仍须U7C加权／无权重同协议消融**。只看当前相关表不能完成这条论证。历史池不是新的空间独立确认；DeepLabv3+等新架构也尚未运行，属于U7D。统一给所有原生分数加同容量F1／AD源校准的额外误差实验亦未纳入本轮，不可把原生相关结果写成该项已完成。

下一步应先冻结U7C权重消融、伪标签来源及稀有类支持检查，再用未参与设计的空间网格和新架构确认。若某些已发表对照表现更好，应保留并据实限定FCDTM结论，不能换更弱基线或按目标结果调整参数后再称独立验证。

## 执行与可信性记录

25个冻结模型先评分、逐个原生strace审计，再合并封存6600记录；源域DoC回归与单伪类别数值修补均不读取目标参考，各自保留旧封存并重新封存。独立评价器核对封存后才打开目标真值。源标注白名单仅为当前候选64张源图；不使用其他候选源域的标签训练当前校准器。源辅助fit／check图组不重叠，但两者共享固定64图源参考锚点，因此check仅是源内诊断，不是整套数据隔离的外测。全部8类的批次测试也不表示每一类都有多个候选可做选模；选模只统计实际存在多候选的类别—目标组。

GPU初始化失败后使用CPU统一新预测；只复用本轮CPU生成且核验过的缓存。CPU并行加载停滞改为主线程预载、每模型串行评分，四模型并发。一个原生审计通过的已完成CPU模型保留。未重置显卡、改旧算法或训练数据。修复不改变公式、种子、图像预算与标签边界。代码、设备、缓存SHA、原始分数、审计及补充协议均保留，正式结果不混用旧GPU评价值。
'''
(B/'实验结果分析.md').write_text(text)
print('Wrote complete plain-language analysis and eight-class summary')
