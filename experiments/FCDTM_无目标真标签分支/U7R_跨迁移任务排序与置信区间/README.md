# U7R 跨迁移任务排序与置信区间

回应Reviewer 2-3(b)。复用U7B已封存的40图任务分数，评分不读取目标标签或评价缓存，评价仅在任务分数再次封存后连接旧U7B评价缓存。没有重新微调、调公式、改符号或拟合任务AD。

## 阅读入口

- 实验结果分析.md：目的、主要结果、全部正式方法的D42区间、四场景、覆盖及局限。
- artifacts/跨任务排序统计.csv：2040条记录；3清单×5场景×2覆盖口径×2块类型，两个面板所有适用方法，含有效任务数及有效重复数。
- artifacts/任务级数据.csv：100个面板—任务记录；一个任务是类别、源模型和目标域，目标使用完整40图。不是四图批次或五个种子。
- artifacts/task_scores_sealed.csv、score_seal.json、score_audit.json：独立评分封存与禁止访问目标评价数据的防护。
- artifacts/score_native_audit.json、score_replay.strace：原生文件访问重放，与正式任务分数逐字节相同。
- artifacts/unit_stats.json、verification.json：独立计数及SciPy点估计／bootstrap核验。
- figures/：32组PNG/PDF、4册任务散点图集；index.json及annotation_audit.json记录全部正式方法和图表统计一致性。

## 实验协议

D42主分析、P37/E50敏感性；Full2048适用16项，Shared256适用18项。主CI为5000次类别块percentile区间，源模型块作为敏感性；至少3块和4000有限重复才给区间。置信区间是固定历史域条件下的描述性区间，跨类别共享影像的依赖不能完全消除，不能声称独立地区推广或方法间显著胜出。新分析前原生分数已固定；本轮不是盲测或前瞻性注册。

原表VII的选模实验移入论文补充Table S1，全部数值保留。新正文表VII为任务级AD排序，二者终点不同。8/16/24是补充选模的无标签图数；本轮任务排序固定40图。

## 复现

目录须与U7B同级，保留原始U7B封存输入。首次在新输出目录创建artifacts后依次运行code/unit_stats.py、score_tasks.py、evaluate_tasks.py、make_outputs.py；SciPy环境运行unit_stats.py --require-scipy和verify_results.py。已经存在score_seal.json时不得覆盖评分；历史评价和图表可从封存输入重建。评分脚本只做已存在40图原生分数的确定性投影，不调用GDAL或重新读目标参考。

ranking_stats.py实现的整数重采样系数已与字面重复数组、SciPy逐项比较。块抽样在所有方法间配对；不能把样本重复当独立新模型。common/own和两像素面板分别报告，不拼接有利结果。

论文修订和重建脚本保存在本地fcdtm-label-free目录；原稿不变，新增内容标红。实验代码及结果仅提交codex/fcdtm-label-free独立分支，main不变。
