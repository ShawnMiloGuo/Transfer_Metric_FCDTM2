# U7A 原论文方法的无目标真实标签正式对比

本实验恢复论文FCDTM、FD、DS、GBC对照体系：FCDTM硬/软伪标签、全局FD、修正平方L2后的DS、球形GBC伪标签及对角敏感性。不是原版有标签方法的复现，不以自建概率代理代替正式对照。

- 协议.md：定义、采样、种子与标签隔离。
- code/common.py、worker.py、run.py：纯分数计算、image-only目标推理与审计封存。
- code/verify.py：不读目标标签的充分统计量复算及独立Scipy FD表达式检查。
- code/evaluate.py、post_stats.py、plots.py：封存后独立评价、排序与描述性模型簇区间及论文式图表。
- artifacts/manifest.json、seal.json：25模型/50任务、输入与代码哈希和预测封存。
- artifacts/audit_*.json、logs/*.strace：源白名单及原生文件访问证据。

统一4图子域、5个预定seed，2500条批次记录；附加800条8/16/24/40图池记录。主清单正文补齐42项，原表37项和全类别50项另外报告。原生分数没有AD单位，不直接计算AD MAE。

类别编号：1耕地，2森林，3草地，4灌木，5湿地，6水体，7建设用地，8裸地。S2为Sentinel-2，L8为Landsat 8，dwq为大湾区，xj为新疆。每任务图用第一预定seed，未挑最佳seed。

完整运行在服务器FCDTM环境：先运行code/run.py（含单元核验，禁止覆盖既有manifest）；全部成功/审计通过才封存。已有seal时不得重跑预测；只可从已封存缓存重新核验/评价。首次评价与补充Scipy核验、GBC近零统计修正的实际时序另文记录，不把历史开发池包装成独立盲法确认。

后续U7B审稿新增对照、U7C深入消融与Holm配对检验、U7D新架构/独立网格不属于本次已完成范围，状态见TODO。
