## 覆盖与采样比较补记

Full2048的D42/P37/E50共同支持分别为42/42、37/37、50/50；Shared256为40/42、36/37、48/50。E50两面板均包含全部8类。Shared256缺少的两个共同任务都是建设用地：新疆L8→大湾区L8，以及新疆S2→新疆L8；该256点目标样本不支持硬前景、DS或GBC的有限分数，不是按真实目标F1剔除。软FCDTM及部分其他方法的own支持仍完整保留。

Full与Shared的像素集合和共同任务数均不同，不能把排序差异全部归因于像素数量，更不能把Full里有利的FCDTM值与Shared的MMD值直接排名。

## 数值核验依据

Kendall采用tau-b并列修正，Spearman采用平均并列秩；整数权重只表示bootstrap时某任务重复出现的次数，不是改用SciPy的weighted-tau指标。400组独立小数组的显式成对计数／字面重复样本检查通过；服务器SciPy对2040条实际统计点估计、1280个实际bootstrap系数核对，误差不超过2.3e-16。任务图右上角与统计表同源核验，NCE的常量N/A没有被填0。

定义依据：[SciPy kendalltau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)、[SciPy spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)。本轮不使用小样本下的默认渐近p值宣称优势。
