# U8 全输入OTCE可扩展近似（用户已确认）

正式标注OTCE-PL-SrcCal-Q1024，不能声称全点精确EMD。所有Full2048源/目标像素均参与。固定源特征MiniBatchKMeans 1024个cell（不读取类别标签拟合），种子20261123+class*100+domain_index；n_init=1、batch4096、max_iter50、max_no_improvement10、reassignment0.01。源/目标逐像素赋予源cell；各记录只汇总其实际图像的cell计数、特征和/平方范数及源真类/目标伪类计数。目标cell均值随该批实际样本计算，不能利用其他目标图的统计。

在非空cell重心上用POT 0.9.6.post1 network-simplex求解不等质量的精确运输；以计数/全部点数为边缘概率。完整近似plan为各cell对内的均匀乘积提升。W取此plan在全部点上的精确平方成本：重心运输成本+源/目标cell内方差总和；这是可行近似plan成本（全点最优成本的上界），不是全点精确最优W。用源/目标cell类别分布与plan计算完整2×2联合分布，再算CE。保留边缘质量误差、cell内离散程度及点数。

数值单测：无压缩6点对精确指派、4对6点对linprog、压缩cell对显式全点plan的W/CE及边缘质量；源域8个check辅助任务各抽固定256对精确EMD比较W/CE误差，报告但不据目标标签改Q。近似误差不保证CE同样有界，不隐瞒对排序的影响。

源校准使用既定16fit/8check辅助图清单和源F1下降；源参考/辅助均使用Full2048全部像素，辅助目标类别使用模型预测。标准化W/CE后岭回归lambda1e-3+截距，不沿用Shared256系数，不用目标标签选超参。源check仅诊断，不调公式/方向。评分入口禁读目标标签/评价，输出后统一封存。
