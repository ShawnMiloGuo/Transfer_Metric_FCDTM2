# 2024-08-19
目的：
对于类别可分离性的思想，尝试对每个维度进行加权。
首先要输出网络最后一层的权重信息，
查看每个类别的权重是否相关，大的都大？小的都小？
查看偏置参数，分析其作用。

```
[1,64,256,256] --> [1,2,256,256]
从64维特征到二分类的概率图，需要1x1的卷积，包含64x2+2=130个可学习参数。
```

实验方法：
先从 `code\feature\draw_distribution_of_a_dataset.py` 拷贝代码，命名为`code\feature\print_model_weight.py`。
删除多余部分，只保留输出权重的部分。

# 2024-09-04
目的：探究分割头参数的权重和偏置对分类结果的影响
实验设置：
对于一个指定的模型，根据其分割头的64维权重和偏置，分别绘制前景类和背景类的线性函数，
将64个函数图按照权重排序，观察有无规律？
实验方法：
从`code\feature\print_model_weight.py`和
`code\feature\draw_distribution_of_a_dataset.py`中，
整合出绘制权重偏置线性函数功能的代码，
命名为`code\feature\draw_line_of_weight_bias`。

# 2024-12-19
目的：
创建一个新的py文件，合并transfer_metric中 FD 和 Ds的计算

方法：
将`transfer_metric_FD_mask0_all_all_label1_no_feature0.py`文件复制一份，
命名为`transfer_metric_FD_Ds.py`，
根据`transfer_metric_separability_Ds.py`的内容对`transfer_metric_FD_Ds.py`文件的内容进行修改合并。
将`transfer_metric_FD_mask0_all_all_label1_no_feature0.py`，`transfer_metric_separability_Ds.py`移动到`./old`。

结果：
获得整合transfer_metric中 FD 和 Ds计算的代码，
即`transfer_metric_FD_Ds.py`。

# 2025-05-27
目的：
把度量操作中的标签换成伪标签重新画图。

方法：
备份之前的`transfer_metric_FD_DS.py`为`transfer_metric_FD_DS_20250527.py`。
修改`transfer_metric_FD_DS.py`，整理各个度量方法的计算方式，包括特征提取函数等。
删除多余的未用函数。