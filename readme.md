说明文件创建时间：2025-06-24
作者：ZT

# 概述
研究方向：可迁移性度量
代码包含的主要功能：
1. 训练卫星影像二分类语义分割模型
2. 计算可迁移性度量分数，以及对应的迁移后精度损失。包含部分现有研究（FD，DS，GBC，OTCE等），和本研究提出的方法
3. 可迁移性度量方法的验证评估。即计算可迁移性度量分数与迁移后精度损失的相关性
4. 模型输出可视化。将测试集影像输入到已训练好的模型中，导出预测结果

# 文件组织说明
注意：常用和重要文件，高亮显示。

code    代码根目录
│  class_counts.py  在多分类数据集中，计算每个类别的像素占整个数据集的比例
│  class_counts_binary.py   在二分类数据集中，批量计算多个二分类数据集的各个类别像素占比
│  eval.py  用已训练好的模型对测试集进行测试，输出测试精度
│  eval_each_class.py   用已训练好的模型对测试集进行测试，输出每个类别的测试精度
│  eval_each_class_binary.py    用已训练好的二分类模型对测试集进行测试，输出每个类别的测试精度。可用于迁移任务的精度测试。
│  note.md  文件更改日志
│  readme.md    此说明文件
│  train.ipynb  模型训练代码的测试版
│  train.py     多分类模型训练代码
│  train_binary.py  ==二分类模型训练代码，通常配合/script/train.sh脚本批量训练模型==
│  
├─component 代码所需组件
│  │  dice_score.py 计算dice loss
│  │  evaluation_index.py   使用CPU计算各个精度指标
│  │  evaluation_index_gpu.py   使用GPU加速计算各个精度指标
│  │  evaluation_index_gpu_binary.py    使用GPU加速计算各个精度指标，适用于二分类任务
│  │  evaluation_index_gpu_each_class.py    计算各个类别的多个精度指标
│  │  utils.py  包含早停机制、计算数据集各类别权重、重定向控制台输出、保存日志等
│  │  
│  ├─dataset    数据集读取实现
│  │  dataset.py 正常数据集读取的实现
│  │  dataset_with_name.py   数据集读取时返回单张图片的名字，适用于模型测试可视化
│  │  data_from_gdal.py  读取tif文件的函数
│  │          
│  └─model  网络框架
│     ├─unet    unet网络
│     │     unet_layer.txt  unet各层信息
│     │     unet_model.py   unet原始网络
│     │     unet_model_zt.py    unet网络更改
│     │     unet_parts.py   unet网络部分组件
│     │          
│     └─unetplusplus    unet++网络
│           unetplusplus_model.py
│          
├─draw  绘制数据集对比图
│      data_dwq_l8.csv
│      data_dwq_s2.csv
│      data_xj_l8.csv
│      data_xj_s2.csv
│      draw_class_counts_binary.py
│      dwq_l8_binary_percent_16x5.png
│      dwq_s2_binary_percent_16x5.png
│      test.ipynb
│      xj_l8_binary_percent_16x5.png
│      xj_s2_binary_percent_16x5.png
│      
├─feature   重要！！！可迁移性度量实验的主要实现代码
│  │  draw_distribution_of_a_dataset.py     绘制一个数据集的多维特征分布
│  │  draw_distribution_of_transfer.py      绘制迁移任务的多维特征分布
│  │  draw_line_of_weight_bias.py       绘制分割头64维权重和偏置
│  │  extract_feature.py    测试文件，提取特征
│  │  extract_feature_all-batch.py  测试文件，提取特征
│  │  metric_gbc.py     GBC方法实现代码
│  │  metric_otce.py    OTCE方法实现代码
│  │  print_model_weight.py 打印模型权重
│  │  note.md   实验日志
│  │  transfer_metric_FD.sh ==重要！！！可迁移性度量实验脚本，批量运行实验==
│  │  transfer_metric_FD_DS.py  ==重要！！！可迁移性度量实验代码，配合脚本transfer_metric_FD.sh使用==
│  │  transfer_metric_FD_DS_20250527.py 可迁移性度量实验代码的旧版本备份
│  │  transfer_metric_separability.sh   可迁移性度量实验脚本，针对可分离性的实验
│  │      
│  ├─old    可迁移性度量实验代码的旧版本备份
│  │      20241210_transfer_metric_FD_mask0_all_all_label1_no_feature0.py
│  │      readme.md
│  │      transfer_metric_all-all.py
│  │      transfer_metric_all-batch.py
│  │      transfer_metric_FD_all-all.py
│  │      transfer_metric_FD_all-all_label1.py
│  │      transfer_metric_FD_all-all_label1_no-feature0.py
│  │      transfer_metric_FD_DS allbatch.py
│  │      transfer_metric_FD_DS batch.py
│  │      transfer_metric_FD_mask0_all_all_label1_no_feature0.py
│  │      transfer_metric_separability.py
│  │      transfer_metric_separability_Ds.py
│  │      
│  ├─result     存放脚本文件的输出日志
│  │      
│  ├─result_fig     测试过程保存的图片
│  │      
│  ├─script     实验后处理的脚本
│  │      extract_from_transfer_metric_FID_weighted_debug_log.py    批量提取需要的指标等数据
│  │      select_from_all_fig.py    挑选符合条件的图片，整合到新的目录
│  │      sort_from_all_fig_DS.py   对DS方法，将可迁移性度量实验的绘图结果按照网格排列，形成一张大图，方便对比
│  │      sort_from_all_fig_FD.py   对FD方法，将可迁移性度量实验的绘图结果按照网格排列，形成一张大图，方便对比
│  │      
│  ├─test   功能或新特性测试，不重要
│  │      
│  └─transfer_metric_draw   可迁移性度量实验结果绘制的测试，不重要
│     └─result_fig  存放绘图测试结果
│      
├─old   旧版本，不重要
├─ppt   为制作答辩PPT，绘图的代码
│      draw.ipynb   PPT绘图代码
│      
├─result    存放脚本文件的日志
│      
├─script    脚本文件
│      eval.sh  多分类测试脚本
│      eval_binary.sh   二分类测试脚本
│      eval_binary_transfer.sh  二分类迁移实验测试脚本
│      extract_from_class-counts-binary_log.py  提取需要的指标等数据
│      extract_from_eval-each-class-binary_log.py   提取需要的指标等数据
│      extract_from_transfer_metric_log.py  提取数据
│      move_pdfs.py 提取PDF文件
│      shtest.sh    脚本功能测试
│      train.sh     ==重要！！！训练二分类模型的脚本==
│      
├─transfer_metric_eval  可迁移性度量实验评估
│  ├─Correlation    相关性评估
│  │  │  DS_F1_t_correlation_cross_domain.py    DS方法与目标域F1分数的相关性评估代码
│  │  │  DS_F1_t_correlation_cross_domain.sh    DS方法与目标域F1分数的相关性评估代码，执行脚本
│  │  │  DS_script_csv.py   DS相关性评估指标，csv文件收集
│  │  │  FD_DS_correlation_cross_domain.py  ==重要！！！FD、DS、GBC等方法与目标域精度损失的相关性评估代码==
│  │  │  FD_DS_correlation_cross_domain.sh  ==重要！！！对应相关性评估代码的执行脚本==
│  │  │  FD_DS_correlation_cross_domain_CN.py   相关性评估代码的中文绘图版本
│  │  │  FD_script_csv.py   FD相关性评估指标，csv文件收集
│  │  │  test.ipynb 功能测试
│  │  │  
│  │  └─old 旧版本文件
│  │          
│  └─OA 不重要，相关性热力图功能测试代码
│      │  correlation_FD.ipynb  相关性热力图绘制
│      │  correlation_test.ipynb    相关性热力图绘制测试
│      │  result_list_name_FD   导出的result_list_name词典内容
│      │  
│      └─FD 不重要，存放相关性数据的测试结果
│              
├─visualization ==重要！！！可视化代码==
│      note.md    代码更改日志
│      visualization_binary.ipynb   可视化功能测试
│      visualization_binary.py  对每张原图（4通道），生成1张包含4个子图的可视化图片
│      visualization_binary_multi_imgs.py   对每张原图（4通道），生成4张可视化图片
│      visualization_binary_multi_imgs.sh   对应的批处理脚本


