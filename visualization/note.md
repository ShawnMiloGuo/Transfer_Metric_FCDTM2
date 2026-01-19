# 20250314 
为了往大论文里加数据集的图，
需要实现对所有数据集的可视化，
包括卫星图像、标签、预测，还要根据精度对图片命名，方便按精度排序。

将 `code/visualization_binary.py` 移入 `code/visualization/visualization_binary.py`。
复制 `visualization_binary.py` 到 `visualization_binary_multi_imgs.py`。

原文件 `visualization_binary.py ` 的功能如下：
加载 xj_sentinel2 的二分类数据集 cls2 和模型，
输出一张1×4的可视化图，从左到右依次为 Img_rgb、Img_nir、Ground Truth、Prediction。
命名包含精度等。

新文件 `visualization_binary_multi_imgs.py` 需要实现以下功能：
批量加载所有二分类数据集和模型，
输出多张可视化的图，包括 1img_rgb、1img_ngb、2gt、3pred
命名包含精度，F1分数排在最前边，图片类型包括 1img_rgb、1img_ngb、2gt、3pred
