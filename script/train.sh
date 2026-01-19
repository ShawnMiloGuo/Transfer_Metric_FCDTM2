#/bin/bash
# lr=0.001
# for multi in 1
# do
# 	logname="20230720_train_SZ91_buildings_M"$multi"_lr"$lr".log"
# 	python -u -W ignore train_muti_104_windows.py --multiclass $multi --l_rate $lr > ./result/$logname
# done
# for lr in 0.0005 0.0003 0.0001
# do
#     result_path_lr="E:\Yiling\at_SIAT_research\z_result\20240308_2_train_landsat\20240308_train_xj_landsat_lr"$lr
#     python -u -W ignore train.py --n_epoch 1000 --data_path $data_path_xj --log_name $log_name_xj --result_path $result_path_lr --l_rate $lr >> ./result/$logname
# done

# dwq_landsat8
# dwq_sentinel2
# xj_landsat8
# xj_sentinel2


logname="20240621_train_binary_dice_gt5.log"

# data_path_dwq_s2="E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val"
# data_path_xj_s2="E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val"
data_path_dwq_l8="E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_landsat8\train_val"
data_path_xj_l8="E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_landsat8\train_val"

# log_name_dwq_s2="train_dwq_sentinel2_cls_"
# log_name_xj_s2="train_xj_sentinel2_cls_"
log_name_dwq_l8="train_dwq_landsat8_cls_"
log_name_xj_l8="train_xj_landsat8_cls_"

# result_path_dwq_s2="E:\Yiling\at_SIAT_research\z_result\20240620_train_binary_CE_dice_gt5\train_dwq_sentinel2_cls_"
# result_path_xj_s2="E:\Yiling\at_SIAT_research\z_result\20240620_train_binary_CE_dice_gt5\train_xj_sentinel2_cls_"
result_path_dwq_l8="E:\Yiling\at_SIAT_research\z_result\20240618_train_landsat\20240621_train_binary_dice_gt5\train_dwq_landsat8_cls_"
result_path_xj_l8="E:\Yiling\at_SIAT_research\z_result\20240618_train_landsat\20240621_train_binary_dice_gt5\train_xj_landsat8_cls_"

# model_arch="unet"
# batch_size=16
# binary_class_index=1
n_classes=2
label_1_percent=0.05

# binary classification with weight
# for binary_class_index in {1..8}
for binary_class_index in 8
do
    # binary_class_index in 1--8
    python -u -W ignore train_binary.py \
     --data_path $data_path_dwq_l8 --log_name $log_name_dwq_l8$binary_class_index".log" --result_path $result_path_dwq_l8$binary_class_index \
     --n_classes $n_classes --binary_class_index $binary_class_index --loss_weight 0 --label_1_percent $label_1_percent \
     >> ./result/$logname
    python -u -W ignore train_binary.py \
     --data_path $data_path_xj_l8 --log_name $log_name_xj_l8$binary_class_index".log" --result_path $result_path_xj_l8$binary_class_index \
     --n_classes $n_classes --binary_class_index $binary_class_index --loss_weight 0 --label_1_percent $label_1_percent \
     >> ./result/$logname
done
