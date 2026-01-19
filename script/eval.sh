#/bin/bash
# logname="20240229_eval.log"
result_path="E:\Yiling\at_SIAT_research\z_result\20240418_eval_each_class_noweight_scheduler5"
logname_pre="20240418_eval_"

path_dwq_l8="E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_landsat8\train_val"
path_dwq_s2="E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val"
path_xj_l8="E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_landsat8\train_val"
path_xj_s2="E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val"

# model_dwq_l8="E:\Yiling\at_SIAT_research\z_result\20240307_1_train_landsat\20240307_train_dwq_landsat\unet_epoch37.pth"
model_dwq_s2="E:\Yiling\at_SIAT_research\z_result\20240412_1_train_scheduler_5\20240412_train_dwq_sentinel2\unet_epoch43.pth"
# model_xj_l8="E:\Yiling\at_SIAT_research\z_result\20240307_1_train_landsat\20240307_train_xj_landsat\unet_epoch78.pth"
model_xj_s2="E:\Yiling\at_SIAT_research\z_result\20240412_1_train_scheduler_5\20240412_train_xj_sentinel2\unet_epoch40.pth"

# python -u -W ignore eval.py > ./result/$logname
python -u -W ignore eval_each_class.py --data_path $path_dwq_s2 --load_model $model_dwq_s2 --log_name $logname_pre"dwqs2_to_dwqs2.log" --result_path $result_path
# python -u -W ignore eval_each_class.py --data_path $path_dwq_l8 --load_model $model_dwq_l8 --log_name $logname_pre"dwql8_to_dwql8.log" --result_path $result_path
python -u -W ignore eval_each_class.py --data_path $path_xj_s2 --load_model $model_xj_s2 --log_name $logname_pre"xjs2_to_xjs2.log" --result_path $result_path
# python -u -W ignore eval_each_class.py --data_path $path_xj_l8 --load_model $model_xj_l8 --log_name $logname_pre"xjl8_to_xjl8.log" --result_path $result_path

# counter=0
# for model in $model_dwq $model_xj
# do
#     for data in $path_dwq $path_xj
#     do
#         ((counter++))
#         python -u -W ignore eval.py --data_path $data --load_model $model --log_name "${logname_pre}${counter}.log"
#     done
# done
