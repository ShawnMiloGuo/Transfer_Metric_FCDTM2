# 'E:\Yiling\at_SIAT_research\z_result\20240613_visualization_binary\visualization_binary_xjs2_dice_cls2_label1_percent_train'
result_path_pre="E:\Yiling\at_SIAT_research\z_result\20250314_visualization_binary\20250314_1500_"
log_name_pre='visualization_binary_' #visualization_binary_dwq_sentinel2_cls1.log

data_path_pre="E:\Yiling\at_SIAT_research\1_dataset\dataset"
load_model_pre="E:/Yiling/at_SIAT_research/2_model_pth/train_" #\train_dwq_landsat8_cls_1\unet_epoch166_best_val.pth


for binary_class_index in 1 2 3 4 5 6 7 8
# for binary_class_index in 3
do
    for dataset_region in dwq xj
    do
        for dataset_sensor in sentinel2 landsat8
        do
            dataset_name=$dataset_region"_"$dataset_sensor
            data_path=$data_path_pre"\\"$dataset_name"\\train_val"
            # load_model=${load_model_pre}${dataset_name}"_cls_"$binary_class_index"\unet_epoch*_best_val.pth"
            load_model_pattern=${load_model_pre}${dataset_name}"_cls_"$binary_class_index"/unet_epoch*_best_val.pth"
            load_model=$(ls $load_model_pattern 2>/dev/null | head -n 1)
            log_name=$log_name_pre$dataset_name"_cls_"$binary_class_index".log"
            result_path=$result_path_pre$dataset_name"\\"${dataset_name}"_cls_"$binary_class_index

            echo $data_path
            echo $load_model
            echo $log_name
            echo $result_path
            
            python visualization_binary_multi_imgs.py \
            --binary_class_index $binary_class_index \
            --data_path $data_path \
            --load_model $load_model \
            --log_name $log_name \
            --result_path $result_path
        done
    done
done
# python visualization_binary_multi_imgs.py \
# --binary_class_index
# --data_path
# --load_model
# --log_name
# --result_path 