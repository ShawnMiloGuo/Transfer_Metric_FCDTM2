#/bin/bash
logname="20240621_eval_each_class_binary_dice_gt5.log"
result_path="E:\Yiling\at_SIAT_research\z_result\20240620_eval_binary_landsat\20240621_eval_each_class_binary_dice_gt5"
logname_pre="20240621_eval_"

# path_dwq_s2="E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val"
# path_xj_s2="E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val"
path_dwq_l8="E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_landsat8\train_val"
path_xj_l8="E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_landsat8\train_val"

label_1_percent=0.05

# python -u -W ignore eval.py > ./result/$logname
for cls in {1..8}
do
    echo "cls: $cls"
    # model_dwq_s2="E:/Yiling/at_SIAT_research/z_result/20240606_2_train_binary_dice_gt1/train_dwq_sentinel2_cls_${cls}/*best_val.pth"
    # model_xj_s2="E:/Yiling/at_SIAT_research/z_result/20240606_2_train_binary_dice_gt1/train_xj_sentinel2_cls_${cls}/*best_val.pth"
    model_dwq_l8="E:/Yiling/at_SIAT_research/z_result/20240618_train_landsat/20240621_train_binary_dice_gt5/train_dwq_landsat8_cls_${cls}/*best_val.pth"
    model_xj_l8="E:/Yiling/at_SIAT_research/z_result/20240618_train_landsat/20240621_train_binary_dice_gt5/train_xj_landsat8_cls_${cls}/*best_val.pth"

    # echo "model_dwq_s2: ${model_dwq_s2}"
    # echo "model_xj_s2: ${model_xj_s2}"

    python -u -W ignore eval_each_class_binary.py \
        --data_path $path_dwq_l8 --load_model $model_dwq_l8 \
        --log_name $logname_pre"dwql8-cls${cls}_to_dwql8-cls${cls}.log" \
        --result_path $result_path \
        --binary_class_index $cls --label_1_percent $label_1_percent >> ./result/$logname
    python -u -W ignore eval_each_class_binary.py \
        --data_path $path_xj_l8 --load_model $model_xj_l8 \
        --log_name $logname_pre"xjl8-cls${cls}_to_xjl8-cls${cls}.log" \
        --result_path $result_path \
        --binary_class_index $cls --label_1_percent $label_1_percent >> ./result/$logname
done


# model_dwq_s2="E:/Yiling/at_SIAT_research/z_result/20240423_4_train_binary_noweight/train_dwq_sentinel2_cls_1/*.pth"
# echo $model_dwq_s2
# for file in $model_dwq_s2
# do
#     echo $file
# done
