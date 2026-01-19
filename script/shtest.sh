# counter=0
# logname_pre="eval_"
# for i in {1..10}
# do
#     ((counter++))
#     echo "Current iteration: $i, name: ${logname_pre}${counter}"
# done

# model_dwq_s2="E:/Yiling/at_SIAT_research/z_result/20240423_3_train_binary_weight/train_dwq_sentinel2_cls_1/*.pth"
# echo $model_dwq_s2

# for cls in {1..8}
# do
#     echo "cls: ${cls}_to_${cls}"
# done

# for cls in {1..8}
# do
#     echo "cls: $cls"
#     model_dwq_s2="E:/Yiling/at_SIAT_research/z_result/20240423_3_train_binary_weight/train_dwq_sentinel2_cls_${cls}/*best_val.pth"
#     echo ${model_dwq_s2}
# done

all_or_batch="---"
for target_domain_all in 0 1
do
    if [ $target_domain_all == 1 ]
    then
        all_or_batch="all"
    else
        all_or_batch="batch4"
    fi
    echo $all_or_batch
done