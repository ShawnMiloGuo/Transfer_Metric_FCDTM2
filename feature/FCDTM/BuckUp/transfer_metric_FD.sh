#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

sh_logname="20260123_transfer_metric_FD_DS.log"

transfer_metric_name="FD" # GBC FD DS
# dwq_s2_xj_s2, dwq_s2_dwq_l8, dwq_l8_xj_l8, xj_s2_xj_l8
task_transfer="dwq_s2_xj_s2"


order=0
all_or_batch="all"
batch_size=4
# s_to_t="dwqs2-xjs2"
img_num=100 #图片数量？

no_feature0=0
only_label_1=0
feature_layer_name="up4"
# log_name="transfer_metric_FD_.log"

by_pred=0
by_pred_gt="by_gt"

for transfer_metric_name in "FD" "DS" "GBC"
do
    if [ $transfer_metric_name == "FD" ]
    then
        result_path_pre="/home/Shanxin.Guo/ZhangtuosCode/code/Transfer_Metric_FCDTM/result/1_FD"
    fi
    if [ $transfer_metric_name == "DS" ]
    then
        result_path_pre="/home/Shanxin.Guo/ZhangtuosCode/code/Transfer_Metric_FCDTM/result/2_DS"
    fi
    if [ $transfer_metric_name == "GBC" ]
    then
        result_path_pre="/home/Shanxin.Guo/ZhangtuosCode/code/Transfer_Metric_FCDTM/result/3_GBC"
    fi

    for by_pred in 0 1 # 是用目标域真实标签作度量还是用模型预测的作度量，0 代表真实标签，1代表模型预测标签
    do
        by_pred_gt="by_pred"$by_pred
        # batch_size 1 4
        for batch_size in 1 4
        do
            # label_index
            # for only_label_1 in 0 1 # FD
            for only_label_1 in 0 # GBC
            do
                if [ $only_label_1 == 0 ]
                then
                    label_index="-" # 同时计算前景和背景类
                else
                    label_index="label1" # 只计算前景类的标签
                fi

                # target_domain_all
                # for target_domain_all in 1 0
                for target_domain_all in 0
                do
                    if [ $target_domain_all == 1 ]
                    then
                        all_or_batch="all" # 目标域所有的数据计算度量
                    else
                        all_or_batch="batch"$batch_size # 按照batchSize来在每个batch上进行度量
                    fi

                    ((order++))
                    # if [ $order -gt 4 ]
                    # then
                    #     continue
                    # fi
                    if [ $batch_size == 4 ] && [ $target_domain_all == 1 ]
                    then
                        continue
                    fi

                    # task_transfer
                    task_order=0
                    for task_transfer in "dwq_s2_xj_s2" "dwq_l8_xj_l8" "dwq_s2_dwq_l8" "xj_s2_xj_l8"
                    do
                        ((task_order++))
                        result_path=$result_path_pre"_"$task_order"_"$task_transfer"/"$order"-"$transfer_metric_name"_"$label_index"all-"$all_or_batch"_"$img_num"img_"$by_pred_gt
                        log_name="transfer_metric_"$transfer_metric_name"_"$task_transfer".log"
                        echo $result_path

                        args=" --batch_size "$batch_size" --target_domain_all "$target_domain_all" --no_feature0 "$no_feature0" --feature_layer_name "$feature_layer_name" --only_label_1 "$only_label_1" --result_path "$result_path" --log_name "$log_name" --task_transfer "$task_transfer" --transfer_metric_name "$transfer_metric_name" --by_pred "$by_pred 
                        echo $args
                        # python -u -W ignore transfer_metric_FD_DS.py --batch_size 1 --target_domain_all 0 --no_feature0 0 --feature_layer_name up4 --only_label_1 0 --result_path /home/Shanxin.Guo/ZhangtuosCode/code/Transfer_Metric_FCDTM/result/1_FD_1_dwq_s2_xj_s2/1-FD_-all-batch1_100img_by_pred0 --log_name transfer_metric_FD_dwq_s2_xj_s2.log --task_transfer dwq_s2_xj_s2 --transfer_metric_name FD --by_pred 0

                        python -u -W ignore transfer_metric_FD_DS.py \
                        --batch_size $batch_size \
                        --target_domain_all $target_domain_all \
                        --no_feature0 $no_feature0 \
                        --feature_layer_name $feature_layer_name \
                        --only_label_1 $only_label_1 \
                        --result_path $result_path \
                        --log_name $log_name \
                        --task_transfer $task_transfer \
                        --transfer_metric_name $transfer_metric_name \
                        --by_pred $by_pred \

                        >> ./result/$sh_logname
                    done
                done
            done
        done
    done
done