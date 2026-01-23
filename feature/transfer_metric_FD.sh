#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

sh_logname="20260123_transfer_metric_FD_DS.log"
# result_path_pre="E:\Yiling\at_SIAT_research\z_result\20250312_transfer_metric_GBC_batch14\20250527_1543"
result_path_pre="/Volumes/Untitled/results/20260123/1_FD"

transfer_metric_name="FD" # GBC FD DS
# dwq_s2_xj_s2, dwq_s2_dwq_l8, dwq_l8_xj_l8, xj_s2_xj_l8
task_transfer="dwq_s2_xj_s2"


order=0
all_or_batch="all"
batch_size=4
# s_to_t="dwqs2-xjs2"
img_num=100

no_feature0=0
only_label_1=0
feature_layer_name="up4"
# log_name="transfer_metric_FD_.log"

by_pred=0
by_pred_gt="by_gt"

for transfer_metric_name in "DS" "GBC"
do
    if [ $transfer_metric_name == "DS" ]
    then
        result_path_pre="/Volumes/Untitled/results/20260123/2_DS"
    else
        result_path_pre="/Volumes/Untitled/results/20260123/3_GBC"
    fi

    for by_pred in 0 1
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
                    label_index="-"
                else
                    label_index="label1"
                fi

                # target_domain_all
                # for target_domain_all in 1 0
                for target_domain_all in 0
                do
                    if [ $target_domain_all == 1 ]
                    then
                        all_or_batch="all"
                    else
                        all_or_batch="batch"$batch_size
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
                        result_path=$result_path_pre"_"$task_order"_"$task_transfer"_\\"$order"_"$transfer_metric_name"_"$label_index"_all-"$all_or_batch"_"$img_num"img_"$by_pred_gt
                        log_name="transfer_metric_"$transfer_metric_name"_"$task_transfer".log"
                        echo $result_path

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