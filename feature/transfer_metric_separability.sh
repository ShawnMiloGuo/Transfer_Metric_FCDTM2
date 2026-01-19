#/bin/bash
sh_logname="20241219_1756_transfer_metric_DS_cross_domain.log"
# z_result\20240627_transfer_metric_FID-mask0\20240627_transfer_metric_FID_decoder_1_all-all_dwqs2-xjs2_100img
result_path_pre="E:\Yiling\at_SIAT_research\z_result\20241219_transfer_metric_Ds_cross_domain\20241219_1756_DS"

# dwq_s2_xj_s2, dwq_s2_dwq_l8, dwq_l8_xj_l8, xj_s2_xj_l8
task_transfer="dwq_s2_xj_s2"
transfer_metric_name="DS"


order=0
all_or_batch="all"
batch_size=4
# s_to_t="dwqs2-xjs2"
img_num=100

by_pred=0
by_pred_gt="by_gt"

# log_name="transfer_metric_GBC_dwqs2-xjs2.log"
# log_name="transfer_metric_weighted_"$score_name"_"$s_to_t".log"

for by_pred in 0 1
do
    by_pred_gt="by_pred"$by_pred
    # batch_size 1 4
    for batch_size in 1 4
    do
        for target_domain_all in 1 0
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
            # task_transfer
            task_order=0
            for task_transfer in "dwq_s2_xj_s2" "dwq_l8_xj_l8" "dwq_s2_dwq_l8" "xj_s2_xj_l8"
            do
                ((task_order++))
                result_path=$result_path_pre"_"$task_order"_"$task_transfer"_\\"$order"_"$transfer_metric_name"_all-"$all_or_batch"_"$img_num"img_"$by_pred_gt
                log_name="transfer_metric_"$transfer_metric_name"_"$task_transfer".log"
                echo $result_path

                python -u -W ignore transfer_metric_FD_DS.py \
                --batch_size $batch_size \
                --target_domain_all $target_domain_all \
                --result_path $result_path \
                --log_name $log_name \
                --by_pred $by_pred \
                --task_transfer $task_transfer \
                --transfer_metric_name $transfer_metric_name \
                >> ./result/$sh_logname
            done
        done
    done
done