# EN paper
# save_path="E:\Yiling\at_SIAT_research\z_result\20250603_EN_FD_F1d_correlation_cross_domain\20250603_1535_MMM_sampleAll_line\1_dwq_s2_xj_s2_"
save_path="E:\Yiling\at_SIAT_research\z_result\20250603_EN_FD_F1d_correlation_cross_domain\20250603_1614_MMM_sampleAll_by_pred0\1_dwq_s2_xj_s2_"

# for transfer_metric_name in "FD" "DS" "GBC"
for transfer_metric_name in "FD" "DS" "GBC"
do
    python FD_DS_correlation_cross_domain.py \
    --transfer_metric_name $transfer_metric_name \
    --save_path $save_path
done


# # CN thesis
# save_path="E:\Yiling\at_SIAT_research\z_result\20250501_FD_F1d_correlation_cross_domain\20250501_1640_MMM\1_dwq_s2_xj_s2_"

# for transfer_metric_name in "FD" "DS" "GBC"
# do
#     python FD_DS_correlation_cross_domain_CN.py \
#     --transfer_metric_name $transfer_metric_name \
#     --save_path $save_path
# done