
save_path="E:\Yiling\at_SIAT_research\z_result\20250419_DS_F1t_correlation_cross_domain\202504301650_MMM\1_dwq_s2_xj_s2_"

# for transfer_metric_name in "FD" "DS" "GBC"
for transfer_metric_name in "DS" "GBC"
do
    python DS_F1_t_correlation_cross_domain.py \
    --transfer_metric_name $transfer_metric_name \
    --save_path $save_path
done
