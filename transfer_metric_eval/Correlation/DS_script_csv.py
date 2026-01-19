import os
import pandas as pd
import sys
# 获取当前脚本所在目录的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)
from component.utils import save_log, test_path_exist

# todo
# 定义度量指标、迁移任务
transfer_metric_name="GBC" # "DS" or "FD" or "GBC"
task_transfer_list = ["dwq_s2_xj_s2", "dwq_l8_xj_l8", "dwq_s2_dwq_l8", "xj_s2_xj_l8"]
# E:\Yiling\at_SIAT_research\z_result\20241226_FD_correlation_cross_domain\20241226_1800\1_dwq_s2_xj_s2_\csv
if transfer_metric_name == "FD":
    save_path_ori = r"E:\Yiling\at_SIAT_research\z_result\20241226_FD_correlation_cross_domain\20241226_1800\1_dwq_s2_xj_s2_\csv"
    combined_path = r"E:\Yiling\at_SIAT_research\z_result\20241226_FD_correlation_cross_domain\20241226_1800\20241227_1730_combined_csv"
    index_name_list = ['OA_delta', 'F1_delta', 'precision_delta', 'OA_delta_relative', 'F1_delta_relative', 'precision_delta_relative',]
elif transfer_metric_name == "DS":
    save_path_ori = r"E:\Yiling\at_SIAT_research\z_result\20250419_DS_F1t_correlation_cross_domain\20250419_2244_DS\1_dwq_s2_xj_s2_\csv"
    combined_path = r"E:\Yiling\at_SIAT_research\z_result\20250419_DS_F1t_correlation_cross_domain\20250419_2244_DS\20250419_2254_combined_csv"
    index_name_list = ['F1_t']
elif transfer_metric_name == "GBC":
    save_path_ori = r"E:\Yiling\at_SIAT_research\z_result\20250419_DS_F1t_correlation_cross_domain\20250419_2244_GBC\1_dwq_s2_xj_s2_\csv"
    combined_path = r"E:\Yiling\at_SIAT_research\z_result\20250419_DS_F1t_correlation_cross_domain\20250419_2244_GBC\20250419_2254_combined_csv"
    index_name_list = ['F1_t']

save_log(combined_path)
dataset_name_source_list_dict = {"dwq_s2_xj_s2": ["dwq_sentinel2", "xj_sentinel2"],
                                "dwq_s2_dwq_l8": ["dwq_sentinel2", "dwq_landsat8"],
                                "dwq_l8_xj_l8": ["dwq_landsat8", "xj_landsat8"],
                                "xj_s2_xj_l8": ["xj_sentinel2", "xj_landsat8"]}
all_csv_files_list = []
for index_name in index_name_list:
    for correlation_method in ["pearson", "spearman"]:
        for batch_size in [1, 4]:
            csv_files_list = []
            for i, task_transfer in enumerate(task_transfer_list):
                dataset_name_source_list = dataset_name_source_list_dict[task_transfer]
                dataset_name_target_list = dataset_name_source_list[::-1]
                save_path = save_path_ori.replace("1_dwq_s2_xj_s2", f"{i+1}_{task_transfer}")
                save_file_prefix = f"{transfer_metric_name}_" + \
                        f"{dataset_name_source_list[0]}-{dataset_name_target_list[0]}_" + \
                        f"batch{batch_size}_" + \
                        f"{correlation_method}_" + \
                        f"{index_name}_"
                csv_name = save_file_prefix + "corr_heatmap_data.csv"
                csv_path = os.path.join(save_path, csv_name)
                csv_files_list.append(csv_path)
                all_csv_files_list.append(csv_path)
            df_list = []
            for csv_file in csv_files_list:
                print(f"Reading {csv_file}")
                df = pd.read_csv(csv_file)
                df_list.append(df)
            combined_df = pd.concat(df_list, ignore_index=False)
            output_csv_prefix = f"{transfer_metric_name}_" + \
                        f"batch{batch_size}_" + \
                        f"{correlation_method}_" + \
                        f"{index_name}_"
            output_csv_path = os.path.join(combined_path, output_csv_prefix + "corr_heatmap_data.csv")
            combined_df.to_csv(output_csv_path, index=True)
            print(f"Combined CSV saved to {output_csv_path}")
def rename_csv_name(csv_file):
    csv_name = os.path.basename(csv_file)
    csv_name = csv_name.replace("dwq_sentinel2", "")
    csv_name = csv_name.replace("xj_sentinel2", "")
    csv_name = csv_name.replace("dwq_landsat8", "")
    csv_name = csv_name.replace("xj_landsat8", "")
    csv_name = csv_name.replace("_-_", "_")
    csv_name = csv_name.replace("corr_heatmap_data.csv", ".")
    return csv_name
all_df_mean_list = []
all_df_abs_mean_list = []
for csv_file in all_csv_files_list:
    print(f"Reading {csv_file}")
    df = pd.read_csv(csv_file)
    csv_name = rename_csv_name(csv_file)
    row_mean = df.iloc[[-2]].copy()
    row_mean.iloc[0, 0] = csv_name + row_mean.iloc[0, 0]
    row_abs_mean = df.iloc[[-1]].copy()
    row_abs_mean.iloc[0, 0] = csv_name + row_abs_mean.iloc[0, 0]
    all_df_mean_list.append(row_mean)
    all_df_abs_mean_list.append(row_abs_mean)
combined_all_df_mean = pd.concat(all_df_mean_list, ignore_index=False)
combined_all_df_abs_mean = pd.concat(all_df_abs_mean_list, ignore_index=False)
output_csv_path_mean = os.path.join(combined_path, f"{transfer_metric_name}_all_corr_heatmap_data_mean.csv")
output_csv_path_abs_mean = os.path.join(combined_path, f"{transfer_metric_name}_all_corr_heatmap_data_abs_mean.csv")
combined_all_df_mean.to_csv(output_csv_path_mean, index=True)
print(f"Combined all CSV saved to {output_csv_path_mean}")
combined_all_df_abs_mean.to_csv(output_csv_path_abs_mean, index=True)
print(f"Combined all CSV saved to {output_csv_path_abs_mean}")

