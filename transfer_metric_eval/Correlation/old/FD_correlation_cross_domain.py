import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
# 获取当前脚本所在目录的父目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
# 将父目录添加到 sys.path
sys.path.append(parent_dir)
from component.utils import save_log, test_path_exist
from tqdm import tqdm

def load_data_FD(task_transfer: str = "dwq_s2_xj_s2",
              batch_size: int = 4,
              parent_path: str = "",
              ):
    dataset_name_source_list_dict = {"dwq_s2_xj_s2": ["dwq_sentinel2", "xj_sentinel2"],
                                    "dwq_s2_dwq_l8": ["dwq_sentinel2", "dwq_landsat8"],
                                    "dwq_l8_xj_l8": ["dwq_landsat8", "xj_landsat8"],
                                    "xj_s2_xj_l8": ["xj_sentinel2", "xj_landsat8"]}
    dataset_name_source_list = dataset_name_source_list_dict[task_transfer]
    dataset_name_target_list = dataset_name_source_list[::-1]
    # batch_size = 4

    # 设置源数据路径
    if batch_size == 1:
        last_dir = "4_FD_label1_all-batch1_100img"
    elif batch_size == 4:
        last_dir = "8_FD_label1_all-batch4_100img"
    else:
        raise ValueError("batch_size should be 1 or 4")
    # E:\Yiling\at_SIAT_research\z_result\20241220_transfer_metric_FD_cross_sensor_batch14\20241220_1655_1_dwq_s2_xj_s2_\4_FD_label1_all-batch1_100img
    # E:\Yiling\at_SIAT_research\z_result\20241220_transfer_metric_FD_cross_sensor_batch14\20241220_1655_1_dwq_s2_xj_s2_\8_FD_label1_all-batch4_100img
    # parent_path = r"E:\Yiling\at_SIAT_research\z_result\20241220_transfer_metric_FD_cross_sensor_batch14\20241220_1655_1_dwq_s2_xj_s2_"
    result_path = os.path.join(parent_path, last_dir)

    # load the result_list_dict
    result_list_dict_name = f"result_list_dict_{dataset_name_source_list[0]}-{dataset_name_target_list[0]}_batch{batch_size}.json"
    result_list_dict_path = os.path.join(result_path, result_list_dict_name)
    with open(result_list_dict_path, 'r') as f:
        result_list_dict = json.load(f)
    # load the result_list_name
    result_list_name_path = os.path.join(result_path, "result_list_name.json")
    with open(result_list_name_path, "r") as file:
        result_list_name = json.load(file)
    print(len(result_list_name))
    for i in range(len(result_list_name)):
        print(i, result_list_name[i])

    df_dict = {}
    print("result_list_dict.keys()")
    print("key, \tlen(result_list_dict[key]), \tlen(result_list_dict[key][0]), \tsource, \ttarget")
    for key in result_list_dict.keys():
        source = result_list_dict[key][0][0]
        target = result_list_dict[key][0][1]
        print(f"{key}, \t{len(result_list_dict[key])}, \t{len(result_list_dict[key][0])}, \t{source}, \t{target}")
        key_new = key.replace(f"{source}_cls", f"{source}-{target}_cls")
        df_dict[key_new] = pd.DataFrame(result_list_dict[key], columns=result_list_name)
    for key in df_dict.keys():
        print(key, df_dict[key].shape)
    return df_dict, result_list_name, dataset_name_source_list, dataset_name_target_list,
def save_csv(df_dict, result_list_name, 
             dataset_name_source_list, dataset_name_target_list,
             task_transfer, transfer_metric_name, 
             index_name, # F1_delta
             correlation_method, # pearson, spearman
             batch_size,
             save_path,
             ):
    # 从 OA_delta 往后取
    selected_columns = result_list_name[4:10] + result_list_name[16:]
    df_corr_F1_diff = pd.DataFrame(columns=selected_columns)
    # index_name = "F1_delta" # F1_delta
    # correlation_method = "pearson" # pearson, spearman
    for key in df_dict.keys():
        temp_df_select = df_dict[key][selected_columns]
        temp_correlation_matrix = temp_df_select.corr(method=correlation_method) # method='pearson' 'spearman' 'spearman'
        df_corr_F1_diff.loc[f"{key}_{index_name}"] = temp_correlation_matrix[index_name]
    # abs, mean
    df_corr_F1_diff.loc[f"{task_transfer}_mean"] = df_corr_F1_diff.mean()
    df_corr_F1_diff.loc[f"{task_transfer}_abs_mean"] = df_corr_F1_diff.abs().mean()

    print(df_corr_F1_diff.iloc[:, :2])

    # save .csv
    save_file_prefix = f"{transfer_metric_name}_" + \
        f"{dataset_name_source_list[0]}-{dataset_name_target_list[0]}_" + \
        f"batch{batch_size}_" + \
        f"{correlation_method}_" + \
        f"{index_name}_"
    csv_name = save_file_prefix + "corr_heatmap_data.csv"
    test_path_exist(save_path)
    df_corr_F1_diff.to_csv(os.path.join(save_path, csv_name), index=True)
    return df_corr_F1_diff, save_file_prefix
# plot the correlation matrix heatmap
def draw_heatmap(data, 
                 save_path, 
                 save_file_prefix,
                 correlation_method):
    heatmap_data = data.iloc[:, :]
    figsize_scale = 4
    div_long_height = heatmap_data.shape[1] / heatmap_data.shape[0]
    figsize_long = heatmap_data.shape[1]
    print(f"figsize_long: {figsize_long}, figsize_long / div_long_height: {figsize_long / div_long_height}")
    plt.figure(figsize=(figsize_long , figsize_long / div_long_height))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix Heatmap ({correlation_method})')
    print(f"heatmap_data.shape: {heatmap_data.shape}")
    # plt.show()
    fig_name = save_file_prefix + "corr_heatmap"
    test_path_exist(save_path)
    plt.savefig(os.path.join(save_path, f"{fig_name}.png"), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 定义度量指标、迁移任务
    transfer_metric_name="FD" # "DS" or "FD"
    task_transfer_list = ["dwq_s2_xj_s2", "dwq_l8_xj_l8", "dwq_s2_dwq_l8", "xj_s2_xj_l8"]
    save_path_ori = r"E:\Yiling\at_SIAT_research\z_result\20241224_FD_correlation_cross_domain\20241224_1953\1_dwq_s2_xj_s2_"

    for i, task_transfer in tqdm(enumerate(task_transfer_list)):
        data_path_ori = r"E:\Yiling\at_SIAT_research\z_result\20241220_transfer_metric_FD_cross_sensor_batch14\20241220_1655_1_dwq_s2_xj_s2_"
        parent_path = data_path_ori.replace("1_dwq_s2_xj_s2", f"{i+1}_{task_transfer}")
        save_path = save_path_ori.replace("1_dwq_s2_xj_s2", f"{i+1}_{task_transfer}")
        save_log(result_path=save_path, 
                 log_name="default.log",)
        for batch_size in [1, 4]:
            df_dict, result_list_name, dataset_name_source_list, dataset_name_target_list = load_data_FD(task_transfer=task_transfer,
                                                                                                        batch_size=batch_size,
                                                                                                        parent_path=parent_path)
            # save .csv
            for correlation_method in ["pearson", "spearman"]:
                for index_name in ['OA_delta', 'F1_delta', 'precision_delta', 'OA_delta_relative', 'F1_delta_relative', 'precision_delta_relative',]: # 'OA_delta', 'F1_delta', 'precision_delta', 'OA_delta_relative', 'F1_delta_relative', 'precision_delta_relative',
                    df_corr_F1_diff, save_file_prefix = save_csv(df_dict, result_list_name, 
                                                                dataset_name_source_list, dataset_name_target_list,
                                                                task_transfer, transfer_metric_name, 
                                                                index_name, # F1_delta
                                                                correlation_method, # pearson, spearman
                                                                batch_size,
                                                                save_path=os.path.join(save_path, "csv"),
                                                                )
                    # plot the correlation matrix heatmap
                    draw_heatmap(data=df_corr_F1_diff, 
                                save_path=os.path.join(save_path, "fig"),
                                save_file_prefix=save_file_prefix,
                                correlation_method=correlation_method)

        

