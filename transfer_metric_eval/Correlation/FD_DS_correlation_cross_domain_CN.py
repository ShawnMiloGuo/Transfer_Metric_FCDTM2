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
import argparse

from matplotlib import rcParams
# 设置全局字体为 SimHei（黑体）
rcParams['font.family'] = 'SimHei'
# 设置负号正常显示
rcParams['axes.unicode_minus'] = False

np.random.seed(42)  # 设置随机种子

# Default font: ['DejaVu Sans']
# Default font size: 10.0

def load_data(
        transfer_metric_name="FD", # "DS" or "FD"
        task_transfer: str = "dwq_s2_xj_s2",
        batch_size: int = 4,
        parent_path: str = "",
        index_name_list: list = ['F1_delta'],
        save_path: str = "",
        ):
    dataset_name_source_list_dict = {"dwq_s2_xj_s2": ["dwq_sentinel2", "xj_sentinel2"],
                                    "dwq_s2_dwq_l8": ["dwq_sentinel2", "dwq_landsat8"],
                                    "dwq_l8_xj_l8": ["dwq_landsat8", "xj_landsat8"],
                                    "xj_s2_xj_l8": ["xj_sentinel2", "xj_landsat8"]}
    dataset_name_source_list = dataset_name_source_list_dict[task_transfer]
    dataset_name_target_list = dataset_name_source_list[::-1]

    # 设置源数据路径
    last_dir_dict = {
        "FD": {1: "4_FD_label1_all-batch1_100img",
               4: "8_FD_label1_all-batch4_100img"},
        "DS": {1: "2_DS_all-batch1_100img_by_pred0",
               4: "4_DS_all-batch4_100img_by_pred0"},
        "GBC": {1: "1_GBC_-_all-batch1_100img",
                4: "2_GBC_-_all-batch4_100img"},
        }
    last_dir = last_dir_dict[transfer_metric_name][batch_size]
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

    # draw_scatter_pre, 重新绘制散点图
    draw_scatter_pre(transfer_metric_name=transfer_metric_name,
                        index_name_list=index_name_list,
                        task_transfer=task_transfer,
                        result_list_dict=result_list_dict,
                        result_list_name=result_list_name,
                        dataset_name_source_list=dataset_name_source_list,
                        dataset_name_target_list=dataset_name_target_list,
                        batch_size=batch_size,
                        result_path=save_path,
                        )

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
    return df_dict, result_list_name, dataset_name_source_list, dataset_name_target_list

def draw_scatter_all_batch(result_list: list,
                           x_col = 19, y_col=9,
                           y_label = 'y_label', 
                           x_title = "mean_difference", y_title = "acc",
                           result_path = "./", figname = "test.png",
                           save_fig = True,
                            transfer_metric_name = "FD",
                            color = None,
                           ):
    # Create a new figure
    # plt.figure()
    # scatter
    # Extract the metric column as x
    if transfer_metric_name == "FD":
        x = [row[x_col] for row in result_list]
    elif transfer_metric_name == "DS":
        x = [-row[x_col] for row in result_list]
    elif transfer_metric_name == "GBC":
        x = [-row[x_col] for row in result_list]
    else:
        raise ValueError("transfer_metric_name should be 'DS' or 'FD' or 'GBC'")
    # Extract the acc columns as y
    y = [row[y_col] for row in result_list]
    # size_point = max(2, min(1000.0/len(x), 20))
    # size_point = 10
    size_point = 16
    print(f"draw_scatter(): size_point = {size_point}")

    replacements = {
        "background": "背景",
        "Cropland": "耕地",
        "Forest": "森林",
        "Grassland": "草地",
        "Shrubland": "灌木",
        "Wetland": "湿地",
        "Water": "水体",
        "Built-up": "建筑",
        "Bareland": "荒地"
    }

    # 替换 y_label
    for old, new in replacements.items():
        y_label = y_label.replace(old, new)
    
    plt.scatter(x, y, label=y_label, s=size_point, alpha=0.8, color=color)
    # title
    # fontsize_1 = 16
    fontsize_1 = 22
    plt.xlabel(x_title, fontsize=fontsize_1)
    plt.ylabel(y_title, fontsize=fontsize_1)
    # fontsize_2 = 14
    fontsize_2 = 16
    plt.xticks(fontsize=fontsize_2)
    plt.yticks(fontsize=fontsize_2)
    
    # Add a legend
    fontsize_3 = 20
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize_3)


    # save the plot
    # plt.show()
    if save_fig:
        test_path_exist(result_path)
        plt.savefig(os.path.join(result_path, figname), bbox_inches='tight')
    # Close the figure
    # plt.close()   

def draw_scatter(range_metric_i, range_accuracy_i,
                 dataset_i_list,
                 binary_class_index_list, binary_class_name_list,
                 result_list_dict, result_list_name,
                 dataset_name_source_list, dataset_name_target_list,
                 batch_size, result_path,
                 transfer_metric_name,):
    # draw_scatter_all-batch
    for metric_i in tqdm(range_metric_i, desc="draw_scatter"):
        for accuracy_i in range_accuracy_i:
            # # Create a new figure
            # plt.figure()
            num_datasets = 20
            colors = plt.cm.tab20(np.linspace(0, 1, num_datasets))  # 使用 'tab20' 颜色映射
            color_index = 0
            for dataset_i in dataset_i_list:
                # Create a new figure
                plt.figure()
                save_fig = False
                for binary_class_index in binary_class_index_list:
                    if transfer_metric_name == "DS":
                        result_list = result_list_dict[f"{dataset_name_source_list[dataset_i]}_cls_{binary_class_index}"]
                    else:
                        result_list = result_list_dict[f"{dataset_name_source_list[dataset_i]}-{dataset_name_target_list[dataset_i]}_cls_{binary_class_index}"]
                    # save the last figure
                    if binary_class_index == binary_class_index_list[-1]:
                        save_fig = True
                    # rename y_label, x_title, y_title
                    source_domain_name = dataset_name_source_list[dataset_i]
                    target_domain_name = dataset_name_target_list[dataset_i]
                    replacements1 = {
                        # "dwq": "GBA",
                        # "xj": "XJ",
                        "dwq": "大湾区",
                        "xj": "新疆",
                        "sentinel2": "S2",
                        "landsat8": "L8",
                    }
                    for old, new in replacements1.items():
                        source_domain_name = source_domain_name.replace(old, new)
                        target_domain_name = target_domain_name.replace(old, new)
                    x_title_name = result_list_name[metric_i]
                    y_title_name = "F1分数精度损失" # Accuracy Drop
                    replacements2 = {
                        # 16 mean_dif_absolute_sum
                        # 17 mean_dif_absolute_abs_sum
                        # 20 mean_dif_absolute_y0_y1_diff
                        # 36 FD_sum
                        # 20 target_Dispersion_score 
                        # 22 target_Dispersion_score_weighted
                        # 20 spherical_GBC
                        "mean_dif_absolute_sum": "Feature difference",
                        "mean_dif_absolute_abs_sum": "Feature difference + abs",
                        "mean_dif_absolute_y0_y1_diff": "本研究",
                        "FD_sum": "FD",
                        "target_Dispersion_score": "DS",
                        "target_Dispersion_score_weighted": "Weighted DS",
                        "spherical_GBC": "GBC",
                    }
                    for old, new in replacements2.items():
                        x_title_name = x_title_name.replace(old, new)

                    # 后缀
                    fig_suffix = ".svg" # ".png"
                    draw_scatter_all_batch(result_list, x_col=metric_i, y_col=accuracy_i,
                                        #    y_label=f"{dataset_name_source_list[dataset_i]}-{dataset_name_target_list[dataset_i]}_cls-{binary_class_index}-{binary_class_name_list[binary_class_index]}",
                                           y_label=f"{source_domain_name} —> {target_domain_name}  {binary_class_name_list[binary_class_index]}",
                                           x_title=x_title_name, y_title=y_title_name,
                                           result_path=os.path.join(result_path, "fig_scatter"),
                                           figname=f"draw_{dataset_i}_{dataset_name_source_list[dataset_i]}-{dataset_name_target_list[dataset_i]}_cls-{binary_class_index}-{binary_class_name_list[binary_class_index]}_batch{batch_size}_{result_list_name[metric_i]}_{result_list_name[accuracy_i]}" + fig_suffix,
                                           save_fig=save_fig,
                                           transfer_metric_name=transfer_metric_name,
                                           color = colors[color_index],
                                           )
                    color_index = (color_index + 1) % num_datasets
                plt.close()

def draw_scatter_pre(transfer_metric_name,
                     index_name_list,
                     task_transfer,
                     result_list_dict, result_list_name,
                     dataset_name_source_list, dataset_name_target_list,
                     batch_size, result_path
                     ):
    if transfer_metric_name == "FD":
        # range_metric_i = range(16, len(result_list_name))
        # 16 mean_dif_absolute_sum
        # 17 mean_dif_absolute_abs_sum
        # 18 mean_dif_relative_sum
        # 19 mean_dif_relative_abs_sum --
        # 20 mean_dif_absolute_y0_y1_diff
        # 21 mean_dif_absolute_abs_y0_y1_diff --
        # 22 mean_dif_relative_y0_y1_diff
        # 36 FD_sum
        range_metric_i = [16, 17, 18, 20, 22, 36]
    elif transfer_metric_name == "DS":
        # range_metric_i = range(19, 23) 
        # 20 target_Dispersion_score 
        # 22 target_Dispersion_score_weighted
        range_metric_i = [20, 22]
    elif transfer_metric_name == "GBC":
        # range_metric_i = range(19, 21)
        # 19 diagonal_GBC --
        # 20 spherical_GBC
        range_metric_i = [20]
    else:
        raise ValueError("transfer_metric_name should be 'DS' or 'FD' or 'GBC'")
    # 根据name找到index，放入range_metric_i和range_accuracy_i
    range_accuracy_i = [result_list_name.index(name) for name in index_name_list]

    dataset_i_list = [0, 1]
    binary_class_index_list_dict = {"dwq_s2_xj_s2": [1, 2, 3, 6, 7, 8],
                                    "dwq_s2_dwq_l8": [1, 2, 6, 7, 8],
                                    "dwq_l8_xj_l8": [1, 6, 7, 8],
                                    "xj_s2_xj_l8": [1, 3, 5, 6, 7, 8]}
    binary_class_index_list = binary_class_index_list_dict[task_transfer]
    binary_class_name_list = ["background", "Cropland", "Forest", "Grassland", "Shrubland", "Wetland", "Water", "Built-up", "Bareland"]
    
    draw_scatter(range_metric_i, 
                 range_accuracy_i,
                 dataset_i_list,
                 binary_class_index_list, 
                 binary_class_name_list,
                 result_list_dict, 
                 result_list_name,
                 dataset_name_source_list, 
                 dataset_name_target_list,
                 batch_size, 
                 result_path,
                 transfer_metric_name,
                 )

def save_csv(
        df_dict, result_list_name, 
        dataset_name_source_list, dataset_name_target_list,
        task_transfer, transfer_metric_name, 
        index_name, # F1_delta
        correlation_method, # pearson, spearman
        batch_size,
        save_path,
        ):
    index_name_index = result_list_name.index(index_name)
    selected_columns = [result_list_name[index_name_index]]
    if transfer_metric_name == "FD":
        selected_columns = selected_columns + result_list_name[16:]
    elif transfer_metric_name == "DS":
        selected_columns = selected_columns + result_list_name[19:]
    elif transfer_metric_name == "GBC":
        selected_columns = selected_columns + result_list_name[19:]
    else:
        raise ValueError("transfer_metric_name should be 'DS' or 'FD' or 'GBC'")
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
    heatmap_data = data.iloc[:, 1:]
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

def get_args():
    parser = argparse.ArgumentParser(description='Correlation Analysis')
    parser.add_argument('--transfer_metric_name', type=str, default="FD", help='transfer metric name: FD, DS, GBC')
    parser.add_argument('--save_path', type=str, 
                        default=r"E:\Yiling\at_SIAT_research\z_result\20250331_correlation_cross_domain\20250331_1703_MMM\1_dwq_s2_xj_s2_", 
                        help='save path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # get args
    args = get_args()

    # todo
    # 定义度量指标、迁移任务、数据路径、结果存放路径
    transfer_metric_name = args.transfer_metric_name # "DS" "FD" "GBC"
    task_transfer_list = ["dwq_s2_xj_s2", "dwq_l8_xj_l8", "dwq_s2_dwq_l8", "xj_s2_xj_l8"]

    if transfer_metric_name == "FD":
        save_path_ori = args.save_path.replace("MMM", "FD")
        data_path_ori = r"E:\Yiling\at_SIAT_research\z_result\20241220_transfer_metric_FD_cross_sensor_batch14\20241220_1655_1_dwq_s2_xj_s2_"
        # index_name_list = ['OA_delta', 'F1_delta', 'precision_delta', 'OA_delta_relative', 'F1_delta_relative', 'precision_delta_relative',]
        index_name_list = ['F1_delta']
    elif transfer_metric_name == "DS":
        save_path_ori = args.save_path.replace("MMM", "DS")
        data_path_ori = r"E:\Yiling\at_SIAT_research\z_result\20241219_transfer_metric_Ds_cross_domain\20241219_1756_DS_1_dwq_s2_xj_s2_"
        # index_name_list = ['OA_t', 'F1_t', 'miou_t', 'precision_t', 'recall_t', 'OA_delta', 'F1_delta', 'miou_delta', 'precision_delta', 'recall_delta',]
        # index_name_list = ['F1_t', 'F1_delta']
        index_name_list = ['F1_delta']
    elif transfer_metric_name == "GBC":
        save_path_ori = args.save_path.replace("MMM", "GBC")
        data_path_ori = r"E:\Yiling\at_SIAT_research\z_result\20250312_transfer_metric_GBC_batch14\20250312_0944_1_dwq_s2_xj_s2_"
        index_name_list = ['F1_delta']
    else:
        raise ValueError("transfer_metric_name should be 'DS' or 'FD' or 'GBC'")

    for i, task_transfer in enumerate(task_transfer_list):
        parent_path = data_path_ori.replace("1_dwq_s2_xj_s2", f"{i+1}_{task_transfer}")
        save_path = save_path_ori.replace("1_dwq_s2_xj_s2", f"{i+1}_{task_transfer}")
        save_log(result_path=save_path, 
                 log_name="default.log",)
        for batch_size in [1, 4]:
            # load data
            df_dict, result_list_name, dataset_name_source_list, dataset_name_target_list = load_data(
                transfer_metric_name=transfer_metric_name,
                task_transfer=task_transfer,
                batch_size=batch_size,
                parent_path=parent_path,
                index_name_list=index_name_list,
                save_path=save_path,)
            # save .csv
            for correlation_method in ["pearson", "spearman"]:
                for index_name in tqdm(index_name_list, desc=f"{transfer_metric_name}_{i}/{len(task_transfer_list)}-b{batch_size}-{correlation_method}"):
                    df_corr_F1_diff, save_file_prefix = save_csv(
                        df_dict, result_list_name, 
                        dataset_name_source_list, dataset_name_target_list,
                        task_transfer, transfer_metric_name, 
                        index_name, # F1_delta
                        correlation_method, # pearson, spearman
                        batch_size,
                        save_path=os.path.join(save_path, "csv"),
                        )
                    # plot the correlation matrix heatmap
                    draw_heatmap(
                        data=df_corr_F1_diff, 
                        save_path=os.path.join(save_path, "fig"),
                        save_file_prefix=save_file_prefix,
                        correlation_method=correlation_method)