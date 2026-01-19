import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


dataset_name_source_list = ["dwq_sentinel2", "xj_sentinel2"]
dataset_name_target_list = dataset_name_source_list[::-1]
batch_size = 4
result_path = r"E:\Yiling\at_SIAT_research\z_result\20241025_transfer_metric_FD_weighted_y_diff\20241025_6_label1_FID_all-batch4_dwqs2-xjs2_100img"

# load the result_list_dict
result_list_dict_name = f"result_list_dict_{dataset_name_source_list[0]}-{dataset_name_target_list[0]}_batch{batch_size}.json"
result_list_dict_path = os.path.join(result_path, result_list_dict_name)
with open(result_list_dict_path, 'r') as f:
    result_list_dict = json.load(f)

for key in result_list_dict.keys():
    print(key, len(result_list_dict[key]), len(result_list_dict[key][0]))


with open("./result_list_name_FD", "r") as file:
    result_list_name = json.load(file)["result_list_name"]
print(len(result_list_name))
for i in range(len(result_list_name)):
    print(i, result_list_name[i])


df_dict = {}
for key in result_list_dict.keys():
    df_dict[key] = pd.DataFrame(result_list_dict[key], columns=result_list_name)

for key in df_dict.keys():
    print(key, df_dict[key].shape)


# 从 OA_delta 往后取
selected_columns = result_list_name[4:]

df_corr_F1_diff = pd.DataFrame(columns=selected_columns)

for key in df_dict.keys():
    temp_df_select = df_dict[key][selected_columns]
    temp_correlation_matrix = temp_df_select.corr() # method='pearson' 'spearman' 'spearman'
    df_corr_F1_diff.loc[key + "_F1_delta"] = temp_correlation_matrix["F1_delta"]

# abs, mean
df_corr_F1_diff.loc["mean"] = df_corr_F1_diff.mean()
df_corr_F1_diff.loc["abs_mean"] = df_corr_F1_diff.abs().mean()

print(df_corr_F1_diff.iloc[:, :2])


# plot the correlation matrix heatmap
heatmap_data = df_corr_F1_diff.iloc[:, :]
figsize_scale = 4
plt.figure(figsize=(8 * figsize_scale, 2.5 * figsize_scale))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()
print(heatmap_data.shape)


heatmap_data.to_csv("./FD/dwq_s2_xj_s2_F1_delta_heatmap_data.csv")