import os
import sys
from eval import eval
from eval import test_path_exist

import matplotlib.pyplot as plt

# autolog
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def plot_lines(data, names, save_path='./', fig_name='fig'):
    # 确保二维数组的行数和名字列表的长度相同
    assert len(data) == len(names), "The data and names must have the same length."

    # 创建一个新的图表
    plt.figure()

    # 为每一对数据创建一个折线图
    for i in range(len(data)):
        plt.plot([1, 2], data[i], label=names[i])

    # 添加图例
    plt.legend()

    # 显示图表
    # plt.show()
    
    save_path = os.path.join(save_path, fig_name)
    plt.savefig(save_path)

# sh_logname="20240308_eval.log"
logname_pre = "20240311_eval_plot"
result_path = r"E:\Yiling\at_SIAT_research\z_result\20240311_eval"

test_path_exist(result_path)
sys.stdout = Logger(os.path.join(result_path, logname_pre+".log"))

name_list = ["dwq_l8", "dwq_s2", "xj_l8", "xj_s2"]
model_dict = {
    "dwq_l8": r"E:\Yiling\at_SIAT_research\z_result\20240307_1_train_landsat\20240307_train_dwq_landsat\unet_epoch37.pth",
    "dwq_s2": r"E:\Yiling\at_SIAT_research\z_result\20240306_train_dwq_s2_class_weights\unet_epoch13.pth",
    "xj_l8": r"E:\Yiling\at_SIAT_research\z_result\20240307_1_train_landsat\20240307_train_xj_landsat\unet_epoch78.pth",
    "xj_s2": r"E:\Yiling\at_SIAT_research\z_result\20240306_train_xj_s2_class_weights\unet_epoch58.pth"
    }
data_dict = {
    "dwq_l8": r"E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_landsat8\train_val",
    "dwq_s2": r"E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val",
    "xj_l8": r"E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_landsat8\train_val",
    "xj_s2": r"E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val"
    }

# for model_name in name_list:
#     for data_name in name_list:
#         command = f'python -u -W ignore eval.py --result_path {result_path} --data_path {data_dict[data_name]} --load_model {model_dict[model_name]} --log_name {logname_pre}{model_name}_to_{data_name}.log >> {sh_logname}'
#         os.system(command)
#         # print(command)

# index_name_list = ['OA', 'kappa', 'precision', 'recall', 'F1', 'miou']

transfer_dict = {}

for model_name in name_list:
    for data_name in name_list:
        index_name_list, eval_index, ious_mean = eval(data_path=data_dict[data_name], 
                                                      load_model=model_dict[model_name], 
                                                      n_classes=9, 
                                                      batch_size=16,
                                                      num_workers=0
                                                      )
        # to do
        transfer_dict[f"{model_name}--{data_name}"] = [index_name_list, eval_index, ious_mean]

transfer_space_name_list = []
transfer_scale_name_list = []
transfer_space_scale_name_list = []
for key_name in transfer_dict.keys():
    source_name, target_name = key_name.split("--")
    source_name_0, source_name_1 = source_name.split("_")
    target_name_0, target_name_1 = target_name.split("_")
    if source_name_0 == target_name_0 and source_name_1 != target_name_1:
        transfer_scale_name_list.append(key_name)
    elif source_name_0 != target_name_0 and source_name_1 == target_name_1:
        transfer_space_name_list.append(key_name)
    elif source_name_0 != target_name_0 and source_name_1 != target_name_1:
        transfer_space_scale_name_list.append(key_name)

print("transfer_space_name_list:", transfer_space_name_list)
print("transfer_scale_name_list:", transfer_scale_name_list)
print("transfer_space_scale_name_list:", transfer_space_scale_name_list)

transfer_type_dict = {
    "transfer_space": transfer_space_name_list,
    "transfer_scale": transfer_scale_name_list,
    "transfer_space_scale": transfer_space_scale_name_list
}
print(transfer_type_dict)


for transfer_type in transfer_type_dict.keys():
    eval_index_data =[]
    for transfer_s_t in transfer_type_dict[transfer_type]:
        source_name = transfer_s_t.split("--")[0]
        eval_index_data.append([transfer_dict[f"{source_name}--{source_name}"][1], transfer_dict[transfer_s_t][1]])
    for i, index_name in enumerate(index_name_list):
        data = [[index[0][i], index[1][i]] for index in eval_index_data]
        plot_lines(data, transfer_type_dict[transfer_type], result_path, transfer_type+"_"+index_name)