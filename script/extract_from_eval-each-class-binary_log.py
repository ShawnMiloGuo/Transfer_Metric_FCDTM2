# 从指定目录的*.log文件中读取每一行信息
# 提取需要的指标等数据
# 将提取的信息保存到一个新的文件中，extract_from_eval_log.csv
import glob
# E:\Yiling\at_SIAT_research\z_result\20240620_eval_binary_landsat\20240620_eval_each_class_binary_CE_dice_gt20\20240620_eval_dwql8-cls1_to_dwql8-cls1.log
log_path = r"E:\Yiling\at_SIAT_research\z_result\20240620_eval_binary_landsat\20240621_eval_each_class_binary_dice_gt5\20240621_eval_*.log"
result_file_path = r"E:\Yiling\at_SIAT_research\z_result\20240620_eval_binary_landsat\20240621_eval_each_class_binary_dice_gt5\20240621_extract_from_eval_log.csv"

# 找到所有的.log文件
log_files = glob.glob(log_path)

# 创建一个字典，用于保存结果
result_dict = {}
result_item_list = ["log_file", "total train images", "total val images", 
                    "class_counts_list", "class_percentage_list", 
                    "OA", 
                    "ious_each_class", "precision_each_class", "recall_each_class", "F1_each_class"]
for item in result_item_list:
    result_dict[item] = []

# 遍历所有的.log文件
str_trans = str.maketrans(":[]", ",,,")
for log_file in log_files:
    # result_dict["log_file"].append(log_file.split("\\")[-1])
    result_dict["log_file"].append(log_file)
    with open(log_file, 'r') as file:
        item_exist = {}
        for item in result_item_list[1:]:
            item_exist[item] = False
        for line in file:
            for item in result_item_list[1:]:
                if line.startswith(item) and not item_exist[item]:
                    # for old_str in [":", "[", "]"]:
                    #     line = line.replace(old_str, ",")
                    line = line.translate(str_trans).replace("\n", "")
                    result_dict[item].append(line)
                    item_exist[item] = True

# 将结果按照指标分别保存在文件中
with open(result_file_path, 'w') as result_file:
    for item in result_item_list:
        result_file.write(item + "\n")
        for line in result_dict[item]:
            result_file.write(line + "\n")
        result_file.write("\n")
