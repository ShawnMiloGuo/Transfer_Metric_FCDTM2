# 从指定目录的.log文件中读取每一行信息
# 提取需要的指标等数据
# 将提取的信息保存到一个新的文件中，extract_from_class-counts-binary_log.csv
import glob

log_path = r"E:\Yiling\at_SIAT_research\z_result\20240620_class_counts_binary_landsat\20240620_class_counts_binary_0.0-0.3.log" # 
result_file_path = r"E:\Yiling\at_SIAT_research\z_result\20240620_class_counts_binary_landsat\extract_from_class-counts-binary_log_0.0-0.3.csv"

# 找到所有的.log文件
log_files = glob.glob(log_path)

# 创建一个字典，用于保存结果
result_dict = {}
result_item_list = ["log_file", "label_1_percent", "data_name", 
                    "total train images", "total val images", 
                    "class_counts_list", "class_percentage_list", 
                    ]
for item in result_item_list:
    result_dict[item] = []

# 遍历所有的.log文件
str_trans = str.maketrans(":", ",", "[]=")
for log_file in log_files:
    result_dict["log_file"].append(log_file)
    with open(log_file, 'r') as file:
        item_exist = {}
        for item in result_item_list[1:]:
            item_exist[item] = False
        for line in file:
            if line.startswith("Warning: No data found in"):
                for item in ["class_counts_list", "class_percentage_list"]:
                    result_dict[item].append(f"{item}, 0, 0")
            if line.startswith("data_name"):
                for item in result_item_list[1:]:
                    item_exist[item] = False
            for item in result_item_list[1:]:
                if line.startswith(item) and not item_exist[item]:
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

# 格式化输出
format_result_file_path = result_file_path.replace(".csv", "_format.csv")
with open(format_result_file_path, 'w') as format_result_file:
    num_raw = len(result_dict["data_name"])//len(result_dict["label_1_percent"])
    percent_list = [line.replace('label_1_percent, ', '') for line in result_dict['label_1_percent']]
    percent_list_str = ",".join(percent_list)
    percent_list_2 = ["1"+str for str in percent_list]
    percent_list_str_2 = ",".join(percent_list_2)
    format_result_file.write(f"cls,{percent_list_str},{percent_list_str_2}\n")
    for raw in range(num_raw):
        data_name = result_dict["data_name"][raw].split(", ")[1].replace(" ", "")
        format_result_file.write(data_name + ",")
        for col in range(len(result_dict["label_1_percent"])):
            train_images = result_dict["total train images"][col * num_raw + raw].split(", ")[1]
            format_result_file.write(train_images + ",")
        for col in range(len(result_dict["label_1_percent"])):
            label_1_percent = result_dict["class_percentage_list"][col * num_raw + raw].split(",")[-1].replace(" ", "")
            format_result_file.write(label_1_percent + ",")
        format_result_file.write("\n")