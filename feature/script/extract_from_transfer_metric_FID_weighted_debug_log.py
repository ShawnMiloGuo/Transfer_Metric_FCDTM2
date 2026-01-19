# 从指定目录的*.log文件中读取每一行信息
# 提取需要的指标等数据
# 将提取的信息保存到一个新的文件中，extract_from_eval_log.csv
import glob
# E:\Yiling\at_SIAT_research\z_result\20240903_transfer_metric_FID_weighted_debug\20240903_1_-_FID_all-all_dwqs2-xjs2_100img\transfer_metric_FID_weighted_dwqs2-xjs2.log
log_path = r"E:\Yiling\at_SIAT_research\z_result\20240903_transfer_metric_FID_weighted_debug\20240903_*\transfer_metric_FID_weighted_dwqs2-xjs2.log"

# 找到所有的.log文件
log_files = glob.glob(log_path)

# 遍历所有的.log文件
str_trans = str.maketrans({":": ",",
                            "[": None,
                            "]": None})
for log_file in log_files:
    result_item_list = ["log_file_path", 
                        "dataset_name_source", 
                        "last_layer_weight_1", 
                        "mean"]

    # 将结果按照指标分别保存在文件中
    result_file_path = log_file.replace(".log", "_extract.csv")
    with open(result_file_path, 'w') as result_file:
        result_file.write(log_file + "\n")
        with open(log_file, 'r') as file:
            for line in file:
                for item in result_item_list[1:]:
                    if line.startswith(item):
                        line = line.translate(str_trans)
                        result_file.write(line)

