import re

# 文件路径
log_file_path = r"E:\Yiling\at_SIAT_research\z_result\20250312_transfer_metric_GBC_batch14\20250312_0944_1_dwq_s2_xj_s2_\1_GBC_-_all-batch1_100img\transfer_metric_GBC_dwq_s2_xj_s2.log"

# 打开日志文件并读取内容
with open(log_file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 正则表达式匹配
source_pattern = re.compile(r"dataset_name_source: (\w+), dataset_name_target: (\w+), binary_class_index: (\d+)")
val_images_pattern = re.compile(r"total val images: (\d+)")

# 结果存储
results = []

# 遍历日志文件
for i, line in enumerate(lines):
    source_match = source_pattern.search(line)
    print(f"{i}, {source_match}")
    if source_match:
        source = source_match.group(1)
        target = source_match.group(2)
        class_index = source_match.group(3)

        # 查找接下来的两行中的 total val images
        val_images_source = val_images_pattern.search(lines[i + 1])
        val_images_target = val_images_pattern.search(lines[i + 2])

        if val_images_source and val_images_target:
            val_images_source_count = val_images_source.group(1)
            val_images_target_count = val_images_target.group(1)

            # 保存结果
            results.append((source, target, class_index, val_images_source_count, val_images_target_count))

# 打印结果
for result in results:
    print(f"Source: {result[0]}, Target: {result[1]}, Class Index: {result[2]}, "
          f"Source Val Images: {result[3]}, Target Val Images: {result[4]}")