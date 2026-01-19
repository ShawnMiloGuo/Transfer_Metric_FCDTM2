import os
import glob
from PIL import Image
from tqdm import tqdm

def create_image_grid_once(src_dir, row_names, col_names, output_path, task_transfer="dwq_sentinel2-xj_sentinel2", grid_eval_index="F1_delta"):
    # 创建一个字典来存储每个图像
    images = {}
    for row in row_names:
        for col in col_names:
            # pattern = os.path.join(src_dir, f"*{col}_{row}_F1_delta.png")
            # pattern = os.path.join(src_dir, f"draw_xj_*{col}_{row}_OA_delta.png")
            pattern = os.path.join(src_dir, f"draw_{task_transfer}_*{col}_{row}_{grid_eval_index}.png")
            files = glob.glob(pattern)
            if files:
                images[(row, col)] = Image.open(files[0])
            else:
                print(f"No image found for row: {row}, col: {col}")

    # 获取每个图像的尺寸（假设所有图像尺寸相同）
    img_width, img_height = next(iter(images.values())).size

    # 创建一个新的图像来存储网格
    grid_width = img_width * len(col_names)
    grid_height = img_height * len(row_names)
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # 将每个图像粘贴到网格中
    for row_idx, row in enumerate(row_names):
        for col_idx, col in enumerate(col_names):
            if (row, col) in images:
                grid_image.paste(images[(row, col)], (col_idx * img_width, row_idx * img_height))

    # 保存合成的大图
    grid_image.save(output_path)
    print(f"Grid image saved to {output_path}")

def create_image_for_dirs(src_directory_list, row_names, col_names, output_image_dir, task_transfer_source, task_transfer_target, grid_eval_index="F1_delta"):
    task_transfer_list = [f"{task_transfer_source}-{task_transfer_target}", f"{task_transfer_target}-{task_transfer_source}"]
    for src_directory in tqdm(glob.glob(src_directory_list)):
        for task_transfer_name in task_transfer_list:
            output_image_name = f"{task_transfer_name}_{os.path.basename(src_directory)}.png"
            output_image_path = os.path.join(output_image_dir, output_image_name)
            create_image_grid_once(os.path.join(src_directory,"fig"), row_names, col_names, output_image_path, task_transfer=task_transfer_name, grid_eval_index=grid_eval_index)

if __name__ == "__main__":
    mean_dif_key = ["FD", 
                    "mean_dif_absolute", "mean_dif_absolute_abs", 
                    "mean_dif_relative", "mean_dif_relative_abs",]
                    # "mean_dif_absolute_normalized", "mean_dif_relative_normalized", 
                    # "mean_dif_absolute_abs_normalized", "mean_dif_relative_abs_normalized"]
    y_diff_dict_key = ["y0_y1_diff", "y0_y1_diff_abs", 
                        "y0_y1_diff_normalized", "y0_y1_diff_abs_normalized", "sum"]
    for i in mean_dif_key:
        print(i)
    for i in y_diff_dict_key:
        print(i)
    # "mean_dif_absolute_sum", "mean_dif_absolute_abs_sum", "mean_dif_relative_sum", "mean_dif_relative_abs_sum",
    row_names = y_diff_dict_key
    col_names = mean_dif_key

    # 4 tasks
    # 20241210_1734_1_dwq_s2_xj_s2_
    # 20241210_1734_2_dwq_l8_xj_l8_
    # 20241210_1734_3_dwq_s2_dwq_l8_
    # 20241210_1734_4_xj_s2_xj_l8_
    task_path_list = ["20241210_1734_1_dwq_s2_xj_s2_", "20241210_1734_2_dwq_l8_xj_l8_", "20241210_1734_3_dwq_s2_dwq_l8_", "20241210_1734_4_xj_s2_xj_l8_"]
    task_transfer_source_list = ["dwq_sentinel2", "dwq_landsat8", "dwq_sentinel2", "xj_sentinel2"]
    task_transfer_target_list = ["xj_sentinel2", "xj_landsat8", "dwq_landsat8", "xj_landsat8"]

    for i in range(4):
        # 4 tasks
        grid_eval_index = "F1_delta" # "F1_delta", "OA_delta"
        src_directory = r"E:\Yiling\at_SIAT_research\z_result\20241210_transfer_metric_FD_cross_sensor_batch14_FD\20241210_1734_1_dwq_s2_xj_s2_\*_100img"
        src_directory = src_directory.replace("20241210_1734_1_dwq_s2_xj_s2_", task_path_list[i])
        output_image_path = rf"E:\Yiling\at_SIAT_research\z_result\20241210_transfer_metric_FD_cross_sensor_batch14_FD\20241211_image_grid_1_{grid_eval_index}\20241210_1734_1_dwq_s2_xj_s2_"
        output_image_path = output_image_path.replace("20241210_1734_1_dwq_s2_xj_s2_", task_path_list[i])

        os.makedirs(output_image_path, exist_ok=True)
        create_image_for_dirs(src_directory, row_names, col_names, output_image_path, 
                              task_transfer_source=task_transfer_source_list[i],
                              task_transfer_target=task_transfer_target_list[i],
                              grid_eval_index=grid_eval_index)
