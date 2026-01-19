import os
import shutil
import glob
from tqdm import tqdm

def select_and_copy_images(src_dir, dst_dir, prefix, suffix):
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 构建匹配模式
    pattern = os.path.join(src_dir, '**', f'{prefix}*{suffix}')
    
    # 使用glob.glob查找匹配的文件
    for src_file_path in tqdm(glob.glob(pattern, recursive=True)):
        # 计算相对路径
        relative_path = os.path.relpath(src_file_path, src_dir)
        dst_file_path = os.path.join(dst_dir, relative_path)
        dst_file_dir = os.path.dirname(dst_file_path)

        # 检查目标文件夹是否存在，不存在则创建
        if not os.path.exists(dst_file_dir):
            os.makedirs(dst_file_dir)

        # 复制文件到目标目录
        shutil.copy2(src_file_path, dst_file_path)
        # print(f"Copied {src_file_path} to {dst_file_path}")

if __name__ == "__main__":
    # 示例参数
    src_directory = r"E:\Yiling\at_SIAT_research\z_result\20240911_transfer_metric_FD_weighted_y_diff"
    dst_directory = r"E:\Yiling\at_SIAT_research\z_result\20240911_transfer_metric_FD_weighted_y_diff\20240919_selected"
    file_prefix = "draw_xj_sentinel2"
    file_suffix = "F1_delta.png"

    select_and_copy_images(src_directory, dst_directory, file_prefix, file_suffix)