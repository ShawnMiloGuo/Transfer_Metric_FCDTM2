import os
import shutil

def move_pdfs_to_current_directory(current_directory):
    for root, dirs, files in os.walk(current_directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(current_directory, file)
                if source_path != destination_path:
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {source_path} to {destination_path}")

if __name__ == "__main__":
    current_directory = os.getcwd()  # 获取当前目录
    move_pdfs_to_current_directory(current_directory)