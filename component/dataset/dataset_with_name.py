import os
from PIL import Image
from torch.utils.data import Dataset
from .data_from_gdal import load_TIF_NoGEO as tif_reader
import numpy as np
from tqdm import tqdm

class rgbn_dataset_with_name(Dataset):
    '''
    path:
    ./annotations/train
    ./annotations/val
    ./images/train
    ./images/val
    '''
    def __init__(self, root_dir, is_train=True, transform=None, binary_class_index = -1, label_1_percent = 0.3):
        self.root_dir = root_dir
        if is_train:
            self.split = 'train'
        else:
            self.split = 'val'
        self.transform = transform
        self.binary_class_index = binary_class_index
        self.image_dir = os.path.join(root_dir, 'images', self.split)
        self.annotation_dir = os.path.join(root_dir, 'annotations', self.split)

        self.image_files = os.listdir(self.image_dir)
        self.annotation_files = os.listdir(self.annotation_dir)

        if self.binary_class_index >= 0:
            # binary classification
            list_file = os.path.join(self.root_dir, f'binary_class_{self.binary_class_index}_{self.split}_{label_1_percent}.txt')
            if os.path.exists(list_file):
                # 如果文件存在，从文件中加载所有图像文件名
                print(f"load binary class list from file: {list_file}")
                with open(list_file, 'r') as f:
                    self.annotation_files = [line.strip() for line in f]
            else:
                # 如果文件不存在，通过一系列操作生成文件
                self.annotation_files = []
                for label_file_name in tqdm(os.listdir(self.annotation_dir)):
                    annotation = np.asarray(tif_reader(self.annotation_dir, label_file_name), dtype=np.int8)
                    annotation = (annotation == self.binary_class_index).astype(np.int8)
                    if np.sum(annotation) > annotation.size * label_1_percent:
                        self.annotation_files.append(label_file_name)

                # 将生成的列表写入文件
                self.annotation_files = sorted(self.annotation_files)
                print(f"write binary class list to file: {list_file}")
                with open(list_file, 'w') as f:
                    for file_name in self.annotation_files:
                        f.write(f'{file_name}\n')
            self.image_files = self.annotation_files.copy()
        
        self.annotation_files = sorted(self.annotation_files)
        self.image_files = sorted(self.image_files)
        
        print(f"total {self.split} images: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        assert self.image_files[idx] == self.annotation_files[idx], \
            f"image file name {self.image_files[idx]} does not match annotation file name {self.annotation_files[idx]}"
        image = np.asarray(tif_reader(self.image_dir, self.image_files[idx]), dtype=np.float32)
        annotation = np.asarray(tif_reader(self.annotation_dir, self.annotation_files[idx]), dtype=np.int8)

        if self.binary_class_index >= 0:
            annotation = (annotation == self.binary_class_index).astype(np.int8)

        if self.transform:
            image = self.transform(image)
            annotation = self.transform(annotation)

        return image, annotation, self.image_files[idx]