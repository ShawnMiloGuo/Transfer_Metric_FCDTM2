from collections import defaultdict
from tqdm import tqdm
import torch
import sys
import os
import time

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def check(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            # print("Early stopping")
            return True

        return False
    
# class weights
def compute_class_weights(train_loader, n_classes, n_train):
    # 初始化一个字典来存储每个类别的样本数量
    class_counts = defaultdict(int)

    # 遍历训练集的数据
    with tqdm(total=n_train, desc=f'Computing class weights', unit='img') as pbar:
        for _, targets_cpu in train_loader:
            # targets = targets_cpu.to(device='cuda', dtype=torch.int)
            targets = targets_cpu
            # 统计每个类别的样本数量
            for class_index in range(n_classes):
                class_counts[class_index] += (targets == class_index).sum().item()
            pbar.update(targets.shape[0])

    # print(f'class_counts: {class_counts}')
    # 计算每个类别的权重，权重设置为类别的样本数量的倒数
    # 如果某个类别的样本数量为零，将其权重设置为零
    class_weights = [1.0 / class_counts[i] if class_counts[i] > 0 else 0 for i in range(n_classes)]
    class_weights = torch.tensor(class_weights)
    # 归一化权重，使其和为1
    class_weights = class_weights / class_weights.sum()

    return class_weights, class_counts

# autolog
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        # self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
# Test whether the directory exists, create it if it does not exist
def test_path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Complete makedirs:",path)
def save_log(result_path: str = './', log_name: str = 'default.log', args=None):
    # Test whether result path exists
    test_path_exist(result_path)
    # LOG
    sys.stdout = Logger(os.path.join(result_path, log_name))
    print("\n","="*100)
    # print start time
    print("Start Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("log save in:", os.path.join(result_path, log_name))

    # print args
    if args is not None:
        args_dict = vars(args)
        for key, value in args_dict.items():
            print(f'{key}: {value}')