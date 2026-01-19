# Description: Evaluate the model on the validation set and output the evaluation index of each class
# especially for binary classification

import argparse
import os
import sys
import torch
import numpy as np
from torch.utils import data
from component.dataset import get_dataset_reader
from component.evaluation_index import all_index as all_index_cpu
# from component.evaluation_index_gpu import all_index as all_index_gpu
from component.evaluation_index_gpu_each_class import all_index as all_index_gpu
from tqdm import tqdm
from class_counts import compute_class_counts


def eval(data_path,
         load_model,
         n_classes: int=2,
         batch_size: int = 1,
         dataset_name: str = 'rgbn',
         calculate_on_gpu: bool = True,
         num_workers: int = 4,
         binary_class_index: int = -1,
         label_1_percent: float = 0.0,
         ):
    print()

    # load model
    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("model_device:", model_device)
    if os.path.exists(load_model):
        print(f'Model loaded from {load_model}')
        model=torch.load(load_model, map_location=model_device)
    else:
        print(f'Path not exist: {load_model}')

    # class counts
    compute_class_counts(data_path, n_classes=n_classes, dataset_name=dataset_name, binary_class_index=binary_class_index, label_1_percent=label_1_percent)
    
    # dataloader, train or val?
    dataset_reader = get_dataset_reader(dataset_name)
    val_dataset = dataset_reader(root_dir=data_path, is_train=0, transform= None, binary_class_index = binary_class_index, label_1_percent = label_1_percent)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # miou, OA, ious
    index_name_list = ['OA', 'kappa', 'precision', 'recall', 'F1', 'miou']
    eval_index_list = []
    ious_list = []
    precision_each_class_list = []
    recall_each_class_list = []
    F1_each_class_list = []

    model.eval()
    n_val = len(val_dataset)
    n_batch = n_val // batch_size + 1
    with tqdm(total=n_val, desc='eval', unit='img') as pbar:
        for i, (images, true_masks_cpu) in enumerate(val_loader):
            images = images.to(device=model_device, dtype=torch.float32)
            true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
            masks_pred = model(images)

            pbar.update(images.shape[0])
            pbar.set_postfix(**{f'Batch({n_batch}) ': i+1})

            # predictions = np.argmax(masks_pred.cpu().detach().numpy(), axis=1)
            # labels = true_masks_cpu
            
            if calculate_on_gpu:
                # calculate index on gpu
                predictions = torch.argmax(masks_pred, dim=1)
                labels = true_masks
                all_index = all_index_gpu
            else:
                # calculate index on cpu
                predictions = np.argmax(masks_pred.cpu().detach().numpy(), axis=1)
                labels = true_masks_cpu
                all_index = all_index_cpu
            # OA, kappa, precision, recall, F1, miou, ious = all_index(predictions, labels, num_classes=n_classes)
            OA, kappa, precision, recall, F1, miou, ious, precision_each_class, recall_each_class, F1_each_class = all_index(predictions, labels, num_classes=n_classes)
            
            eval_index_list.append([OA, kappa, precision, recall, F1, miou])
            ious_list.append(ious)
            precision_each_class_list.append(precision_each_class)
            recall_each_class_list.append(recall_each_class)
            F1_each_class_list.append(F1_each_class)

    eval_index = np.mean(eval_index_list,axis=0)
    ious_mean = (np.mean(ious_list, axis=0)).tolist()
    precision_each_class_mean = (np.mean(precision_each_class_list, axis=0)).tolist()
    recall_each_class_mean = (np.mean(recall_each_class_list, axis=0)).tolist()
    F1_each_class_mean = (np.mean(F1_each_class_list, axis=0)).tolist()

    
    # log
    for i, index_name in enumerate(index_name_list):
        print(f'{index_name}: {eval_index[i]}')
    print("ious_each_class:", ious_mean)
    print("precision_each_class:", precision_each_class_mean)
    print("recall_each_class:", recall_each_class_mean)
    print("F1_each_class:", F1_each_class_mean)

    return index_name_list, eval_index, ious_mean


# Test whether the directory exists, create it if it does not exist
def test_path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Complete makedirs:",path)
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
def save_log(args):
    # Test whether result path exists
    result_path = args.result_path
    test_path_exist(result_path)
    # LOG
    sys.stdout = Logger(os.path.join(result_path, args.log_name))
    print("\n\n","*"*30)
    print("log save in :", os.path.join(result_path, args.log_name))
    # 将 args 转换为字典，按照键值对的方式输出
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f'{key}: {value}')
def get_args():
    parser = argparse.ArgumentParser(description='Hyperparams for evaluating.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--n_classes', type=int, default=2, help='The number of classes')
    parser.add_argument('--binary_class_index', type=int, default=-1, help='The class index of binary classification')
    parser.add_argument('--label_1_percent', type=float, default=0.0, help='which the percentage of label_1 greater than ')
    #"1 Dataset Path"
    parser.add_argument('--data_path', type=str, help='Path to datasets, including images/train & val, annotations/train & val', 
                        default=r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val')
    parser.add_argument('--load_model',  type=str, default=r'E:\Yiling\at_SIAT_research\z_result\20240222_train_dwq_s2\unet_epoch29.pth', help='Load model from a .pth file')
    parser.add_argument('--dataset_name', type=str, default='rgbn', help='Dataset to use [\'rgbn etc\']')
    #"2 Result Path and Log Name"
    parser.add_argument('--log_name', type=str, help='log name', 
                        default='20240229_eval_dwqs2_to_dwqs2.log')
    parser.add_argument('--result_path', type=str, help='Path to RESULT', 
                        default=r'E:\Yiling\at_SIAT_research\z_result\20240324_eval_each_class')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    save_log(args)
    eval(
        data_path=args.data_path,
        load_model=args.load_model,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        binary_class_index=args.binary_class_index,
        label_1_percent=args.label_1_percent,
    )