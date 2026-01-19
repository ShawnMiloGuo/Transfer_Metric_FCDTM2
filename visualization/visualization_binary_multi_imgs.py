# Description: Evaluate the model on the validation set and output the evaluation index of each class
# especially for binary classification

import argparse
import os
import sys
# Get the parent directory of the current file, add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
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
         dataset_name: str = 'rgbn_with_name',
         calculate_on_gpu: bool = True,
         num_workers: int = 4,
         binary_class_index: int = -1,
         label_1_percent: float = 0.2,
         result_path: str = './',
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
        return

    # class counts
    # compute_class_counts(data_path, n_classes=n_classes, dataset_name='rgbn', binary_class_index=binary_class_index, label_1_percent=label_1_percent)
    
    # dataloader, train or val?
    is_train = 1
    dataset_reader = get_dataset_reader(dataset_name)
    val_dataset = dataset_reader(root_dir=data_path, is_train=is_train, transform= None, binary_class_index = binary_class_index, label_1_percent = label_1_percent)
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
        for i, (images, true_masks_cpu, img_name) in enumerate(val_loader):
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
            
            # save images
            path_save_img = os.path.join(result_path, 'images')
            true_masks_cpu = true_masks_cpu.squeeze().numpy()
            img_label_1_percent = true_masks_cpu.sum() / true_masks_cpu.size

            # img_name_prefix = f"F1_{F1:.4f}_miou_{miou:.4f}_{img_name[0].split('.')[0]}_label-1_{img_label_1_percent:.4f}"
            img_name_prefix = f"F1_{F1:.4f}_label-1_{img_label_1_percent:.4f}_miou_{miou:.4f}_{img_name[0].split('.')[0]}"

            images_cpu = images.cpu().squeeze().permute(1,2,0)
            predictions_cpu = predictions.cpu().squeeze()
            # save_img_lbl_pred(images_cpu.numpy(), true_masks_cpu, predictions_cpu.numpy(), img_name_prefix, path_save_img)
            save_img_lbl_pred_4(images_cpu.numpy(), true_masks_cpu, predictions_cpu.numpy(), img_name_prefix, path_save_img)

            # if i == 0:
            #     break

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
    # 输出图片数量
    print(f'Number of images: {n_val}')

    return index_name_list, eval_index, ious_mean

# 保存一张图片，分为几个子图
def save_img_lbl_pred(images, true_masks, predictions, img_name_prefix, path_save_img):
    # save images
    test_path_exist(path_save_img)
    img_rgb = images[:, :, [2, 1, 0]] #0-B，1-G，2-R，3-N
    img_nir = images[:, :, 3]

    plt.figure(figsize=(12, 3))
    # 添加第一个子图：原始图像RGB
    plt.subplot(1, 4, 1)
    plt.imshow(np.interp(img_rgb.clip(0,0.3), (0, 0.3), (0, 255)).astype(np.uint8))  # clip(0,0.3) 限制在0-0.3之间，interp((0, 0.3), (0, 255)) 将0-0.3映射到0-255
    plt.title('Img_rgb')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # 添加第二个子图：原始图像NIR
    plt.subplot(1, 4, 2)
    plt.imshow(np.interp(img_nir.clip(0,0.3), (0, 0.3), (0, 255)).astype(np.uint8), cmap='gray')
    plt.title('Img_nir')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    label_1_value = 0.8
    # 添加第三个子图：真实标签
    plt.subplot(1, 4, 3)
    plt.imshow(true_masks, cmap='gray')
    # plt.imshow(np.where(true_masks == 1, label_1_value, true_masks), cmap='gray', vmin=0, vmax=1)
    plt.title('Ground Truth')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # 添加第四个子图：预测标签
    plt.subplot(1, 4, 4)
    plt.imshow(predictions, cmap='gray')
    # plt.imshow(np.where(predictions == 1, label_1_value, predictions), cmap='gray', vmin=0, vmax=1)
    plt.title('Prediction')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(os.path.join(path_save_img, f'{img_name_prefix}.png'), dpi=100)
    plt.close()

# 分别保存 1img_1rgb、1img_2ngb、2gt、3pred 这四个图
def save_img_lbl_pred_4(images, true_masks, predictions, img_name_prefix, path_save_img):
    # save images
    test_path_exist(path_save_img)
    img_rgb = images[:, :, [2, 1, 0]] #0-B，1-G，2-R，3-N
    img_ngb = images[:, :, [3, 1, 0]]

    # 保存第一个图：原始图像RGB
    plt.imshow(np.interp(img_rgb.clip(0,0.3), (0, 0.3), (0, 255)).astype(np.uint8))
    plt.axis('off')
    plt.savefig(os.path.join(path_save_img, f'{img_name_prefix}_1img_1rgb.png'), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 保存第二个图：原始图像NIR
    plt.imshow(np.interp(img_ngb.clip(0,0.3), (0, 0.3), (0, 255)).astype(np.uint8))
    plt.axis('off')
    plt.savefig(os.path.join(path_save_img, f'{img_name_prefix}_1img_2ngb.png'), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 保存第三个图：真实标签
    plt.imshow(true_masks, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(path_save_img, f'{img_name_prefix}_2gt.png'), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 保存第四个图：预测标签
    plt.imshow(predictions, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(path_save_img, f'{img_name_prefix}_3pred.png'), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()


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
    parser.add_argument('--binary_class_index', type=int, default=2, help='The class index of binary classification')
    parser.add_argument('--label_1_percent', type=float, default=0.2, help='which the percentage of label_1 greater than ')
    #"1 Dataset Path"
    parser.add_argument('--data_path', type=str, help='Path to datasets, including images/train & val, annotations/train & val', 
                        default=r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val')
    parser.add_argument('--load_model',  type=str, default=r'E:\Yiling\at_SIAT_research\z_result\20240605_1_train_binary_dice\train_xj_sentinel2_cls_2\unet_epoch200_best_val.pth', help='Load model from a .pth file')
    parser.add_argument('--dataset_name', type=str, default='rgbn_with_name', help='Dataset to use [\'rgbn, rgbn_with_name, etc\']')
    #"2 Result Path and Log Name"
    parser.add_argument('--log_name', type=str, help='log name', 
                        default='20240621_visualization_binary_xjs2_dice_cls2_label1_percent.log')
    parser.add_argument('--result_path', type=str, help='Path to RESULT', 
                        default=r'E:\Yiling\at_SIAT_research\z_result\20240613_visualization_binary\visualization_binary_xjs2_dice_cls2_label1_percent_train')
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
        result_path=args.result_path,
    )