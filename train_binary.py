import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from component.dataset import get_dataset_reader
from component.model import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from component.evaluation_index import all_index as all_index_cpu
from component.evaluation_index_gpu_binary import all_index as all_index_gpu
from component.dice_score import dice_loss
from component.utils import EarlyStopping
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
only_dice_loss = True  # only use (dice loss) or use (cross entropy loss and dice loss)

def train_model(
        n_epoch,
        batch_size,
        data_path,
        result_path,
        band_num,
        n_classes,
        ignore_index,
        manual_seed,
        load_model,
        model_arch,
        dataset_name,
        args,
        l_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        momentum: float = 0.95,
        binary_class_index: int = -1,
        loss_weight: int = 0,
        label_1_percent: float = 0.0,
        ):
    # seed
    random.seed(manual_seed)  # 生产固定随机数，0.84
    np.random.seed(manual_seed)  # 生产固定随机数0.54
    torch.manual_seed(manual_seed)  # 生产固定随即数
    if torch.cuda.is_available():  # 如果gpu中cuda和torch版本相同
        torch.cuda.manual_seed_all(manual_seed)  # 为所有GPU设置种子，生产随机数
        torch.backends.cudnn.enabled = True  # 设置非确定算法，使得更加优化
    
    # binary classification
    if binary_class_index >= 0:
        assert n_classes == 2, f'n_classes should be 2 for binary classification, but be {n_classes}'
    # dataset
    dataset_reader = get_dataset_reader(dataset_name)
    train_dateset = dataset_reader(root_dir=data_path, is_train=True, transform= None, binary_class_index=binary_class_index, label_1_percent = label_1_percent)
    train_loader = data.DataLoader(train_dateset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = dataset_reader(root_dir=data_path, is_train=False, transform= None, binary_class_index=binary_class_index, label_1_percent = label_1_percent)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model_init = get_model(model_arch)
    model = model_init(n_classes = n_classes, in_channels = band_num)
    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device = model_device)
    model.apply(init_weights)
    if load_model is not None :
        if os.path.exists(load_model):
            print(f'Model loaded from {load_model}')
            model=torch.load(load_model, map_location=model_device)
        else:
            print(f'Path not exist: {load_model}')
    
    
    # loss, 类别不均衡问题!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # criteria
    class_weight = compute_class_weights(train_loader, n_classes, len(train_dateset))
    print(f'class_weight: {class_weight}')
    print(f"use weighted loss: {loss_weight}")
    if loss_weight:
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(device = model_device), ignore_index=ignore_index)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=l_rate, weight_decay=weight_decay, momentum=momentum)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)

    # train, 早停机制
    early_stopping = EarlyStopping(patience=30)
    
    n_train = len(train_dateset)
    n_val = len(val_dataset)
    n_batch_train = n_train // batch_size + 1
    n_batch_val = n_val // batch_size + 1
    epoch_list = []
    best_epoch_index = []

    index_name_list = ['OA', 'kappa', 'precision', 'recall', 'F1', 'miou']
    train_loss_list = []
    train_index_list = []
    train_ious_list = []
    train_preloss_list = []
    train_dice_list = []
    train_precision_each_class_list = []
    train_recall_each_class_list = []
    train_F1_each_class_list = []

    val_loss_list = []
    val_index_list = []
    val_ious_list = []
    val_preloss_list = []
    val_dice_list = []
    val_precision_each_class_list = []
    val_recall_each_class_list = []
    val_F1_each_class_list = []

    for epoch in range(1, n_epoch + 1):
        epoch_list.append(epoch)
        # train
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{n_epoch}', unit='train') as pbar:
            for i, (images, true_masks_cpu) in enumerate(train_loader):
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                images = images.to(device=model_device, dtype=torch.float32)
                true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
                
                masks_pred = model(images)
                # dice loss
                dice = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                # alpha_loss = (loss.item() + dice.item()) / (2 * loss.item())
                # alpha_dice = (loss.item() + dice.item()) / (2 * dice.item())
                # loss = loss * alpha_loss + dice * alpha_dice
                if only_dice_loss:
                    loss = dice
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss = loss + dice
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                print(f'Epoch {epoch}/{n_epoch}, {i+1}/{n_batch_train}, train loss: {loss.item()}')
                # if i == 2:
                #     break
        

        # val
        model.eval()
        # test train_dateset
        train_val_loss = val_model(model, n_train, epoch, n_epoch, 
                             train_loader, model_device, 
                             criterion, n_batch_train, n_classes, 
                             train_loss_list, train_index_list, train_ious_list, train_preloss_list, train_dice_list, 
                             train_precision_each_class_list, train_recall_each_class_list, train_F1_each_class_list,
                             train_or_val = 'train')
        # test val_dateset
        val_loss = val_model(model, n_val, epoch, n_epoch, 
                             val_loader, model_device, 
                             criterion, n_batch_val, n_classes, 
                             val_loss_list, val_index_list, val_ious_list, val_preloss_list, val_dice_list,
                             val_precision_each_class_list, val_recall_each_class_list, val_F1_each_class_list,
                             train_or_val = 'val')
        # scheduler.step(val_loss)


        # save model
        save_model_flag = 0
        # save the best model, and remove the last best model
        if epoch == 1 :
            best_val_loss = val_loss
            best_val_loss_index = 1
            save_model_flag = 1
        if val_loss < best_val_loss:
            os.remove(os.path.join(result_path, f'{args.model_arch}_epoch{best_val_loss_index}_best_val.pth'))
            os.remove(os.path.join(result_path, f'{args.model_arch}_epoch{best_val_loss_index}_best_val_dict.pth'))
            best_val_loss = val_loss
            best_val_loss_index = epoch
            save_model_flag = 1
        if save_model_flag == 1:
            torch.save(model, os.path.join(result_path, f'{args.model_arch}_epoch{epoch}_best_val.pth'))
            torch.save(model.state_dict(), os.path.join(result_path, f'{args.model_arch}_epoch{epoch}_best_val_dict.pth'))
        # save the model every 10 epochs
        # if epoch % 10 == 0 or epoch == n_epoch:
        #     torch.save(model, os.path.join(result_path, f'{args.model_arch}_epoch{epoch}.pth'))
        best_epoch_index.append(best_val_loss_index)
        

        # print, draw, d2l
        print(f'epoch, ', epoch_list)
        print(f'best_epoch_index, ', best_epoch_index)
        plot_train_test_trend(epoch_list, best_epoch_index, ylabel='best_epoch', save_path=result_path, fig_name='fig_best_epoch')
        print(f'train_loss, ', train_loss_list)
        print(f'val_loss, ', val_loss_list)
        plot_train_test_trend(train_loss_list, val_loss_list, ylabel='loss', save_path=result_path, fig_name='fig_loss')
        train_index_list_array = np.array(train_index_list)
        val_index_list_array = np.array(val_index_list)
        for i, index in enumerate(index_name_list):
            print(f'train_{index}, ', (train_index_list_array[:,i]).tolist())
            print(f'val_{index}, ', (val_index_list_array[:,i]).tolist())
            plot_train_test_trend(train_index_list_array[:,i], val_index_list_array[:,i], ylabel=index, save_path=result_path, fig_name='fig_'+index)
        print(f'train_ious, ', train_ious_list)
        print(f'val_ious, ', val_ious_list)
        # ious
        plot_train_test_trend(train_ious_list, val_ious_list, ylabel='ious', save_path=result_path, fig_name='fig_ious', 
                              train_label=['train'+str(i) for i in range(n_classes)],
                              test_label=['val'+str(i) for i in range(n_classes)])
        # precision_each_class_list
        plot_train_test_trend(train_precision_each_class_list, val_precision_each_class_list, ylabel='precision', save_path=result_path, fig_name='fig_precision_each_class',
                              train_label=['train'+str(i) for i in range(n_classes)],
                              test_label=['val'+str(i) for i in range(n_classes)])
        # recall_each_class_list
        plot_train_test_trend(train_recall_each_class_list, val_recall_each_class_list, ylabel='recall', save_path=result_path, fig_name='fig_recall_each_class',
                              train_label=['train'+str(i) for i in range(n_classes)],
                              test_label=['val'+str(i) for i in range(n_classes)])
        # F1_each_class_list
        plot_train_test_trend(train_F1_each_class_list, val_F1_each_class_list, ylabel='F1', save_path=result_path, fig_name='fig_F1_each_class',
                              train_label=['train'+str(i) for i in range(n_classes)],
                              test_label=['val'+str(i) for i in range(n_classes)])
        print(f'train_preloss_list, ', train_preloss_list)
        print(f'train_dice_list, ', train_dice_list)
        print(f'val_preloss_list, ', val_preloss_list)
        print(f'val_dice_list, ', val_dice_list)
        train_preloss_dice_list=[list(i) for i in zip(train_preloss_list, train_dice_list)]
        val_preloss_dice_list=[list(i) for i in zip(val_preloss_list, val_dice_list)]
        plot_train_test_trend(train_preloss_dice_list, val_preloss_dice_list, ylabel='loss_dice', 
                              save_path=result_path, fig_name='fig_preloss_dice',
                              train_label=['train_preloss', 'train_dice'], test_label=['val_preloss', 'val_dice'])
        
        # early stopping
        if early_stopping.check(val_loss):
            if epoch > 20:
                print(f'Early stopping on epoch {epoch}')
                break

def val_model(model, n_val, epoch, n_epoch, 
              data_loader, model_device,
              criterion, n_batch, n_classes, 
              val_loss_list, val_index_list, val_ious_list, val_preloss_list, val_dice_list,
              val_precision_each_class_list, val_recall_each_class_list, val_F1_each_class_list,
              train_or_val = 'val', calculate_on_gpu = True):
    # val
    model.eval()
    batch_val_loss = []
    batch_val_index = []
    batch_val_ious = []
    batch_val_dice = []
    batch_val_preloss = []
    batch_val_precision_each_class = []
    batch_val_recall_each_class = []
    batch_val_F1_each_class = []
    with torch.no_grad():
        with tqdm(total=n_val, desc=f'Epoch {epoch}/{n_epoch}', unit=train_or_val) as pbar:
            for i, (images, true_masks_cpu) in enumerate(data_loader):
                images = images.to(device=model_device, dtype=torch.float32)
                true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
                masks_pred = model(images)

                if only_dice_loss:
                    loss = torch.tensor(0.0)
                else:
                    loss = criterion(masks_pred, true_masks)
                batch_val_preloss.append(loss.item())

                # dice loss
                dice = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                if only_dice_loss:
                    loss = dice
                else:
                    loss = loss + dice

                pbar.update(images.shape[0])
                batch_val_loss.append(loss.item())
                batch_val_dice.append(dice.item())
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                print(f'Epoch {epoch}/{n_epoch}, {i+1}/{n_batch}, {train_or_val} loss: {loss.item()}')

                
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
                OA, kappa, precision, recall, F1, miou, ious, precision_each_class, recall_each_class, F1_each_class = all_index(predictions, labels, num_classes=n_classes)
                batch_val_index.append([OA, kappa, precision, recall, F1, miou])
                batch_val_ious.append(ious)
                batch_val_precision_each_class.append(precision_each_class)
                batch_val_recall_each_class.append(recall_each_class)
                batch_val_F1_each_class.append(F1_each_class)

                # if i == 2:
                #     break
    val_loss = np.mean(batch_val_loss)
    val_loss_list.append(np.mean(batch_val_loss))
    val_index_list.append(np.mean(batch_val_index,axis=0))
    val_ious_list.append((np.mean(batch_val_ious, axis=0)).tolist())
    val_preloss_list.append(np.mean(batch_val_preloss))
    val_dice_list.append(np.mean(batch_val_dice))
    val_precision_each_class_list.append((np.mean(batch_val_precision_each_class, axis=0)).tolist())
    val_recall_each_class_list.append((np.mean(batch_val_recall_each_class, axis=0)).tolist())
    val_F1_each_class_list.append((np.mean(batch_val_F1_each_class, axis=0)).tolist())
    return val_loss

# plot
def plot_train_test_trend(train_losses, test_losses, ylabel='Loss', save_path=None, fig_name='fig', train_label='train', test_label='val'):
    """
    绘制训练和测试损失的趋势图，并可选地保存为图片
    
    参数：
    - train_losses: 训练损失列表
    - test_losses: 测试损失列表
    - save_path: 图片保存路径，如果为 None则不保存图片, 默认为 None
    """
    # 创建 x 轴数据，表示每个 epoch
    epochs = np.arange(1, len(train_losses) + 1)

    # 绘制训练和测试损失的趋势图
    plt.figure()
    plt.plot(epochs, train_losses, label=train_label)
    plt.plot(epochs, test_losses, label=test_label)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(ylabel + ': train vs val')
    plt.legend()
    plt.grid(True)
    
    # 如果提供了保存路径，则保存图片
    if save_path is not None:
        save_path = os.path.join(save_path, fig_name)
        plt.savefig(save_path)
        # print(f"Figure saved in: {save_path}")
    else:
        plt.show()

# weight init
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # model = UNet()  # 假设UNet是你的模型类
    # model.apply(init_weights)

# class weights
def compute_class_weights(train_loader, n_classes, n_train):
    # 初始化一个字典来存储每个类别的样本数量
    class_counts = defaultdict(int)

    # 遍历训练集的数据
    with tqdm(total=n_train, desc=f'Computing class weights', unit='img') as pbar:
        for _, targets_cpu in train_loader:
            targets = targets_cpu.to(device='cuda', dtype=torch.int)
            # 统计每个类别的样本数量
            for class_index in range(n_classes):
                class_counts[class_index] += (targets == class_index).sum().item()
            pbar.update(targets.shape[0])

    print(f'class_counts: {dict(class_counts)}')
    # 计算每个类别的权重，权重设置为类别的样本数量的倒数
    # 如果某个类别的样本数量为零，将其权重设置为零
    class_weights = [1.0 / class_counts[i] if class_counts[i] > 0 else 0 for i in range(n_classes)]
    class_weights = torch.tensor(class_weights)
    # 归一化权重，使其和为1
    class_weights = class_weights / class_weights.sum()

    return class_weights

# Test whether the directory exists, create it if it does not exist
def test_path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Complete makedirs: ",path)
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
    parser = argparse.ArgumentParser(description='Hyperparams for training.')
    # training parameters
    parser.add_argument('--n_epoch', type=int, default=1000, help='Number of the epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--l_rate', type=float, default=1e-3, help='Learning Rate')
    # parser.add_argument('--l_rate', type=float, default=1e-4, help='Learning Rate')
    
    #"1 Dataset Path"
    parser.add_argument('--data_path', type=str, help='Path to datasets, including images/train & val, annotations/train & val', 
                        default=r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val')
    #"2 Result Path and Log Name"
    parser.add_argument('--log_name', type=str, help='log name', 
                        default='20240228_train_xj_s2.log')
    parser.add_argument('--result_path', type=str, help='Path to RESULT', 
                        default=r'E:\Yiling\at_SIAT_research\z_result\20240228_train_xj_s2_lr1e-3_epoch100')
    
    # image parameters
    parser.add_argument('--band_num', type=int, default=4, help='The number of band') 
    parser.add_argument('--n_classes', type=int, default=9, help='The number of classes')
    parser.add_argument('--binary_class_index', type=int, default=-1, help='The class index of binary classification')
    parser.add_argument('--label_1_percent', type=float, default=0.0, help='which the percentage of label_1 greater than ')
    parser.add_argument('--loss_weight', type=int, default=0, help='Whether to use loss weight')
    parser.add_argument('--ignore_index', type=int, default=-100, help='ignore index')
    parser.add_argument('--img_rows', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', type=int, default=256,
                        help='Width of the input image')
    
    # unnecessary
    parser.add_argument('--manual_seed', type=int, default=0, help='Manual seed')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay, [1e-4, 1e-2]')
    parser.add_argument('--momentum', type=float, default=0.95, help='Momentum for SGD')
    parser.add_argument('--optim', type=str, default='SGD', help='Optimizer to use [\'SGD, Nesterov etc\']')
    parser.add_argument('--load_model',  type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--model_arch', type=str, default='unet',
                        help='Architecture to use [\'unet, hrunet, deeplab, segnet, unetplus,fcn etc\']')
    parser.add_argument('--dataset_name', type=str, default='rgbn',
                        help='Dataset to use [\'rgbn etc\']')
    # parser.add_argument('--ost', type=str, default='8', help='Output stride to use [\'32, 16, 8 etc\']')
    # parser.add_argument('--freeze', action='store_true', help='Freeze BN params')
    # parser.add_argument('--restore', default=True, action='store_true', help='Restore Optimizer params')
    # parser.add_argument('--split', type=str, default='train',
    #                     help='Sets to use [\'train_aug, train, trainvalrare, trainval_aug, trainval etc\']')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    save_log(args)

    # time
    # Start time
    start_localtime=time.localtime()
    start_time_zt=time.time()
    print("Start time: ",time.strftime("%Y-%m-%d-%H:%M:%S", start_localtime))


    # Start train
    train_model(n_epoch=args.n_epoch,
                batch_size=args.batch_size,
                data_path=args.data_path,
                result_path=args.result_path,
                band_num=args.band_num,
                n_classes=args.n_classes,
                ignore_index=args.ignore_index,
                manual_seed=args.manual_seed,
                load_model=args.load_model,
                model_arch=args.model_arch,
                dataset_name=args.dataset_name,
                args=args,
                l_rate=args.l_rate,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
                binary_class_index=args.binary_class_index,
                loss_weight=args.loss_weight,
                label_1_percent=args.label_1_percent,
                )


    # End time, time_cost
    print("Start time: ",time.strftime("%Y-%m-%d-%H:%M:%S", start_localtime))
    print("End time: ",time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    end_time_zt=time.time()
    cost_zt=end_time_zt-start_time_zt
    m, s = divmod(cost_zt, 60)
    h, m = divmod(m, 60)
    print ("Time cost: ", "%02d:%02d:%02d" % (h, m, s))
