# extract features from image.tif
import torch
import os
import sys
sys.path.append("../")
from component.utils import test_path_exist
from torchmetrics.functional.classification import multiclass_precision
from torchmetrics.functional.classification import multiclass_recall
from torchmetrics.functional.classification import multiclass_f1_score
from tqdm import tqdm
from component.dataset import get_dataset_reader
from torch.utils import data
import csv
import matplotlib.pyplot as plt
import glob
from component.utils import save_log

def all_index(predictions, labels, num_classes):
    '''
    OA, F1, miou, precision, recall
    '''
    predictions = predictions.flatten()
    labels = labels.flatten()
    OA = (predictions == labels).float().mean().item()
    miou, ious = mean_iou(predictions, labels, num_classes)

    precision = multiclass_precision(predictions, labels, num_classes, "macro").item()
    recall = multiclass_recall(predictions, labels, num_classes, "macro").item()
    F1 = multiclass_f1_score(predictions, labels, num_classes, "macro").item()
    return OA, F1, miou, precision, recall
def mean_iou(predictions, labels, num_classes):
    # 计算平均 Intersection over Union（mIOU）
    # 初始化混淆矩阵
    confusion_mat = torch.zeros((num_classes, num_classes), device=predictions.device)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_mat[i, j] = torch.sum((labels == i) & (predictions == j))
    # 计算每个类别的 IOU
    ious = []
    for i in range(num_classes):
        intersection = confusion_mat[i, i]
        union = torch.sum(confusion_mat[i, :]) + torch.sum(confusion_mat[:, i]) - intersection
        if union == 0:
            iou = torch.tensor(0.0)  # 如果该类别在真实标签中不存在，则 IOU 为 0
        else:
            iou = intersection / union
        ious.append(iou.item())
    # 计算 mIOU
    mean_iou = torch.mean(torch.tensor(ious)).item()
    return mean_iou, ious

# Register the hook
def hook(module, input, output):
    feature_from_hook['hook_output'] = output
    # print(output)

def feature_and_index_batch(iterator_val_loader, model_device, model):
    '''
    OA, F1, miou, precision, recall, feature_source_mean, feature_source_var
    '''
    images, true_masks_cpu = next(iterator_val_loader)
    images = images.to(device=model_device, dtype=torch.float32)
    true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
    with torch.no_grad():
        model_output = model(images)
    predictions = torch.argmax(model_output, dim=1)
    feature_source = feature_from_hook['hook_output']
    feature_source_mean = torch.mean(feature_source, dim=[0,2,3]).flatten().cpu()
    feature_source_var = torch.var(feature_source, dim=[0,2,3]).flatten().cpu()
    # if i == 0:
    #     feature_from_hook['all_hook_output_source'] = feature_source
    # else:
    #     feature_from_hook['all_hook_output_source'] = torch.cat([feature_from_hook['all_hook_output_source'], feature_source], dim = -1)
    #     print(f'all_hook_output_source.shape:', feature_from_hook['all_hook_output_source'].shape)
    
    OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2)
    return OA, F1, miou, precision, recall, feature_source_mean, feature_source_var

def feature_and_index_all_batch(val_loader, model_device, model, source_or_target = 'source'):
    '''
    OA, F1, miou, precision, recall, feature_source_mean, feature_source_var
    '''
    all_feature_list_name = 'all_hook_output_source' if source_or_target == 'source' else 'all_hook_output_target'
    feature_from_hook[all_feature_list_name] = []
    index_list = []
    num_image = 0
    feature_mean_batch_list = []
    for i in tqdm(range(len(val_loader))):
        images, true_masks_cpu = next(val_loader)
        # count the number of images
        num_image += images.shape[0]
        images = images.to(device=model_device, dtype=torch.float32)
        true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
        with torch.no_grad():
            model_output = model(images)
        predictions = torch.argmax(model_output, dim=1)
        feature_source = feature_from_hook['hook_output'].cpu()
        if num_image < 500:
            feature_from_hook[all_feature_list_name].append(feature_source)
            print(f'i={i}, {all_feature_list_name}.shape:', len(feature_from_hook[all_feature_list_name]), feature_from_hook[all_feature_list_name][-1].shape)
        OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2)
        index_list.append([OA, F1, miou, precision, recall])
        feature_mean_batch_list.append(torch.mean(feature_source, dim=[0,2,3]).flatten().cpu())
    feature_mean = torch.mean(torch.stack(feature_mean_batch_list), dim=0).flatten()
    all_features = torch.cat(feature_from_hook[all_feature_list_name], dim=0)
    feature_var = torch.var(all_features, dim=[0,2,3]).flatten().cpu()
    OA_all, F1_all, miou_all, precision_all, recall_all = torch.tensor(index_list).mean(dim=0).tolist()
    return OA_all, F1_all, miou_all, precision_all, recall_all, feature_mean, feature_var


def draw_scatter(result_list: list, 
                 x_col = 16, y_col=[i for i in range(6,11)], 
                 y_label = [f'Column {i}' for i in range(6,11)], 
                 x_title = "mean_difference", 
                 result_path = "./", 
                 figname = "test.png"):
    # Create a new figure
    plt.figure()

    # scatter
    # Extract the first column as x
    x = [row[x_col] for row in result_list]

    # Extract the rest of the columns as y
    ys = [[row[i] for row in result_list] for i in y_col]

    size_point = max(2, min(1000.0/len(x), 20))
    print(f"draw_scatter(): size_point = {size_point}")
    # Create a scatter plot for each y
    for i, y in enumerate(ys):
        plt.scatter(x, y, label=y_label[i], s=size_point)

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # title
    plt.xlabel(x_title)

    # Show the plot
    # plt.show()
    test_path_exist(result_path)
    plt.savefig(os.path.join(result_path, figname), bbox_inches='tight')

    # Close the figure
    plt.close()

def draw_scatter_each_row(result_list: list,
                          x_col = 19, y_col=9,
                          x_title = "mean_difference",
                          y_title = "OA_t",
                          result_path = "./",
                          figname = "test.png"
                          ):
    plt.figure()
    size_point = max(2, min(1000.0/len(result_list), 20))
    print(f"draw_scatter_each_row(): size_point = {size_point}")
    for row in range(len(result_list)):
        y_label = f"{result_list[row][0]}-{result_list[row][1]}_cls-{result_list[row][2]}-{result_list[row][3]}"
        plt.scatter(result_list[row][x_col], result_list[row][y_col], label=y_label, s=size_point)
    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # title
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    # save the plot
    test_path_exist(result_path)
    plt.savefig(os.path.join(result_path, figname), bbox_inches='tight')

    # Close the figure
    plt.close()
    return
                     

def transfer_metric_once(model_path, model_chdir,
         data_path_source, data_path_target,
         batch_size: int=1,
         binary_class_index = 1, 
         label_1_percent = 0.0
         ):
    
    # Load the model
    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.chdir(model_chdir)
    model = torch.load(model_path)
    model.eval()

    global feature_from_hook
    feature_from_hook = {}

    # layer_name 
    # 'up4', -> [1, 64, 256, 256]         
    # 'outc', -> [1, 9, 256, 256]
    layer_name = 'up4'
    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(hook)

    # set dataset
    dataset_name = 'rgbn'
    num_workers = 0

    # data loader
    dataset_reader = get_dataset_reader(dataset_name)
    val_dataset_source = dataset_reader(root_dir=data_path_source, is_train=0, transform= None, binary_class_index = binary_class_index, label_1_percent = label_1_percent)
    val_loader_source = data.DataLoader(val_dataset_source, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    val_dataset_target = dataset_reader(root_dir=data_path_target, is_train=0, transform= None, binary_class_index = binary_class_index, label_1_percent = label_1_percent)
    val_loader_target = data.DataLoader(val_dataset_target, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # get OA, F1, ...
    # get mean difference, variance difference, ...
    iterator_val_loader_source = iter(val_loader_source)
    iterator_val_loader_target = iter(val_loader_target)
    # source domain
    OA_s, F1_s, miou_s, precision_s, recall_s, mean_s, var_s = feature_and_index_all_batch(iterator_val_loader_source, model_device, model)
    # target domain
    OA_t, F1_t, miou_t, precision_t, recall_t, mean_t, var_t = feature_and_index_all_batch(iterator_val_loader_target, model_device, model)
    
    # Remove the hook
    handle.remove()
    
    # mean, var, ...
    mean_difference = (mean_s - mean_t).abs().mean().item()
    var_difference = (var_s - var_t).abs().mean().item()

    return [OA_s, F1_s, miou_s, precision_s, recall_s,
            OA_t, F1_t, miou_t, precision_t, recall_t,
            OA_s-OA_t, F1_s-F1_t, miou_s-miou_t, precision_s-precision_t, recall_s-recall_t,
            mean_difference, var_difference]
    
    
def main(model_path_prefix,
         data_path_source, data_path_target,
         result_path,
         label_1_percent = 0.0,
         binary_class_index_list = [i for i in range(1, 9)]
         ):
    
    model_chdir = r'C:\Users\ZT\OneDrive\Studying\Yiling\00Research\202401_experiment\code'
    test_path_exist(result_path)

    print(f"label_1_percent: {label_1_percent}")
    
    # result_list_name
    # dwqs2-xjs2_cls1_mean-difference_delta-OA
    result_list_name =["source", "target", "class_index", "class_name",  # 0-3
                         "OA_s", "F1_s", "miou_s", "precision_s", "recall_s", # 4-8
                         "OA_t", "F1_t", "miou_t", "precision_t", "recall_t", # 9-13
                         "OA_delta", "F1_delta", "miou_delta", "precision_delta", "recall_delta", # 14-18
                         "mean_difference", "var_difference"] # 19-20
    binary_class_name_list = ["background", "Cropland", "Forest", "Grassland", "Shrubland", "Wetland", "Water", "Built-up", "Bareland"]
    # result_list
    result_list = []

    batch_size = 1
    dataset_name_source_list = ["dwq_sentinel2", "xj_sentinel2"]
    dataset_name_target_list = dataset_name_source_list[::-1]
    # binary_class_index = 1
    for dataset_i in range(2):
        dataset_name_source = dataset_name_source_list[dataset_i]
        dataset_name_target = dataset_name_target_list[dataset_i]
        for binary_class_index in binary_class_index_list:
            model_path = model_path_prefix + rf"\train_{dataset_name_source}_cls_{binary_class_index}\unet_*_best_val.pth"
            # find the model file
            model_files = glob.glob(model_path)
            if len(model_files) == 0:
                print(f"model_files: {model_files} is empty!")
                continue
            print(f"model_files: {model_files}")
            transfer_metric_once_result = transfer_metric_once(model_path=model_files[0], model_chdir=model_chdir,
                                                               data_path_source=data_path_source, data_path_target=data_path_target,
                                                               batch_size=batch_size, binary_class_index=binary_class_index, 
                                                               label_1_percent=label_1_percent)
            result_list.append([dataset_name_source, dataset_name_target, 
                                binary_class_index, binary_class_name_list[binary_class_index]] + 
                                transfer_metric_once_result)

    # draw_scatter
    # draw_scatter(result_list, x_col=16, y_col=[i for i in range(6,10)], 
    #              y_label=[result_list_name[i] for i in range(6,10)], 
    #              result_path=os.path.join(result_path, "fig"), figname=f"draw_dwqs2-xjs2_cls{binary_class_index}_batch{batch_size}_mean-target.png")
    # draw_scatter(result_list, x_col=16, y_col=[i for i in range(11,15)], 
    #              y_label=[result_list_name[i] for i in range(11,15)], 
    #              result_path=os.path.join(result_path, "fig"), figname=f"draw_dwqs2-xjs2_cls{binary_class_index}_batch{batch_size}_mean-diff.png")
    # for metric_i in range(19,20):
    #     for accuracy_i in range(6, 16):
    #         draw_scatter(result_list, x_col=metric_i, y_col=[accuracy_i], 
    #                      y_label=[result_list_name[accuracy_i]], x_title=result_list_name[metric_i],
    #                      result_path=os.path.join(result_path, "fig"), figname=f"draw_dwqs2-xjs2_cls{binary_class_index}_batch{batch_size}_{result_list_name[metric_i]}_{result_list_name[accuracy_i]}.png")
    # draw_scatter_each_row
    for metric_i in range(19,20):
        for accuracy_i in range(9, 19):
            draw_scatter_each_row(result_list, x_col=metric_i, y_col=accuracy_i, 
                                  x_title=result_list_name[metric_i], y_title=result_list_name[accuracy_i],
                                  result_path=os.path.join(result_path, "fig"), figname=f"draw_{dataset_name_source_list[0]}-{dataset_name_source_list[1]}_cls-{binary_class_index}-{binary_class_name_list[binary_class_index]}_batch{batch_size}_{result_list_name[metric_i]}_{result_list_name[accuracy_i]}.png")
    
    # write .csv
    result_csv_name = f"result_{dataset_name_source_list[0]}-{dataset_name_source_list[1]}_batch{batch_size}.csv"
    result_csv_path = os.path.join(result_path, result_csv_name)
    with open(result_csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(result_list_name) 
        writer.writerows(result_list)
    

if __name__ == "__main__":
    model_path_prefix = r"E:\Yiling\at_SIAT_research\z_result\20240516_1_train_binary_weight"
    data_path_source = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val'
    data_path_target = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val'

    # E:\Yiling\at_SIAT_research\z_result\20240517_
    result_path = r"E:\Yiling\at_SIAT_research\z_result\20240522_transfer_metric_all-all_1_dwqs2-xjs2"
    
    save_log(result_path=result_path, log_name="transfer_metric_all-all_dwqs2-xjs2.log")

    main(model_path_prefix=model_path_prefix,
         data_path_source=data_path_source, data_path_target=data_path_target,
         result_path=result_path,
         label_1_percent=0.2,
         binary_class_index_list=[1, 3, 6, 7, 8]
         )
    