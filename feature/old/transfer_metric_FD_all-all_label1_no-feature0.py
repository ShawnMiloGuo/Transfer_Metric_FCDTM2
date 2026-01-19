# extract features from image.tif
import torch
import os
import sys
# Get the parent directory of the current file, add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

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
import json
from scipy.linalg import sqrtm
import numpy as np

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

def feature_and_index_batch(iterator_val_loader, model_device, model, label_index = None, no_feature0 = True):
    '''
    OA, F1, miou, precision, recall, feature_source_mean, feature_source_var
    '''
    images, true_masks_cpu = next(iterator_val_loader)
    images = images.to(device=model_device, dtype=torch.float32)
    true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
    with torch.no_grad():
        model_output = model(images)
    predictions = torch.argmax(model_output, dim=1)
    feature_source = feature_from_hook['hook_output'].cpu()
    # extract the feature of label_index
    batch_features = feature_source.permute(0, 2, 3, 1)
    if label_index is not None:
        batch_labels = true_masks_cpu.unsqueeze(1).expand_as(feature_source).permute(0, 2, 3, 1)
        extracted_batch_features = batch_features[batch_labels == label_index].reshape(-1, feature_source.shape[1])
    else:
        extracted_batch_features = batch_features.reshape(-1, feature_source.shape[1])
    
    # no_feature0 = True
    if no_feature0:
        feature_source_mean = torch.zeros(extracted_batch_features.shape[1], dtype=torch.float32)
        feature_source_var = torch.zeros(extracted_batch_features.shape[1], dtype=torch.float32)
        for i in range(extracted_batch_features.shape[1]):
            feature_mask = extracted_batch_features[:, i] != 0.0
            masked_feature = extracted_batch_features[feature_mask, i]
            feature_source_mean[i] = torch.mean(masked_feature)
            feature_source_var[i] = torch.var(masked_feature)
    else:
        feature_source_mean = torch.mean(extracted_batch_features, dim=0)
        feature_source_var = torch.var(extracted_batch_features, dim=0)
    
    OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2)
    return OA, F1, miou, precision, recall, feature_source_mean, feature_source_var, np.array(extracted_batch_features, dtype=np.float32)

def feature_and_index_all_batch(val_loader, model_device, model,  label_index = None, no_feature0 = True):
    '''
    OA, F1, miou, precision, recall, feature_source_mean, feature_source_var
    '''
    index_list = []
    num_image = 0
    feature_mean_batch_list = []
    extracted_features = []
    for i in tqdm(range(len(val_loader)), desc="features_all_batch"):
        images, true_masks_cpu = next(val_loader)
        # count the number of images
        num_image += images.shape[0]
        images = images.to(device=model_device, dtype=torch.float32)
        true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
        with torch.no_grad():
            model_output = model(images)
        predictions = torch.argmax(model_output, dim=1)
        feature_source = feature_from_hook['hook_output'].cpu()
        # extract the feature of label_index
        batch_features = feature_source.permute(0, 2, 3, 1)
        if label_index is not None:
            batch_labels = true_masks_cpu.unsqueeze(1).expand_as(feature_source).permute(0, 2, 3, 1)
            extracted_batch_features = batch_features[batch_labels == label_index].reshape(-1, feature_source.shape[1])
        else:
            extracted_batch_features = batch_features.reshape(-1, feature_source.shape[1])
        extracted_features.extend(extracted_batch_features.tolist())
        
        
        OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2)
        index_list.append([OA, F1, miou, precision, recall])
        # feature_mean_batch_list.append(torch.mean(feature_source, dim=[0,2,3]).flatten().cpu())
        
        # if num_image > 150:
        if num_image > 100:
            break
    
    extracted_features_tensor = torch.tensor(extracted_features, dtype=torch.float32)
    # no_feature0 = True
    if no_feature0:
        feature_mean = torch.zeros(extracted_features_tensor.shape[1], dtype=torch.float32)
        feature_var = torch.zeros(extracted_features_tensor.shape[1], dtype=torch.float32)
        for i in range(extracted_features_tensor.shape[1]):
            feature_mask = extracted_features_tensor[:, i] != 0.0
            masked_feature = extracted_features_tensor[feature_mask, i]
            feature_mean[i] = torch.mean(masked_feature)
            feature_var[i] = torch.var(masked_feature)
    else:
        feature_mean = torch.mean(extracted_features_tensor, dim=0)
        feature_var = torch.var(extracted_features_tensor, dim=0)
    
    # feature_var = feature_mean
    OA_all, F1_all, miou_all, precision_all, recall_all = torch.tensor(index_list).mean(dim=0).tolist()
    return OA_all, F1_all, miou_all, precision_all, recall_all, feature_mean, feature_var, np.array(extracted_features, dtype=np.float32)

def frechet_distance(mu_s, mu_t, sigma_s, sigma_t):
    '''
    mu_s, mu_t: mean of source and target
    sigma_s, sigma_t: covariance matrix of source and target
    '''
    # F范数
    mean_F_norm = torch.norm(mu_s - mu_t, p=2).item()
    # 方差
    tr_cov = torch.trace(sigma_s + sigma_t - 2 * torch.sqrt(sigma_s @ sigma_t))
    tr_cov = 0.0 if torch.isnan(tr_cov) else tr_cov.item()
    # 保证tr_cov为正数
    tr_cov = abs(tr_cov)
    return mean_F_norm + tr_cov

# calculate frechet inception distance
def calculate_fid(act1, act2):
    """
    act1, numpy array from source domain. 
    act2, numpy array from target domain. 
    """
	# calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
	# calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_fid_mask_0(act1, act2):
    """
    act1, numpy array from source domain. 
    act2, numpy array from target domain. 
    """
    masked_act1 = np.ma.array(act1, mask=np.where(act1 == 0.0, True, False))
    masked_act2 = np.ma.array(act2, mask=np.where(act2 == 0.0, True, False))
    # calculate mean and covariance statistics
    mu1, sigma1 = np.ma.mean(masked_act1, axis=0), np.ma.cov(masked_act1, rowvar=False)
    mu2, sigma2 = np.ma.mean(masked_act2, axis=0), np.ma.cov(masked_act2, rowvar=False)
    mu1, sigma1 = np.ma.filled(mu1, 0.0), np.ma.filled(sigma1, 0.0)
    mu2, sigma2 = np.ma.filled(mu2, 0.0), np.ma.filled(sigma2, 0.0)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
    
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

def draw_scatter_all_batch(result_list: list,
                           x_col = 19, y_col=9,
                           y_label = 'y_label', 
                           x_title = "mean_difference", y_title = "acc",
                           result_path = "./", figname = "test.png",
                           save_fig = True
                           ):
    # Create a new figure
    # plt.figure()

    # scatter
    # Extract the metric column as x
    x = [row[x_col] for row in result_list]

    # Extract the acc columns as y
    y = [row[y_col] for row in result_list]

    size_point = max(2, min(1000.0/len(x), 20))
    print(f"draw_scatter(): size_point = {size_point}")
    
    plt.scatter(x, y, label=y_label, s=size_point)

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # title
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    # save the plot
    # plt.show()
    if save_fig:
        test_path_exist(result_path)
        plt.savefig(os.path.join(result_path, figname), bbox_inches='tight')

    # Close the figure
    # plt.close()   
    
def transfer_metric_all_batch(model_path, model_chdir,
         data_path_source, data_path_target,
         batch_size: int=1,
         binary_class_index = 1, 
         label_1_percent = 0.2,
         append_pre:list = ["source", "target", "class_index", "class_name"],
         result_list:str = [],
         target_domain_all = False,
         no_feature0 = True,
         ):
    # Load the model
    model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.chdir(model_chdir)
    model = torch.load(model_path)
    model.eval()

    global feature_from_hook
    feature_from_hook = {}

    # layer_name 
    """
    'up4', -> [1, 64, 256, 256]         
    'outc', -> [1, 9, 256, 256]
    'down4', [1, 512, 32, 32] -> [1, 1024, 16, 16]
    """
    layer_name = 'up4'
    # layer_name = 'down4'
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
    
    iterator_val_loader_source = iter(val_loader_source)
    iterator_val_loader_target = iter(val_loader_target)
    
    # get OA, F1, ...
    # get mean difference, variance difference, ...

    # extract features of specified label_index
    extracted_features_label_index = None # all features of foreground and background
    # label_index = 1 # foreground

    # source domain 
    OA_s, F1_s, miou_s, precision_s, recall_s, mean_s, var_s, all_extracted_features_s = feature_and_index_all_batch(iterator_val_loader_source, 
                                                                                                                     model_device, model, 
                                                                                                                     label_index=extracted_features_label_index,
                                                                                                                     no_feature0=no_feature0)
    all_features_s_permute = all_extracted_features_s
    print(f"all_features_s_permute.shape: {all_features_s_permute.shape}")
    # target domain
    # target_domain_all = False
    if target_domain_all:
        OA_t, F1_t, miou_t, precision_t, recall_t, mean_t, var_t, all_extracted_features_t = feature_and_index_all_batch(iterator_val_loader_target, 
                                                                                                                         model_device, model, 
                                                                                                                         label_index=extracted_features_label_index,
                                                                                                                         no_feature0=no_feature0)
    for i in tqdm(range(len(val_loader_target)), desc="features_one_batch"):
        if not target_domain_all:
            OA_t, F1_t, miou_t, precision_t, recall_t, mean_t, var_t, all_extracted_features_t = feature_and_index_batch(iterator_val_loader_target, 
                                                                                                                         model_device, model, 
                                                                                                                         label_index=extracted_features_label_index,
                                                                                                                         no_feature0=no_feature0)
        # mean, var, ...
        # mean_difference = (mean_s - mean_t).abs().mean().item()
        # F范数
        mean_F_norm = torch.norm(mean_s - mean_t, p=2).item()
        # mean_F_norm = torch.norm(mean_s - mean_t, p=2).pow(2).item()
        var_difference = (var_s - var_t).abs().mean().item()

        # calculate_fid
        all_features_t_permute = all_extracted_features_t
        # fid = calculate_fid(all_features_s_permute.numpy(), all_features_t_permute.numpy())
        # fid = calculate_fid(all_features_s_permute, all_features_t_permute)
        fid = calculate_fid_mask_0(all_features_s_permute, all_features_t_permute)

        if i != 0:
            append_pre = ["", f"{i+1}", "", ""]
        result_list.append(append_pre + 
                        [OA_s, F1_s, miou_s, precision_s, recall_s, 
                            OA_t, F1_t, miou_t, precision_t, recall_t, 
                            OA_s-OA_t, F1_s-F1_t, miou_s-miou_t, precision_s-precision_t, recall_s-recall_t,
                            mean_F_norm, var_difference,
                            fid])
        
        if target_domain_all:
            break
    
    # Remove the hook
    handle.remove()
    return result_list

    
def main(model_path_prefix,
         data_path_source, data_path_target,
         result_path,
         batch_size: int = 1,
         label_1_percent = 0.2,
         binary_class_index_list = [i for i in range(1, 9)],
         dataset_i_list = [0, 1],
         target_domain_all = False,
         no_feature0=True,
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
                         "mean_F_norm", "var_difference", # 19-20
                         "FID", # 21
                         ] 
    binary_class_name_list = ["background", "Cropland", "Forest", "Grassland", "Shrubland", "Wetland", "Water", "Built-up", "Bareland"]
    # result_list
    # result_list = []
    result_list_dict = {}

    dataset_name_source_list = ["dwq_sentinel2", "xj_sentinel2"]
    dataset_name_target_list = dataset_name_source_list[::-1]
    # binary_class_index = 1
    for dataset_i in dataset_i_list:
        dataset_name_source = dataset_name_source_list[dataset_i]
        dataset_name_target = dataset_name_target_list[dataset_i]
        if dataset_i == 1:
            data_path_source, data_path_target = data_path_target, data_path_source
        for binary_class_index in binary_class_index_list:
            print(f"\n\ndataset_name_source: {dataset_name_source}, dataset_name_target: {dataset_name_target}, binary_class_index: {binary_class_index}")
            # result of per class
            dict_key = dataset_name_source + f"_cls_{binary_class_index}"
            result_list_dict[dict_key] = []

            # find the model file
            model_path = model_path_prefix + rf"\train_{dataset_name_source}_cls_{binary_class_index}\unet_*_best_val.pth"
            model_files = glob.glob(model_path)
            if len(model_files) == 0:
                print(f"model_files: {model_files} is empty!")
                continue
            print(f"model_files: {model_files}")

            # all-batch
            transfer_metric_all_batch(model_path=model_files[0], model_chdir=model_chdir,
                                      data_path_source=data_path_source, data_path_target=data_path_target,
                                      batch_size=batch_size, binary_class_index=binary_class_index, 
                                      label_1_percent=label_1_percent, 
                                      append_pre=[dataset_name_source, dataset_name_target, binary_class_index, binary_class_name_list[binary_class_index]],
                                      result_list=result_list_dict[dict_key],
                                      target_domain_all=target_domain_all,
                                      no_feature0=no_feature0)

    # write .csv
    result_csv_name = f"result_{dataset_name_source_list[0]}-{dataset_name_target_list[0]}_batch{batch_size}.csv"
    result_csv_path = os.path.join(result_path, result_csv_name)
    with open(result_csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(result_list_name) 
        for key in result_list_dict.keys():
            result_list = result_list_dict[key]
            writer.writerows(result_list)
    
    # save the result_list_dict
    result_list_dict_name = f"result_list_dict_{dataset_name_source_list[0]}-{dataset_name_target_list[0]}_batch{batch_size}.json"
    result_list_dict_path = os.path.join(result_path, result_list_dict_name)
    with open(result_list_dict_path, "w") as file:
        # Use json.dump to write dict_data to a file
        json.dump(result_list_dict, file)
    # with open('filename.json', 'r') as f:
    #     dict_data = json.load(f)
        
    # draw_scatter_all-batch
    for metric_i in range(19,22):
        for accuracy_i in range(9, 19):
            # Create a new figure
            plt.figure()
            for dataset_i in dataset_i_list:
                save_fig = False
                for binary_class_index in binary_class_index_list:
                    result_list = result_list_dict[dataset_name_source_list[dataset_i] + f"_cls_{binary_class_index}"]
                    # save the last figure
                    if binary_class_index == binary_class_index_list[-1]:
                        save_fig = True
                    draw_scatter_all_batch(result_list, x_col=metric_i, y_col=accuracy_i,
                                           y_label=f"{dataset_name_source_list[dataset_i]}-{dataset_name_target_list[dataset_i]}_cls-{binary_class_index}-{binary_class_name_list[binary_class_index]}",
                                           x_title=result_list_name[metric_i], y_title=result_list_name[accuracy_i],
                                           result_path=os.path.join(result_path, "fig"),
                                           figname=f"draw_{dataset_name_source_list[dataset_i]}-{dataset_name_target_list[dataset_i]}_cls-{binary_class_index}-{binary_class_name_list[binary_class_index]}_batch{batch_size}_{result_list_name[metric_i]}_{result_list_name[accuracy_i]}.png",
                                           save_fig=save_fig)
            plt.close()
    

if __name__ == "__main__":
    model_path_prefix = r"E:\Yiling\at_SIAT_research\2_model_pth" # E:\Yiling\at_SIAT_research\z_result\20240516_1_train_binary_weight
    # binary_class_index_list=[i for i in range(1, 9)]
    binary_class_index_list=[1, 2, 3, 6, 7, 8]

    data_path_source = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val'
    data_path_target = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val'

    # E:\Yiling\at_SIAT_research\z_result\20240
    # result_path = r"E:\Yiling\at_SIAT_research\z_result\20240604_transfer_metric_FID\20240604_transfer_metric_FID_all-batch_label1_1_dwqs2-xjs2_batch8_200img"
    result_path = r"E:\Yiling\at_SIAT_research\z_result\20240624_transfer_metric_FID_no_feature0\20240626_transfer_metric_FID_mask0_decoder_all-batch_1_dwqs2-xjs2_batch4_100img_no_feature0"
    
    save_log(result_path=result_path, log_name="transfer_metric_FID_feature0_dwqs2-xjs2.log")

    main(model_path_prefix=model_path_prefix,
         data_path_source=data_path_source, data_path_target=data_path_target,
         result_path=result_path,
         batch_size=4,
         label_1_percent=0.2,
         binary_class_index_list=binary_class_index_list,
         dataset_i_list=[0, 1], # "dwq_sentinel2", "xj_sentinel2"
         target_domain_all=False, # True: all, False: batch
         no_feature0=True, # True: no feature 0, False: all features
         )


