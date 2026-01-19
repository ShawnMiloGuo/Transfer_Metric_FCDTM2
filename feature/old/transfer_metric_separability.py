# extract features from image.tif
import torch
import os
import sys
# Get the parent directory of the current file, add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from component.utils import test_path_exist
from tqdm import tqdm
from component.dataset import get_dataset_reader
from torch.utils import data
import csv
import matplotlib.pyplot as plt
import glob
from component.utils import save_log
import json
import numpy as np
import argparse

from transfer_metric_FD_mask0_all_all_label1_no_feature0 import all_index, draw_scatter_all_batch 
from gbc import get_gbc_score

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
    
    OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2, ignore_label_0=False if label_index is None else True)
    return OA, F1, miou, precision, recall, feature_source_mean, feature_source_var, np.array(extracted_batch_features, dtype=np.float32)

def feature_and_index_all_batch(val_loader, model_device, model, label_index = None, no_feature0 = True):
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
        
        
        OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2, ignore_label_0=False if label_index is None else True)
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

# saparate the feature to foreground and background by prediciton
def feature_list_from_pred_and_index_all_batch(val_loader, model_device, model, no_feature0 = False):
    '''
    OA_all, F1_all, miou_all, precision_all, recall_all, feature_mean_all, feature_mean_dict, extracted_features_dict
    '''
    index_list = []
    num_image = 0
    extracted_features_dict = {}
    num_classes = 2
    for i in range(num_classes):
        extracted_features_dict[i] = []

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
        batch_pred = predictions.cpu().unsqueeze(1).expand_as(feature_source).permute(0, 2, 3, 1)
        for i in range(num_classes):
            extracted_batch_features = batch_features[batch_pred == i].reshape(-1, feature_source.shape[1])
            extracted_features_dict[i].extend(extracted_batch_features.tolist())
        
        OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2)
        index_list.append([OA, F1, miou, precision, recall])

        # if num_image > 150:
        if num_image > 100:
            break
    extracted_features_tensor_dict = {}
    feature_mean_dict = {}
    for i in range(num_classes):
        extracted_features_tensor_dict[i] = torch.tensor(extracted_features_dict[i], dtype=torch.float32)
        feature_mean_dict[i] = torch.mean(extracted_features_tensor_dict[i], dim=0)

    feature_mean_all = torch.mean(torch.cat([extracted_features_tensor_dict[i] for i in range(num_classes)], dim=0), dim=0)
    
    OA_all, F1_all, miou_all, precision_all, recall_all = torch.tensor(index_list).mean(dim=0).tolist()
    return OA_all, F1_all, miou_all, precision_all, recall_all, feature_mean_all, feature_mean_dict, extracted_features_dict

def feature_list_from_pred_and_index_one_batch(val_loader, model_device, model, no_feature0 = False):
    '''
    OA_all, F1_all, miou_all, precision_all, recall_all, feature_mean_all, feature_mean_dict, extracted_features_dict
    '''
    index_list = []
    num_image = 0
    extracted_features_dict = {}
    num_classes = 2
    for i in range(num_classes):
        extracted_features_dict[i] = []

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
    batch_pred = predictions.cpu().unsqueeze(1).expand_as(feature_source).permute(0, 2, 3, 1)
    for i in range(num_classes):
        extracted_batch_features = batch_features[batch_pred == i].reshape(-1, feature_source.shape[1])
        extracted_features_dict[i].extend(extracted_batch_features.tolist())
    
    OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2)
    index_list.append([OA, F1, miou, precision, recall])

    extracted_features_tensor_dict = {}
    feature_mean_dict = {}
    for i in range(num_classes):
        extracted_features_tensor_dict[i] = torch.tensor(extracted_features_dict[i], dtype=torch.float32)
        feature_mean_dict[i] = torch.mean(extracted_features_tensor_dict[i], dim=0)

    feature_mean_all = torch.mean(torch.cat([extracted_features_tensor_dict[i] for i in range(num_classes)], dim=0), dim=0)
    
    OA_all, F1_all, miou_all, precision_all, recall_all = torch.tensor(index_list).mean(dim=0).tolist()
    return OA_all, F1_all, miou_all, precision_all, recall_all, feature_mean_all, feature_mean_dict, extracted_features_dict

def feature_label_and_index_all_batch(val_loader, model_device, model, one_batch = False):
    '''
    OA_all, F1_all, miou_all, precision_all, recall_all, extracted_features, extracted_labels
    '''
    index_list = []
    num_image = 0
    extracted_features = []
    extracted_labels = []

    def one_batch_iter():
        images, true_masks_cpu = next(val_loader)
        # count the number of images
        nonlocal num_image
        num_image += images.shape[0]
        images = images.to(device=model_device, dtype=torch.float32)
        true_masks = true_masks_cpu.to(device=model_device, dtype=torch.long)
        with torch.no_grad():
            model_output = model(images)
        predictions = torch.argmax(model_output, dim=1)
        feature_source = feature_from_hook['hook_output'].cpu()
        # extract the feature and label
        batch_features = feature_source.permute(0, 2, 3, 1)
        batch_labels = true_masks_cpu
        extracted_features.extend(batch_features.reshape(-1, feature_source.shape[1]).tolist())
        extracted_labels.extend(batch_labels.reshape(-1).tolist())
        
        OA, F1, miou, precision, recall = all_index(predictions, true_masks, num_classes=2)
        index_list.append([OA, F1, miou, precision, recall])

    if one_batch:
        one_batch_iter()
    else:
        for i in tqdm(range(len(val_loader)), desc="features_all_batch"):
            one_batch_iter()
            # if num_image > 150:
            # if num_image > 100:
            if num_image > 50:
                break
    
    OA_all, F1_all, miou_all, precision_all, recall_all = torch.tensor(index_list).mean(dim=0).tolist()
    return OA_all, F1_all, miou_all, precision_all, recall_all, extracted_features, extracted_labels

# calculate Dispersion_score
def calculate_Dispersion_score(mean_all, mean_dict, num_samples_dict):
    num_classes = len(mean_dict)
    weighted_sum = 0.0
    for i in range(num_classes):
        weighted_sum += num_samples_dict[i] * torch.norm(mean_all - mean_dict[i], p=2).item()
    score = weighted_sum / (num_classes - 1)
    log_score = np.log(score).item()
    return score, log_score


def transfer_metric_all_batch_Ds(model_path, model_chdir,
         data_path_source, data_path_target,
         batch_size: int=1,
         binary_class_index = 1, 
         label_1_percent = 0.2,
         append_pre:list = ["source", "target", "class_index", "class_name"],
         result_list:str = [],
         target_domain_all = False,
         no_feature0 = True,
         feature_layer_name = 'up4',
         only_label_1 = False,
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
    layer_name = feature_layer_name
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
    # 1: foreground, None: all features of foreground and background
    extracted_features_label_index = 1 if only_label_1 else None

    # source domain 
    OA_s, F1_s, miou_s, precision_s, recall_s, feature_mean_all_s, feature_mean_dict_s, all_extracted_features_dict_s = \
          feature_list_from_pred_and_index_all_batch(iterator_val_loader_source, model_device, model)
    # all_features_s_permute = all_extracted_features_s # [N, feature_dim]
    # print(f"all_features_s_permute.shape: {all_features_s_permute.shape}")

    # target domain
    # target_domain_all = False
    if target_domain_all:
        OA_t, F1_t, miou_t, precision_t, recall_t, feature_mean_all_t, feature_mean_dict_t, all_extracted_features_dict_t = \
              feature_list_from_pred_and_index_all_batch(iterator_val_loader_target, model_device, model)
    for i in tqdm(range(len(val_loader_target)), desc="features_one_batch"):
        if not target_domain_all:
            OA_t, F1_t, miou_t, precision_t, recall_t, feature_mean_all_t, feature_mean_dict_t, all_extracted_features_dict_t = \
                  feature_list_from_pred_and_index_one_batch(iterator_val_loader_target, model_device, model)
        
        # calculate Dispersion_score
        num_classes = 2
        num_samples_dict_t = {i: len(all_extracted_features_dict_t[i]) for i in range(num_classes)}
        target_Dispersion_score_no_log, target_Dispersion_score = calculate_Dispersion_score(feature_mean_all_t, feature_mean_dict_t, num_samples_dict_t)
        

        if i != 0:
            append_pre = ["", f"{i+1}", "", ""]
        result_list.append(append_pre + 
                        [OA_s, F1_s, miou_s, precision_s, recall_s, 
                            OA_t, F1_t, miou_t, precision_t, recall_t, 
                            OA_s-OA_t, F1_s-F1_t, miou_s-miou_t, precision_s-precision_t, recall_s-recall_t,
                            target_Dispersion_score_no_log, target_Dispersion_score])
        
        if target_domain_all:
            # target_feature_0_percent = np.sum(all_features_t_permute == 0.0, axis=0) / all_features_t_permute.shape[0]
            # print(f"target_feature_0_percent: {target_feature_0_percent}")
            # print(f"target_feature_0_percent_mean: {np.mean(target_feature_0_percent)}")
            break
    
    # Remove the hook
    handle.remove()
    return result_list

def transfer_metric_all_batch(model_path, model_chdir,
         data_path_source, data_path_target,
         batch_size: int=1,
         binary_class_index = 1, 
         label_1_percent = 0.2,
         append_pre:list = ["source", "target", "class_index", "class_name"],
         result_list:str = [],
         target_domain_all = False,
         no_feature0 = True,
         feature_layer_name = 'up4',
         only_label_1 = False,
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
    layer_name = feature_layer_name
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
    # 1: foreground, None: all features of foreground and background
    extracted_features_label_index = 1 if only_label_1 else None

    # source domain 
    OA_s, F1_s, miou_s, precision_s, recall_s, extracted_features_s, extracted_labels_s = \
          feature_label_and_index_all_batch(iterator_val_loader_source, model_device, model, one_batch=False)
    # all_features_s_permute = all_extracted_features_s # [N, feature_dim]
    # print(f"all_features_s_permute.shape: {all_features_s_permute.shape}")

    # target domain
    # target_domain_all = False
    if target_domain_all:
        OA_t, F1_t, miou_t, precision_t, recall_t, extracted_features_t, extracted_labels_t = \
              feature_label_and_index_all_batch(iterator_val_loader_target, model_device, model, one_batch=False)
    for i in tqdm(range(len(val_loader_target)), desc="features_one_batch"):
        if not target_domain_all:
            OA_t, F1_t, miou_t, precision_t, recall_t, extracted_features_t, extracted_labels_t = \
                  feature_label_and_index_all_batch(iterator_val_loader_target, model_device, model, one_batch=True)
        
        # calculate Dispersion_score
        # todo

        #calculate GBC ('diagonal', 'spherical')
        diagonal_GBC = get_gbc_score(extracted_features_t, extracted_labels_t, 'diagonal')
        spherical_GBC = get_gbc_score(extracted_features_t, extracted_labels_t, 'spherical')
        

        if i != 0:
            append_pre = ["", f"{i+1}", "", ""]
        result_list.append(append_pre + 
                        [OA_s, F1_s, miou_s, precision_s, recall_s, 
                            OA_t, F1_t, miou_t, precision_t, recall_t, 
                            OA_s-OA_t, F1_s-F1_t, miou_s-miou_t, precision_s-precision_t, recall_s-recall_t,
                            diagonal_GBC, spherical_GBC])
        
        if target_domain_all:
            # target_feature_0_percent = np.sum(all_features_t_permute == 0.0, axis=0) / all_features_t_permute.shape[0]
            # print(f"target_feature_0_percent: {target_feature_0_percent}")
            # print(f"target_feature_0_percent_mean: {np.mean(target_feature_0_percent)}")
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
         feature_layer_name = 'up4', # 'up4', 'outc', 'down4'
         only_label_1 = False,
         ):
    
    model_chdir = r'C:\Users\ZT\OneDrive\Studying\Yiling\00Research\202401_experiment\code'
    test_path_exist(result_path)

    print(f"label_1_percent: {label_1_percent}")
    
    # result_list_name
    # dwqs2-xjs2_cls1_mean-difference_delta-OA
    # result_list_name =["source", "target", "class_index", "class_name",  # 0-3
    #                      "OA_s", "F1_s", "miou_s", "precision_s", "recall_s", # 4-8
    #                      "OA_t", "F1_t", "miou_t", "precision_t", "recall_t", # 9-13
    #                      "OA_delta", "F1_delta", "miou_delta", "precision_delta", "recall_delta", # 14-18
    #                      "mean_F_norm", "mean_sum", "var_sum", # 19-21
    #                      "FID", # 22
    #                      ] 
    result_list_name =["source", "target", "class_index", "class_name",  # 0-3
                         "OA_s", "F1_s", "miou_s", "precision_s", "recall_s", # 4-8
                         "OA_t", "F1_t", "miou_t", "precision_t", "recall_t", # 9-13
                         "OA_delta", "F1_delta", "miou_delta", "precision_delta", "recall_delta", # 14-18
                         "diagonal_GBC", "spherical_GBC", # 19-20
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
                                      no_feature0=no_feature0,
                                      feature_layer_name = feature_layer_name,
                                      only_label_1=only_label_1,
                                      )

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
    for metric_i in range(19,21):
        # for accuracy_i in [9, 10, 11, 14, 15, 16, 17, 18]:
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
    
def get_args():
    parser = argparse.ArgumentParser(description='Hyperparams for transfer_metric.')
    parser.add_argument('--model_path_prefix', type=str, default=r"E:\Yiling\at_SIAT_research\2_model_pth", help='prefix of model_path')
    # data_path_source, data_path_target
    parser.add_argument('--data_path_source', type=str, help='Path to source datasets, including images/train & val, annotations/train & val', 
                        default=r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val')
    parser.add_argument('--data_path_target', type=str, help='Path to target datasets, including images/train & val, annotations/train & val',
                        default=r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val')
    # result_path, log_name
    parser.add_argument('--result_path', type=str, help='Path to RESULT',
                        default=r'E:\Yiling\at_SIAT_research\z_result\20240627_transfer_metric_FID-mask0\20240627_transfer_metric_FID_decoder_1_all-all_dwqs2-xjs2_100img')
    parser.add_argument('--log_name', type=str, default="transfer_metric_FID_dwqs2-xjs2.log", help='log_name')
    # batch_size, label_1_percent
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--label_1_percent', type=float, default=0.2, help='which the percentage of label_1 greater than ')
    # target_domain_all, no_feature0
    parser.add_argument('--target_domain_all', type=int, default=1, help='1: all, 0: batch')
    parser.add_argument('--no_feature0', type=int, default=0, help='1: no feature 0, 0: all features')
    # feature_layer_name
    parser.add_argument('--feature_layer_name', type=str, default='up4', help="name of the specified unet feature_layer, decoder -> 'up4', encoder -> 'down4'")
    # only_label_1
    parser.add_argument('--only_label_1', type=int, default=0, help='1: only label 1, 0: all features')
    
    # binary_class_index_list
    # parser.add_argument('--binary_class_index_list', type=int, nargs='+', default=[1, 2, 3, 6, 7, 8], help='The class index of binary classification')
    # parser.add_argument('--dataset_i_list', type=int, nargs='+', default=[0, 1], help='The index of dataset')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    binary_class_index_list=[1, 2, 3, 6, 7, 8]
    
    save_log(result_path=args.result_path, log_name=args.log_name, args=args)

    main(model_path_prefix = args.model_path_prefix,
         data_path_source = args.data_path_source, data_path_target = args.data_path_target,
         result_path = args.result_path,
         batch_size = args.batch_size, label_1_percent = args.label_1_percent,
         binary_class_index_list = binary_class_index_list,
         dataset_i_list = [0, 1], # "dwq_sentinel2", "xj_sentinel2"
         target_domain_all = args.target_domain_all, # True: all, False: batch
         no_feature0 = args.no_feature0, # True: no feature 0, False: all features
         feature_layer_name = args.feature_layer_name, # 'up4', 'outc', 'down4'
         only_label_1 = args.only_label_1, # 1: only label 1, 0: all features
         )


