import os
import sys
# Get the parent directory of the current file, add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
from torch.utils import data
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import numpy as np

from component.utils import test_path_exist
from component.utils import save_log
from component.dataset import get_dataset_reader

# Register the hook
def hook(module, input, output):
    feature_from_hook['hook_output'] = output
    # print(output)

# def extract_features_all_batch(val_loader, model_device, model, label_index = 1):
#     '''
#     all_features
#     '''
#     all_feature_list_name = 'all_hook_output'
#     feature_from_hook[all_feature_list_name] = []
#     label_list = []
#     num_image = 0
#     for i in tqdm(range(len(val_loader))):
#         images, true_masks_cpu = next(val_loader)
#         # count the number of images
#         num_image += images.shape[0]
#         images = images.to(device=model_device, dtype=torch.float32)
#         with torch.no_grad():
#             model_output = model(images)
#         feature_source = feature_from_hook['hook_output'].cpu()
#         feature_from_hook[all_feature_list_name].append(feature_source)
#         label_list.append(true_masks_cpu)
        
#         if num_image > 500:
#             break
            
#     all_features = torch.cat(feature_from_hook[all_feature_list_name], dim=0)
#     feature_from_hook[all_feature_list_name].clear()
#     all_labels = torch.cat(label_list, dim=0).unsqueeze(1).expand_as(all_features)
#     label_list.clear()

#     all_features = all_features.permute(0, 2, 3, 1)
#     all_labels = all_labels.permute(0, 2, 3, 1)
#     extracted_features = all_features[all_labels == label_index]
#     extracted_features = extracted_features.reshape(-1, feature_source.shape[1])

#     return extracted_features

def extract_features_all_batch_low_memory(val_loader, model_device, model, label_index_list = [0, 1],):
    '''
    all_features
    '''
    all_feature_list_name = 'all_hook_output'
    feature_from_hook[all_feature_list_name] = []
    extracted_features_dict = {}
    for label_index in label_index_list:
        extracted_features_dict[label_index] = []
    num_image = 0
    for i in tqdm(range(len(val_loader))):
        images, true_masks_cpu = next(val_loader)
        # count the number of images
        num_image += images.shape[0]
        images = images.to(device=model_device, dtype=torch.float32)
        with torch.no_grad():
            model_output = model(images)
        feature_source = feature_from_hook['hook_output'].cpu()
        batch_features = feature_source.permute(0, 2, 3, 1)
        batch_labels = true_masks_cpu.unsqueeze(1).expand_as(feature_source).permute(0, 2, 3, 1)
        for label_index in label_index_list:
            extracted_batch_features = batch_features[batch_labels == label_index].reshape(-1, feature_source.shape[1])
            extracted_features_dict[label_index].extend(extracted_batch_features.tolist())

        # if num_image > 20:
        # if num_image > 300:
        if num_image > 200:
            break
    for label_index in label_index_list:
        print(f"label_index: {label_index}, extracted_features: {len(extracted_features_dict[label_index])}, {len(extracted_features_dict[label_index][0])}")
        extracted_features_dict[label_index] = np.array(extracted_features_dict[label_index], dtype=np.float32)
    
    # return np.array(extracted_features, dtype=np.float32)
    return extracted_features_dict


# get the rank of the weights of the outc layer
def get_rank_index(lst):
    desc_indices = [i[0] for i in sorted(enumerate(lst), key=lambda x:x[1], reverse=True)]
    ranks = [desc_indices.index(i) for i in range(len(lst))]
    return desc_indices, ranks

def get_distribution_per_dataset(model_path, model_chdir,
                             data_path_source, data_path_target,
                             batch_size: int=1,
                             binary_class_index = 1,
                             label_1_percent = 0.2,
                             label_index_list = [0, 1],
                             get_transfer = False,
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

    # extract features
    # extracted_features_dict = {}
    # label_index_list = [0, 1]
    iterator_val_loader = iter(val_loader_target) if get_transfer else iter(val_loader_source)
    extracted_features_dict = extract_features_all_batch_low_memory(iterator_val_loader, model_device, model, label_index_list)
    
    # Remove the hook
    handle.remove()

    # get outc weight rank
    outc_weight = model.outc.weight.permute(2,3,0,1).reshape(-1,64).cpu()[1].tolist()
    outc_weight_desc_indices, outc_weight_ranks = get_rank_index(outc_weight)
    print(f"outc_weight: {outc_weight}")
    print(f"outc_weight_desc_indices: {outc_weight_desc_indices}")
    print(f"outc_weight_ranks: {outc_weight_ranks}")

    return extracted_features_dict, outc_weight, outc_weight_desc_indices

def draw_normal_distribution(data, label='legend', plt=plt):
    # Create histogram
    # plt.hist(data, bins=30, density=True, alpha=0.6, label=label)
    # bins = len(data) // 100
    bins = 100
    mask = data != 0
    data = data[mask]
    plt.hist(data, bins='auto', density=True, alpha=0.6, label=label)

    # Add a best fit line
    # xmin, xmax = plt.get_xlim()
    # x = np.linspace(xmin, xmax, 100)
    # mu, std = np.mean(data), np.std(data)
    # p = np.exp(-(x-mu)**2 / (2*std**2)) / (np.sqrt(2*np.pi*std**2))
    # plt.plot(x, p, 'k', linewidth=2)
    
    # plt.legend()
def print_sum_num_0(data, index):
    num = sum(data == 0)
    print(f"dim: {index}, num of 0: {num}, total: {len(data)}, percent: {num/len(data)}")
def draw_distribution_all_dim(extracted_features_dict, 
                              result_path,
                              figname='distribution_all_dim.png',
                              outc_weight=[], 
                              outc_weight_desc_indices=[]
                              ):
    # Create a figure and a set of subplots
    size_per_subplot = 4
    col_subplots = 8
    row_subplots = 8
    fig, axs = plt.subplots(col_subplots, row_subplots, figsize=(col_subplots * size_per_subplot, row_subplots * size_per_subplot), sharex=False, sharey=False)
    with tqdm(total=col_subplots * row_subplots, desc="drawing") as pbar:                
        for col in range(col_subplots):
            for row in range(row_subplots):
                dim_index = col * row_subplots + row
                data_0 = extracted_features_dict[0][:, outc_weight_desc_indices[dim_index]]
                #
                print_sum_num_0(data_0, dim_index)
                draw_normal_distribution(data_0, label='label_0', plt=axs[col, row])

                data_1 = extracted_features_dict[1][:, outc_weight_desc_indices[dim_index]]
                #
                print_sum_num_0(data_1, dim_index)
                draw_normal_distribution(data_1, label='label_1', plt=axs[col, row])
                axs[col, row].set_title(f'Dim {outc_weight_desc_indices[dim_index]}, weight: {outc_weight[outc_weight_desc_indices[dim_index]]:.4f}')
                pbar.update(1)
    # for ax in axs.flat:
    #     ax.label_outer()
    # Add a legend to the figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    # save the figure
    test_path_exist(result_path)
    plt.savefig(os.path.join(result_path, figname))
    # plt.savefig(os.path.join(result_path, figname.replace('.png', '.svg')))

    # close the figure
    plt.close(fig)

def draw_distribution_one_dim(extracted_features_dict, 
                              result_path,
                              figname='distribution_one_dim.png',
                              mean_axis = 1,
                              ):
    """
    mean_axis: [0, 1]. 0: mean of each sample, 1: mean of each dim
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))

    data_0 = np.mean(extracted_features_dict[0], axis=mean_axis)
    print(f"data_0.shape: {data_0.shape}")
    print_sum_num_0_index = "dim-mean" if mean_axis == 1 else "batch-mean"
    print_sum_num_0(data_0, print_sum_num_0_index)

    data_1 = np.mean(extracted_features_dict[1], axis=mean_axis)
    print(f"data_1.shape: {data_1.shape}")
    print_sum_num_0(data_1, print_sum_num_0_index)
    
    draw_normal_distribution(data_0, label='label_0', plt=ax)
    draw_normal_distribution(data_1, label='label_1', plt=ax)

    fig.legend(loc='upper right')
    # title
    ax_title = "Dim-mean" if mean_axis == 1 else "Batch-mean"
    ax.set_title(f'Distribution of {ax_title}')
    # save the figure
    test_path_exist(result_path)
    plt.savefig(os.path.join(result_path, figname))

    plt.close(fig)
    
def main(model_path_prefix,
         dataset_path_source_list,
         result_path,
         batch_size: int = 1,
         label_1_percent = 0.2,
         binary_class_index_list = [i for i in range(1, 9)],
         dataset_i_list = [0, 1],
         ):

    model_chdir = r'C:\Users\ZT\OneDrive\Studying\Yiling\00Research\202401_experiment\code'
    test_path_exist(result_path)

    dataset_name_source_list = ["dwq_sentinel2", "xj_sentinel2"]
    dataset_name_target_list = dataset_name_source_list[::-1]
    dataset_path_source_list = dataset_path_source_list
    dataset_path_target_list = dataset_path_source_list[::-1]
    for get_transfer in [False, True]:
        print(f"get_transfered_distribution: {get_transfer}")
        for dataset_i in dataset_i_list:
            dataset_name_source = dataset_name_source_list[dataset_i]
            dataset_name_target = dataset_name_target_list[dataset_i]
            # if dataset_i == 1:
            #     data_path_source, data_path_target = data_path_target, data_path_source
            data_path_source = dataset_path_source_list[dataset_i]
            data_path_target = dataset_path_target_list[dataset_i]
            for binary_class_index in binary_class_index_list:
                print(f"dataset_name_source: {dataset_name_source}, dataset_name_target: {dataset_name_target}, binary_class_index: {binary_class_index}")
                # binary_class_index = 1
                # find the model file
                model_path = model_path_prefix + rf"\train_{dataset_name_source}_cls_{binary_class_index}\unet_*_best_val.pth"
                model_files = glob.glob(model_path)
                if len(model_files) == 0:
                    print(f"model_files: {model_files} is empty!")
                    # continue
                print(f"model_files: {model_files}")

                label_index_list = [0, 1]
                extracted_features_dict, outc_weight, outc_weight_desc_indices = get_distribution_per_dataset(model_files[0], model_chdir,
                                            data_path_source, data_path_target,
                                            batch_size=batch_size,
                                            binary_class_index = binary_class_index,
                                            label_1_percent = label_1_percent,
                                            label_index_list = label_index_list,
                                            get_transfer = get_transfer,
                                            )
                dataset_name_transfer_to = dataset_name_target if get_transfer else dataset_name_source
                draw_distribution_all_dim(extracted_features_dict, 
                                        result_path=os.path.join(result_path, "fig"),
                                        figname=f'{dataset_name_source}-to-{dataset_name_transfer_to}_cls_{binary_class_index}_distribution_all_dim.png',
                                        outc_weight=outc_weight, outc_weight_desc_indices=outc_weight_desc_indices
                                        )
                draw_distribution_one_dim(extracted_features_dict,
                                            result_path=os.path.join(result_path, "fig"),
                                            figname=f'{dataset_name_source}-to-{dataset_name_transfer_to}_cls_{binary_class_index}_distribution_one_dim-mean.png',
                                            mean_axis = 1,
                                            )
                draw_distribution_one_dim(extracted_features_dict,
                                            result_path=os.path.join(result_path, "fig"),
                                            figname=f'{dataset_name_source}-to-{dataset_name_transfer_to}_cls_{binary_class_index}_distribution_one_batch-mean.png',
                                            mean_axis = 0,
                                            )
    
if __name__ == "__main__":
    model_path_prefix = r"E:\Yiling\at_SIAT_research\2_model_pth" # E:\Yiling\at_SIAT_research\z_result\20240516_1_train_binary_weight
    # binary_class_index_list=[i for i in range(1, 9)]
    binary_class_index_list=[1, 2, 3, 6, 7, 8]

    data_path_source = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val'
    data_path_target = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val'

    # E:\Yiling\at_SIAT_research\z_result\20240
    result_path = r"E:\Yiling\at_SIAT_research\z_result\20240801_draw_distribution_of_a_dataset\20240802_draw_distribution_of_a_dataset_1_dwqs2_xjs2_200imgs"
    
    save_log(result_path=result_path, log_name="draw_distribution_of_a_dataset_dwqs2_xjs2.log")

    main(model_path_prefix=model_path_prefix,
         dataset_path_source_list=[data_path_source, data_path_target],
         result_path=result_path,
         batch_size=4,
         label_1_percent=0.2,
         binary_class_index_list=binary_class_index_list,
         dataset_i_list=[0, 1] # "dwq_sentinel2", "xj_sentinel2"
         )

