import os
import sys
# Get the parent directory of the current file, add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
from tqdm import tqdm
import glob
import numpy as np

from component.utils import test_path_exist
from component.utils import save_log


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

    # layer_name 
    # 'up4', -> [1, 64, 256, 256]         
    # 'outc', -> [1, 9, 256, 256]
    layer_name = 'up4'

    # get outc weight rank
    # model.outc.weight.detach().cpu().squeeze().shape --> torch.Size([2, 64])
    # model.outc.weight.shape --> torch.Size([2, 64, 1, 1])
    last_layer_weight = model.outc.weight.permute(2,3,0,1).reshape(-1,64).cpu().tolist()
    last_layer_bias = model.outc.bias.cpu().tolist()
    print(f"last_layer_weight: {last_layer_weight}")
    print(f"last_layer_bias: {last_layer_bias}")

    outc_weight = model.outc.weight.permute(2,3,0,1).reshape(-1,64).cpu()[1].tolist()
    outc_weight_desc_indices, outc_weight_ranks = get_rank_index(outc_weight)
    print(f"outc_weight: {outc_weight}")
    print(f"outc_weight_desc_indices: {outc_weight_desc_indices}")
    print(f"outc_weight_ranks: {outc_weight_ranks}")

    return outc_weight, outc_weight_desc_indices
    
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
    for dataset_i in dataset_i_list:
        dataset_name_source = dataset_name_source_list[dataset_i]
        dataset_name_target = dataset_name_target_list[dataset_i]
        # if dataset_i == 1:
        #     data_path_source, data_path_target = data_path_target, data_path_source
        data_path_source = dataset_path_source_list[dataset_i]
        data_path_target = dataset_path_target_list[dataset_i]
        for binary_class_index in tqdm(binary_class_index_list):
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
            outc_weight, outc_weight_desc_indices = get_distribution_per_dataset(model_files[0], model_chdir,
                                        data_path_source, data_path_target,
                                        batch_size=batch_size,
                                        binary_class_index = binary_class_index,
                                        label_1_percent = label_1_percent,
                                        label_index_list = label_index_list,
                                        )
            
    
if __name__ == "__main__":
    model_path_prefix = r"E:\Yiling\at_SIAT_research\2_model_pth" # E:\Yiling\at_SIAT_research\z_result\20240516_1_train_binary_weight
    # binary_class_index_list=[i for i in range(1, 9)]
    binary_class_index_list=[1, 2, 3, 6, 7, 8]

    data_path_source = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val'
    data_path_target = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val'

    # E:\Yiling\at_SIAT_research\z_result\20240
    result_path = r"E:\Yiling\at_SIAT_research\z_result\20240819_print_model_weight\20240823_1032_debug_weight_1_dwqs2_xjs2"
    
    save_log(result_path=result_path, log_name="print_model_weight_dwqs2_xjs2.log")

    main(model_path_prefix=model_path_prefix,
         dataset_path_source_list=[data_path_source, data_path_target],
         result_path=result_path,
         batch_size=4,
         label_1_percent=0.2,
         binary_class_index_list=binary_class_index_list,
         dataset_i_list=[0, 1] # "dwq_sentinel2", "xj_sentinel2"
         )

