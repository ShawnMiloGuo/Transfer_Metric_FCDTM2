import os
import sys
# Get the parent directory of the current file, add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import numpy as np

from component.utils import test_path_exist
from component.utils import save_log

from typing import List


# get the rank of the weights of the outc layer
def get_rank_index(lst):
    desc_indices = [i[0] for i in sorted(enumerate(lst), key=lambda x:x[1], reverse=True)]
    ranks = [desc_indices.index(i) for i in range(len(lst))]
    return desc_indices, ranks

def get_weight_per_model(model_path, model_chdir,):
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

    outc_weight_1 = model.outc.weight.permute(2,3,0,1).reshape(-1,64).cpu()[1].tolist()
    outc_weight_1_desc_indices, outc_weight_1_ranks = get_rank_index(outc_weight_1)
    print(f"outc_weight_1: {outc_weight_1}")
    print(f"outc_weight_1_desc_indices: {outc_weight_1_desc_indices}")
    print(f"outc_weight_1_ranks: {outc_weight_1_ranks}")

    return last_layer_weight, last_layer_bias, outc_weight_1, outc_weight_1_desc_indices

def draw_line_one_dim(x: List[float], y1: List[float], y2: List[float], label: List[str], plt=plt):
    # draw the reference line y=0, x=0
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=1.0)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=1.0)
    
    # draw the line
    plt.plot(x, y1, label=label[0], color='blue', marker='')
    plt.plot(x, y2, label=label[1], color='red', marker='')


def draw_line_all_dim(last_layer_weight_list, last_layer_bias_list,
                              result_path,
                              figname='distribution_all_dim.png',
                              outc_weight=[], 
                              outc_weight_desc_indices=[]
                              ):
    # Create a figure and a set of subplots
    size_per_subplot = 4
    col_subplots = 8
    row_subplots = 8
    fig, axs = plt.subplots(col_subplots, row_subplots, 
                            figsize=(col_subplots * size_per_subplot, row_subplots * size_per_subplot), 
                            sharex=True, sharey=True)
    fig.suptitle(figname.replace('.png', ''), fontsize=16)
    with tqdm(total=col_subplots * row_subplots, desc="drawing") as pbar:                
        for col in range(col_subplots):
            for row in range(row_subplots):
                dim_index = col * row_subplots + row
                x = [0, 10]
                y1 = [last_layer_weight_list[0][outc_weight_desc_indices[dim_index]] * x[i] + last_layer_bias_list[0] for i in range(2)]
                y2 = [last_layer_weight_list[1][outc_weight_desc_indices[dim_index]] * x[i] + last_layer_bias_list[1] for i in range(2)]
                draw_line_one_dim(x, y1, y2, label=['label_0', 'label_1'], plt=axs[col, row])
                axs[col, row].set_title(f'Dim {outc_weight_desc_indices[dim_index]}, weight_1: {outc_weight[outc_weight_desc_indices[dim_index]]:.4f}')
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

def main(model_path_prefix,
         result_path,
         binary_class_index_list = [i for i in range(1, 9)],
         dataset_i_list = [0, 1],
         ):

    model_chdir = r'C:\Users\ZT\OneDrive\Studying\Yiling\00Research\202401_experiment\code'
    test_path_exist(result_path)

    dataset_name_source_list = ["dwq_sentinel2", "xj_sentinel2"]
    for dataset_i in dataset_i_list:
        dataset_name_source = dataset_name_source_list[dataset_i]
        
        for binary_class_index in tqdm(binary_class_index_list):
            print(f"model: train_{dataset_name_source}_cls_{binary_class_index}")
            # binary_class_index = 1
            # find the model file
            model_path = model_path_prefix + rf"\train_{dataset_name_source}_cls_{binary_class_index}\unet_*_best_val.pth"
            model_files = glob.glob(model_path)
            if len(model_files) == 0:
                print(f"model_files: {model_files} is empty!")
                # continue
            print(f"model_files: {model_files}")

            label_index_list = [0, 1]
            last_layer_weight, last_layer_bias, outc_weight_1, outc_weight_1_desc_indices = get_weight_per_model(model_files[0], model_chdir,)
            # draw_line_all_dim
            draw_line_all_dim(last_layer_weight_list=last_layer_weight, last_layer_bias_list=last_layer_bias,
                                result_path=result_path,
                                figname=f'{dataset_name_source}_cls_{binary_class_index}.png',
                                outc_weight=outc_weight_1, 
                                outc_weight_desc_indices=outc_weight_1_desc_indices
                                )
            
    
if __name__ == "__main__":
    model_path_prefix = r"E:\Yiling\at_SIAT_research\2_model_pth" # E:\Yiling\at_SIAT_research\z_result\20240516_1_train_binary_weight
    # binary_class_index_list=[i for i in range(1, 9)]
    binary_class_index_list=[1, 2, 3, 6, 7, 8]

    data_path_source = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val'
    data_path_target = r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val'

    # E:\Yiling\at_SIAT_research\z_result\20240
    result_path = r"E:\Yiling\at_SIAT_research\z_result\20240904_draw_line_of_weight_bias\20240906_draw_line_axhline_axvline_alpha10"
    
    save_log(result_path=result_path, log_name="draw_line_of_weight_bias_dwqs2_xjs2.log")

    main(model_path_prefix=model_path_prefix,
         result_path=result_path,
         binary_class_index_list=binary_class_index_list,
         dataset_i_list=[0, 1] # "dwq_sentinel2", "xj_sentinel2"
         )

