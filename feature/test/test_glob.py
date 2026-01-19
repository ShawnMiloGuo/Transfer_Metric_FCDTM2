import glob

model_path_prefix = r"E:\Yiling\at_SIAT_research\z_result\20240516_1_train_binary_weight" #E:\Yiling\at_SIAT_research\z_result\20240516_1_train_binary_weight
binary_class_index_list=[i for i in range(1, 9)]
dataset_name_source = "dwq_sentinel2"

for binary_class_index in binary_class_index_list:
    model_path = model_path_prefix + rf"\train_{dataset_name_source}_cls_{binary_class_index}\unet_*_best_val.pth"
    # find the model file
    model_files = glob.glob(model_path)
    print(f"class: {binary_class_index}, model_files: {model_files}")
    if len(model_files) == 0:
        print(f"model_files: {model_files} is empty!")
        continue
    print(f"model_files: {model_files}")
