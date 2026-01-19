# compute and print the class counts for a binary classification,
# with warnings for empty training or validation datasets.

from component.utils import compute_class_weights, save_log
from component.dataset import get_dataset_reader
from torch.utils import data
from tqdm import tqdm

def compute_class_counts(
        data_path,
        n_classes: int = 2,
        dataset_name: str = 'rgbn', 
        batch_size: int = 16,
        binary_class_index: int = 1,
        label_1_percent: float = 0.3
        ):
    # dataset
    dataset_reader = get_dataset_reader(dataset_name)
    train_dataset = dataset_reader(root_dir=data_path, is_train=True, transform= None, binary_class_index = binary_class_index, label_1_percent = label_1_percent)
    val_dataset = dataset_reader(root_dir=data_path, is_train=False, transform= None, binary_class_index = binary_class_index, label_1_percent = label_1_percent)

    # check if the datasets are empty
    if len(train_dataset) == 0:
        print(f"Warning: No data found in {data_path} for training")
        return
    if len(val_dataset) == 0:
        print(f"Warning: No data found in {data_path} for validation")
        return

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'dataset path: {data_path}')
    print(f'n_classes: {n_classes}')

    # train set
    # print_result(train_loader, n_classes, len(train_dataset), dataset_type='train')
    # 遍历train_loader
    # for i, (image, annotation) in enumerate(tqdm(train_loader, desc="Processing train_loader")):
    #     pass


    # val set
    # print_result(val_loader, n_classes, len(val_dataset), dataset_type='val')
    # 遍历val_loader
    # for i, (image, annotation) in enumerate(tqdm(val_loader, desc="Processing val_loader")):
    #     pass

def print_result(data_loader, n_classes, len_dataset, dataset_type='train'):
    class_weights, class_counts = compute_class_weights(data_loader, n_classes, len_dataset)
    print(f'{dataset_type} set: ')
    class_percentages = {k: v / sum(class_counts.values()) for k, v in class_counts.items()}

    print(f'class_counts_list: {list(class_counts.values())}')
    print(f'class_percentage_list: {list(class_percentages.values())}')

if __name__ == '__main__':
    # save_log(result_path=r'E:\Yiling\at_SIAT_research\z_result\20240620_class_counts_binary_landsat', log_name='20240620_class_counts_binary_0.0-0.3.log')
    save_log(result_path=r'E:\Yiling\at_SIAT_research\z_result\20250326_class_counts_binary_all_dataset', log_name='20250326_1718_class_counts_binary_all_dataset.log')
    data_path_dict = {
        'dwq_sentinel2': r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val',
        'xj_sentinel2': r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val',
        'dwq_landsat8': r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_landsat8\train_val',
        'xj_landsat8': r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_landsat8\train_val',
    }
    # for label_1_percent in [0.3, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0]:
    for label_1_percent in [0.2]:
        print("\n\n")
        print('='*100)
        print('='*100)
        print(f'label_1_percent: {label_1_percent}')
        print('_'*100)
        for data_name, data_path in data_path_dict.items():
            for binary_class_index in range(1, 9):
                print(f'\ndata_name: {data_name}_cls_{binary_class_index}_label-1-percent_{label_1_percent}', '='*100)
                compute_class_counts(data_path, n_classes=2, dataset_name='rgbn', batch_size=1, binary_class_index=binary_class_index, label_1_percent=label_1_percent)
