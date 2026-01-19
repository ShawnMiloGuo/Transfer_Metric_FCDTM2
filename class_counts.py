from component.utils import compute_class_weights, save_log
from component.dataset import get_dataset_reader
from torch.utils import data

def compute_class_counts(
        data_path,
        n_classes: int = 9,
        dataset_name: str = 'rgbn', 
        batch_size: int = 16,
        binary_class_index: int = -1,
        label_1_percent: float = 0.0,
        ):
    # dataset
    dataset_reader = get_dataset_reader(dataset_name)
    train_dateset = dataset_reader(root_dir=data_path, is_train=True, transform= None, binary_class_index = binary_class_index, label_1_percent = label_1_percent)
    train_loader = data.DataLoader(train_dateset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = dataset_reader(root_dir=data_path, is_train=False, transform= None, binary_class_index = binary_class_index, label_1_percent = label_1_percent)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'dataset path: {data_path}')
    print(f'n_classes: {n_classes}')

    # train set
    print_result(train_loader, n_classes, len(train_dateset), dataset_type='train')

    # val set
    print_result(val_loader, n_classes, len(val_dataset), dataset_type='val')

def print_result(data_loader, n_classes, len_dataset, dataset_type='train'):
    class_weights, class_counts = compute_class_weights(data_loader, n_classes, len_dataset)
    print(f'{dataset_type} set: ')
    print(f'len({dataset_type}_dataset): {len_dataset}')
    print(f'class_counts: {dict(class_counts)}')
    class_percentages = {k: v / sum(class_counts.values()) for k, v in class_counts.items()}
    print(f'class_percentage: {class_percentages}')

    print(f'class_counts_list: {list(class_counts.values())}')
    print(f'class_percentage_list: {list(class_percentages.values())}')
    print(f'class_weights_list: {class_weights.tolist()}')

if __name__ == '__main__':
    save_log(result_path=r'E:\Yiling\at_SIAT_research\z_result\20240419_class_counts', log_name='20240419_class_counts.log')
    data_path_dict = {
        'dwq_sentinel2': r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val',
        'dwq_landsat8': r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_landsat8\train_val',
        'xj_sentinel2': r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_sentinel2\train_val',
        'xj_landsat8': r'E:\Yiling\at_SIAT_research\1_dataset\dataset\xj_landsat8\train_val',
    }
    for data_name, data_path in data_path_dict.items():
        print(f'\ndata_name: {data_name}', '='*50)
        compute_class_counts(data_path, n_classes=9, dataset_name='rgbn', batch_size=16)
