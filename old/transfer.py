import argparse

def transfer():
    print()

    # load model

    # dataloader, source, target, train or val?

    # miou, OA, ious

    # log

def get_args():
    parser = argparse.ArgumentParser(description='Hyperparams for transferring.')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    
    #"1 Dataset Path"
    parser.add_argument('--source_data_path', type=str, help='Path to datasets, including images/train & val, annotations/train & val', 
                        default=r'E:\Yiling\at_SIAT_research\1_dataset\dataset\dwq_sentinel2\train_val')
    
    #"2 Result Path and Log Name"
    parser.add_argument('--log_name', type=str, help='log name', 
                        default='20240222_train.log')
    parser.add_argument('--result_path', type=str, help='Path to RESULT', 
                        default=r'E:\Yiling\at_SIAT_research\z_result\20240222_train')
    
    # image parameters
    parser.add_argument('--band_num', type=int, default=4, help='The number of band') 
    parser.add_argument('--n_classes', type=int, default=9, help='The number of classes')
    parser.add_argument('--ignore_index', type=int, default=-100, help='ignore index')
    parser.add_argument('--img_rows', type=int, default=256,
                        help='Height of the input image')
    parser.add_argument('--img_cols', type=int, default=256,
                        help='Width of the input image')
    
    # unnecessary
    parser.add_argument('--manual_seed', type=int, default=0, help='Manual seed')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
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
    # log 

    transfer()