from .dataset import rgbn_dataset
from .dataset_with_name import rgbn_dataset_with_name

def get_dataset_reader(name):
    return{
        'rgbn':rgbn_dataset,
        'rgbn_with_name':rgbn_dataset_with_name,
    }[name]
