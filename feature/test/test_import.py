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

print("test_import.py is imported")