import pdb
import copy

import torch
from torch import nn
from transformers import BertModel, BertTokenizer
CUDA_ENABLED  = 0
from torch.utils.data import Dataset
from itertools import islice
from torch.utils.data import DataLoader
import numpy as np

def to_cuda(tensor):
    if CUDA_ENABLED:
        tensor = tensor.cuda()
    return tensor
def build_sentence_list(start_token, sentences):
    text = [start_token]
    for sentence in sentences:
        text += sentence + ['SEP']
    return text