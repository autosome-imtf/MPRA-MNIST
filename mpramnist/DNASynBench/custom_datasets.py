import torch
import pandas as pd
#from datasets import Dataset as HF_Dataset
from torch.utils.data import Dataset as Torch_Dataset
#from transformers import AutoTokenizer

def one_hot(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encode = torch.zeros((4, len(seq)))
    for i, j in enumerate(seq):
        if j in mapping:
            encode[mapping[j], i] = 1
    return encode

class Bench_Dataset(Torch_Dataset):
    '''
    General custom dataset for benchmarking
    '''
    def __init__(self, task):
        self.task = task
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.loc[idx, 'sequence']
        label = self.df.loc[idx, 'label']
        return seq, label

    def generation(self, args):
        df = self.task(**args)
        self.df = df.reset_index(drop=True)

    def min_len(self):
        return min([len(self.df.loc[x, 'sequence']) for x in range(len(self))])

    def max_len(self):
        return max([len(self.df.loc[x, 'sequence']) for x in range(len(self))])

class One_Hot_Dataset(Torch_Dataset):
    '''
    Custom dataset with ohe-hot encoding of sequences
    '''
    def __init__(self, task):
        self.task = task
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.loc[idx, 'sequence']
        label = self.df.loc[idx, 'label']
        return one_hot(seq), label

    def generation(self, args):
        df = self.task(**args)
        self.df = df.reset_index(drop=True)

    def min_len(self):
        return min([len(self.df.loc[x, 'sequence']) for x in range(len(self))])

    def max_len(self):
        return max([len(self.df.loc[x, 'sequence']) for x in range(len(self))])