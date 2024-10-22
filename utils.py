
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, T
    
def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True #type: ignore
    torch.backends.cudnn.benchmark = False #type: ignore

CODES = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

INV_CODES = {value: key for key, value in CODES.items()}

COMPL = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G',
    'N': 'N'
}

def n2id(n):
    return CODES[n.upper()]

def id2n(i):
    return INV_CODES[i]

def n2compl(n):
    return COMPL[n.upper()]

def parameter_count(model):
    pars = 0  
    for _, p  in model.named_parameters():    
        pars += torch.prod(torch.tensor(p.shape))
    return pars

def revcomp(seq):
    return "".join((n2compl(x) for x in reversed(seq)))

def get_rev(df):
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    revdf['rev'] = 1
    return revdf

def add_rev(df):
    df = df.copy()
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    df['rev'] = 0
    revdf['rev'] = 1
    df = pd.concat([df, revdf]).reset_index(drop=True)
    return df

class Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()
    def forward(self, seq):
        if isinstance(seq, torch.FloatTensor):
            return seq
        seq = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq))
        code = F.one_hot(code, num_classes=5) # 5th class is N
        
        code[code[:, 4] == 1] = 0.25 # encode Ns with .25
        code = code[:, :4].float() 
        return code.transpose(0, 1)

def reverse_complement(seq, mapping={"A": "T", "G":"C", "T":"A", "C": "G", 'N': 'N'}):
    s = "".join(mapping[s] for s in reversed(seq))
    return s

def encode_seq(seq: str):
    seq = [n2id(x) for x in seq] # type: ignore 
    code = torch.LongTensor(seq)
    code = F.one_hot(code, num_classes=5) # 5th class is N
    code = code[:, :5].float()
    code[code[:, 4] == 1] = 0.25 # encode Ns with .25
         
    return code[:, :4].transpose(0, 1)