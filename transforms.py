import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, T

CODES = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

def n2id(n):
    return CODES[n.upper()]

class Compose:
    """
    Composes several transforms together. This transform does not support torchscript.

    Parameters
    ----------
    Seq (sequence): SeqObj.
    """
    def __init__(self, transforms):
        
        self.transforms = transforms
        self.Totensor = Seq2Tensor()

    def __call__(self, Seq):
        totensor = False # variable to use Seq2Tensor last
        
        for transformation in self.transforms:
            
            if repr(transformation) == "Seq2Tensor()":
                totensor = True
                continue
                
            Seq = transformation(Seq)
            
        if totensor:
            Seq = self.Totensor(Seq)
            
        return Seq
        
    def __repr__(self):
        
        format_string = self.__class__.__name__ + '(' # print like: Compose(
        for t in self.transforms:                     # AddFlanks()
            format_string += '\n'                     # RandomCrop()
            format_string += '    {0}'.format(t)      # Reverse()
        format_string += '\n)'                        # )
        
        return format_string
        
class Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.

    Parameters
    ----------
    Seq (sequence): SeqObj.
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, Seq):

        if isinstance(Seq.seq, torch.FloatTensor):
            print("Sequence is tensor already")
            return Seq
        
        code = [n2id(x) for x in Seq.seq]
        code = torch.from_numpy(np.array(code))
        code = F.one_hot(code, num_classes=5) # 5th class is N
        
        code[code[:, 4] == 1] = 0.25 # encode Ns with .25
        code = code[:, :4].float()
        X = code.transpose(0, 1)
        to_concat = [X]
        
        if Seq.use_reverse_channel: # adding reverse_channel
            rev = torch.full( (1, Seq.seqsize), Seq.rev, dtype=torch.float32)
            to_concat.append(rev)
            
        # add channels scalar and vector
        if Seq.add_feature_channel:
            for ch in Seq.feature_channels:
                if ch in Seq.scalars.keys():
                    rev = torch.full( (1, Seq.seqsize),  Seq.scalars[ch].val, dtype=torch.float32)
                    to_concat.append(rev)
                if ch in Seq.vectors.keys():
                    rev = torch.tensor([Seq.vectors[ch].val])
                    to_concat.append(rev)
            
         # create final tensor
        if len(to_concat) > 1:
            Seq.seq = torch.concat(to_concat, dim=0)
        else:
            Seq.seq = X
            
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class AddFlanks(nn.Module):
    '''
    Add forward and back flanks to 5' -> 3'

    Parameters
    ----------
    Seq (sequence): SeqObj.
    left_flank (str)
    right_flank (str)
    '''
    def __init__(self, left_flank = "", right_flank = ""):
        
        super().__init__()
        self.left_side = left_flank.upper()
        self.right_side = right_flank.upper()
        
    def forward(self, Seq):

        assert set(list(self.left_side)).issubset(set(CODES)), "left flank is not DNA seq"
        assert set(list(self.right_side)).issubset(set(CODES)), "right flank is not DNA seq"

        Seq.is_flank_added = True
        Seq.seq = self.left_side + Seq.seq + self.right_side

        # change vector feature
        for name in Seq.vectors:
            left_flank_vect = [Seq.vectors[name].tp.levels[Seq.vectors[name].pad_value] for i in range(len(self.left_side))]
            right_flank_vect = [Seq.vectors[name].tp.levels[Seq.vectors[name].pad_value] for i in range(len(self.right_side))]
            Seq.vectors[name].val = left_flank_vect + Seq.vectors[name].val + right_flank_vect

        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class LeftCrop(nn.Module):
    '''
    Operation of cutting a slice of a sequence with a random length from min_crop to max_crop. 
    Right end of sequence always constant and the crop operation results in the cutting off of the left side of sequence.
    i.e len(ABCDEFGHKLMNOPQRSTUV) = 20 -----> 
    RightCrop(SeqObj("ABCDEFGHKLMNOPQRSTUV"), min_crop = 15, max_crop = 18) can result:
    --> FGHKLMNOPQRSTUV
    --> EFGHKLMNOPQRSTUV
    --> DEFGHKLMNOPQRSTUV
    --> CDEFGHKLMNOPQRSTUV
    min_crop can be equal to max_crop, then length of output seq wil be min_crop = max_crop

    Parameters
    ----------
    Seq (sequence): SeqObj.
    min_crop, max_crop: int - min max sizes of cropped sequence from left side
    '''
    def __init__(self, min_crop, max_crop):
        super().__init__()
        self.min = min_crop
        self.max = max_crop
        
    def forward(self, Seq):

        crop_coordinate = torch.randint(size=(1,), low = self.min, high = self.max + 1).item()
        
        Seq.seq = Seq.seq[ - crop_coordinate : ]

        #change vector feature
        for name in Seq.vectors:
            Seq.vectors[name].val = Seq.vectors[name].val[ - crop_coordinate : ]
        
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class RightCrop(nn.Module):
    '''
    Operation of cutting a slice of a sequence with a random length from min_crop to max_crop. 
    Left end of sequence always constant and the crop operation results in the cutting off of the right side of sequence.
    i.e len(ABCDEFGHKLMNOPQRSTUV) = 20 -----> 
    RightCrop(SeqObj("ABCDEFGHKLMNOPQRSTUV"), min_crop = 15, max_crop = 18) can result:
    --> ABCDEFGHKLMNOPQ
    --> ABCDEFGHKLMNOPQR
    --> ABCDEFGHKLMNOPQRS
    --> ABCDEFGHKLMNOPQRST
    min_crop can be equal to max_crop, then length of output seq wil be min_crop = max_crop

    Parameters
    ----------
    Seq (sequence): SeqObj.
    min_crop, max_crop  (int): - min, max sizes of cropped sequence from right side
    '''
    def __init__(self, min_crop, max_crop):
        super().__init__()
        self.min = min_crop
        self.max = max_crop
        
    def forward(self, Seq):

        crop_coordinate = torch.randint(size=(1,), low = self.min, high = self.max + 1).item()
        
        Seq.seq = Seq.seq[ : crop_coordinate]

        # change vvector feature
        for name in Seq.vectors:
            Seq.vectors[name].val = Seq.vectors[name].val[ : crop_coordinate]
        
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomCrop(nn.Module):
    '''
    Operation of cutting a random slice of a sequence with a given length

    Parameters
    ----------
    Seq (sequence): SeqObj.
    output_size (int): Expected lehgth of the crop.
    '''
    def __init__(self, 
                 output_size, 
                ):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, Seq):
        
        crop_coordinate = torch.randint(size=(1,), low = 0, high = Seq.seqsize - self.output_size + 1).item()
            
        Seq.seq = Seq.seq[ crop_coordinate : crop_coordinate + self.output_size ] 

        #change vector feature
        for name in Seq.vectors:
            Seq.vectors[name].val = Seq.vectors[name].val[ crop_coordinate : crop_coordinate + self.output_size]
            
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class Reverse(nn.Module):
    '''
    Reverse complement transformation to sequence
    i.e AATCGG -> CCGATT
    
    Parameters
    ----------
    Seq (sequence): SeqObj.
    prob (float from 0 to 1): reverse complement tranformation probability 
    '''
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        
    def forward(self, Seq):
        
        rand = torch.rand((1,)).item()
        if  rand < self.prob or self.prob == 1:
            Seq.reverse = True
            Seq.seq = reverse_complement(Seq.seq)
            Seq.rev = 1.0
        return Seq
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

def reverse_complement(seq, mapping={"A": "T", "G":"C", "T":"A", "C": "G", 'N': 'N'}):
        s = "".join(mapping[s] for s in reversed(seq.upper()))
        return s

class AddFeatureChannels(nn.Module):
    '''
    Operation of adding additional channel. Storing info about features of sequence
    
    Parameters
    ----------
    Seq (sequence): SeqObj.
    channels (list[str]): dataframe's feture channels
    '''
    def __init__(self, 
                 channels # = array of str[]
                ):
        super().__init__()
        self.channels = channels
        
    def forward(self, Seq):
        Seq.add_feature_channel = True
        
        #user must know scalar and vector features of dataframe
        for ch in self.channels:
            Seq.feature_channels.append(ch)
            
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class AddReverseChannel(nn.Module):
    '''
    Operation of adding additional channel. Storing info about is sequence reversed or not
    
    Parameters
    ----------
    Seq (sequence): SeqObj.
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, Seq):
        Seq.use_reverse_channel = True
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'
