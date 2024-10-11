import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    def __init__(self, transforms):
        
        self.transforms = transforms
        self.Totensor = Seq2Tensor()

    def __call__(self, Seq):
        totensor = False # variable to use Seq2Tensor lastly
        
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
            rev = torch.full( (1, Seq.right_end - Seq.left_start + 1), Seq.rev, dtype=torch.float32)
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

        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class LeftCrop(nn.Module):
    '''

    '''
    def __init__(self, left_indent):
        super().__init__()
        self.left_indent = left_indent
        
    def forward(self, Seq):

        crop_coordinate = torch.randint(size=(1,), low = 0, high = self.left_indent).item()
        
        Seq.seq = Seq.seq[crop_coordinate : ]
        
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class RightCrop(nn.Module):
    '''

    '''
    def __init__(self, right_indent):
        super().__init__()
        self.right_indent = right_indent
        
    def forward(self, Seq):

        crop_coordinate = torch.randint(size=(1,), low = 0, high = self.right_indent).item()
        
        Seq.seq = Seq.seq[ : - crop_coodinate]
        
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomCrop(nn.Module):
    '''
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
            
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'
        
class Reverse(nn.Module):
    '''

    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, Seq):
        Seq.reverse = True
        r = torch.rand((1,)).item()
        if  r > 0.5:
            Seq.seq = reverse_complement(Seq.seq)
            Seq.rev = 1.0
        return Seq
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
def reverse_complement(seq, mapping={"A": "T", "G":"C", "T":"A", "C": "G", 'N': 'N'}):
        s = "".join(mapping[s] for s in reversed(seq))
        return s
class AddReverseChannel(nn.Module):
    '''

    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, Seq):
        Seq.use_reverse_channel = True
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'
