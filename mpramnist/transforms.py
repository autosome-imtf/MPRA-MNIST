import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, T, Optional
from .dataclass import SeqObj

CODES = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

def n2id(n):
    return CODES[n.upper()]
    
class Seq2Tensor(nn.Module):
    '''
    Encodes sequences into tensors using one-hot encoding and additional features.

    Methods
    -------
    forward(Seq):
        Applies one-hot encoding and adds optional feature channels to the sequence.

    Notes
    -----
    - If the input is already a tensor, it is returned unchanged.
    - Supports reverse channel and additional feature channels.
    - This method modifies the input object in-place.
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, Seq: SeqObj) -> SeqObj:

        if isinstance(Seq.seq, torch.FloatTensor):
            return Seq # Sequence is already a tensor, no further processing required.
        
        code = [n2id(x) for x in Seq.seq]
        code = torch.from_numpy(np.array(code))
        code = F.one_hot(code, num_classes=5).float() # 5th class is N
        

        # Encode 'N' class with 0.25
        code[code[:, 4] == 1] = 0.25 
        code = code[:, :4]
        X = code.transpose(0, 1)
         
        to_concat = [X]
        
        # Add reverse channel if enabled
        if getattr(Seq, 'use_reverse_channel', False):
            rev = torch.full( (1, Seq.seqsize), Seq.rev, dtype=torch.float32)
            to_concat.append(rev)
            
        # Add additional feature channels
        if getattr(Seq, 'add_feature_channel', False):
            for ch in Seq.feature_channels:
                if ch in Seq.scalars.keys():
                    rev = torch.full( (1, Seq.seqsize),  Seq.scalars[ch].val, dtype=torch.float32)
                    to_concat.append(rev)
                if ch in Seq.vectors.keys():
                    rev = torch.tensor([Seq.vectors[ch].val])
                    to_concat.append(rev)
            
        # Concatenate all channels
        Seq.seq = torch.cat(to_concat, dim=0) if len(to_concat) > 1 else X
        Seq.one_hot_encoded = True
        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class Compose:
    """
    Composes several transforms together. This transform does not support torchscript.

    Parameters
    ----------
    transforms : List[Callable]
        List of transformations to apply sequentially.
    totensor_cls : Callable, optional
        A callable that transforms sequence to tensor, by default Seq2Tensor.
    """
    def __init__(self, transforms, totensor_cls = Seq2Tensor):
        self.transforms = transforms
        self.Totensor = totensor_cls()

    def __call__(self, Seq):
        transforms = [t for t in self.transforms if not isinstance(t, Seq2Tensor)]
        totensor = any(isinstance(t, Seq2Tensor) for t in self.transforms)
    
        for transformation in transforms:
            Seq = transformation(Seq)
            
        if totensor:
            Seq = self.Totensor(Seq)
    
        return Seq
            
    def __repr__(self):
        transformations = '\n    '.join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}(\n    {transformations}\n)"
        
class AddFlanks(nn.Module):
    """
    Adds left and right flanks to the DNA sequence in the 5' -> 3' direction.

    Parameters
    ----------
    left_flank : str, optional
        The sequence to add to the left (5') end. Default is an empty string.
    right_flank : str, optional
        The sequence to add to the right (3') end. Default is an empty string.

    Methods
    -------
    forward(Seq):
        Adds the specified flanks to the sequence and updates any vector features.
        Note: This method modifies the input object in-place.
    """
    def __init__(self, left_flank: Optional[str] = "", right_flank: Optional[str] = ""):
        
        super().__init__()
        self.left_side = left_flank.upper()
        self.right_side = right_flank.upper()
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        # Validate that flanks contain only valid DNA characters
        if not set(self.left_side).issubset(CODES):
            raise ValueError(f"Left flank contains invalid DNA characters: {self.left_side}")
        if not set(self.right_side).issubset(CODES):
            raise ValueError(f"Right flank contains invalid DNA characters: {self.right_side}")

        # Add flanks to the sequence
        Seq.is_flank_added = True
        Seq.seq = self.left_side + Seq.seq + self.right_side

         # Update vector features 
        for name in Seq.vectors:
            left_flank_vect = [Seq.vectors[name].tp.levels[Seq.vectors[name].pad_value] for i in range(len(self.left_side))]
            right_flank_vect = [Seq.vectors[name].tp.levels[Seq.vectors[name].pad_value] for i in range(len(self.right_side))]
            Seq.vectors[name].val = left_flank_vect + Seq.vectors[name].val + right_flank_vect

        return Seq
        
    def __repr__(self):
        return self.__class__.__name__ + '()'

class LeftCrop(nn.Module):
    '''
    Randomly crops a sequence from the left side, keeping the right end constant.

    The length of the cropped sequence will be between `min_crop` and `max_crop` (both inclusive).
    
    For example, given the sequence 'ABCDEFGHKLMNOPQRSTUV' with length 20:
        LeftCrop(SeqObj("ABCDEFGHKLMNOPQRSTUV"), min_crop = 15, max_crop = 18) can result:
        -->    FGHKLMNOPQRSTUV (crop length 15)
        -->   EFGHKLMNOPQRSTUV (crop length 16)
        -->  DEFGHKLMNOPQRSTUV (crop length 17)
        --> CDEFGHKLMNOPQRSTUV (crop length 18)
        
    If `min_crop == max_crop`, the output will have a fixed length of `min_crop`.

    Parameters
    ----------
    min_crop : int
        Minimum length of the cropped sequence.
    max_crop : int
        Maximum length of the cropped sequence.

    Methods
    -------
    forward(Seq):
        Modifies the input sequence object by cropping its left side.
        
    Raises
    ------
    ValueError
        If min_crop or max_crop are negative, or if min_crop > max_crop.
    '''
    def __init__(self, min_crop: int, max_crop: int):
        super().__init__()
        if min_crop <= 0 or max_crop <= 0:
            raise ValueError("min_crop and max_crop must be non-negative integers.")
        if min_crop > max_crop:
            raise ValueError("min_crop cannot be greater than max_crop.")
            
        self.min = min_crop
        self.max = max_crop
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        """Modifies the input sequence by cropping from the left side."""
        if len(Seq.seq) < self.min:
            raise ValueError(f"Sequence length ({len(Seq.seq)}) is shorter than min_crop ({self.min}).")
            
        crop_coordinate = torch.randint(size=(1,), low = self.min, high = self.max + 1).item()
        
        # Crop the sequence
        Seq.seq = Seq.seq[ - crop_coordinate : ]

        #change vector feature
        for name in Seq.vectors:
            Seq.vectors[name].val = Seq.vectors[name].val[ - crop_coordinate : ]
        
        return Seq
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_crop={self.min}, max_crop={self.max})"

class RightCrop(nn.Module):
    '''
    Randomly crops a sequence from the right side while keeping the left end intact.
    
    The length of the cropped sequence will be between `min_crop` and `max_crop` (both inclusive).
    For example, given the sequence 'ABCDEFGHKLMNOPQRSTUV' with length 20:
        RightCrop(SeqObj("ABCDEFGHKLMNOPQRSTUV"), min_crop = 15, max_crop = 18) can produce:
        --> ABCDEFGHKLMNOPQ (crop length 15)
        --> ABCDEFGHKLMNOPQR (crop length 16)
        --> ABCDEFGHKLMNOPQRS (crop length 17)
        --> ABCDEFGHKLMNOPQRST (crop length 18)
    
    If `min_crop == max_crop`, the output will have a fixed length of `min_crop`.

    Parameters
    ----------
    min_crop : int
        Minimum length of the cropped sequence.
    max_crop : int
        Maximum length of the cropped sequence.

    Methods
    -------
    forward(Seq):
        Modifies the input sequence object by cropping its right side.
         
    Raises
    ------
    ValueError
        If min_crop or max_crop are negative, or if min_crop > max_crop.
    '''
    def __init__(self, min_crop: int, max_crop: int):
        super().__init__()
        if min_crop <= 0 or max_crop <= 0:
            raise ValueError("min_crop and max_crop must be positive integers.")
        if min_crop > max_crop:
            raise ValueError("min_crop cannot be greater than max_crop.")
    
        self.min = min_crop
        self.max = max_crop
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        """Modifies the input sequence by cropping from the right side."""
        if len(Seq.seq) < self.min:
            raise ValueError(f"Sequence length {len(Seq.seq)} is smaller than the minimum crop size {self.min}.")
        
        crop_coordinate = torch.randint(size=(1,), low = self.min, high = self.max + 1).item()
        
        # Crop the sequence
        Seq.seq = Seq.seq[ : crop_coordinate]

        # change vector feature
        for name in Seq.vectors:
            Seq.vectors[name].val = Seq.vectors[name].val[ : crop_coordinate]
        
        return Seq
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_crop={self.min}, max_crop={self.max})"

class CenterCrop(nn.Module):
    """
    CenterCrop crops a sequence to a specified length by keeping the central portion.
    
    Parameters
    ----------
    output_size : int
        Desired length of the cropped sequence. Must be a positive integer.
        
    Methods
    -------
    forward(Seq: SeqObj) -> SeqObj:
        Applies a center crop to the sequence and its vector features, reducing its length to `output_size`.
    """
    def __init__(self, 
                 output_size: int, 
                ):
        super().__init__()
        if output_size <= 0:
            raise ValueError("Output size must be a positive integer.")
        self.output_size = output_size
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        if Seq.seqsize < self.output_size:
            raise ValueError(
                f"Sequence size ({Seq.seqsize}) must be greater than or equal to output_size ({self.output_size})."
            )
        crop_len = Seq.seqsize - self.output_size
        
        if crop_len > 1:
            Seq.seq = Seq.seq[ crop_len // 2 + crop_len % 2 : ]
            Seq.seq = Seq.seq[ : - crop_len // 2 + crop_len % 2]
        elif crop_len == 1:
            Seq.seq = Seq.seq[ 1 : ]

        #change vector feature
        for name in Seq.vectors:
            if crop_len > 1:
                Seq.vectors[name].val = Seq.vectors[name].val[ crop_len // 2 + crop_len % 2 : ]
                Seq.vectors[name].val = Seq.vectors[name].val[ : - crop_len // 2 + crop_len % 2]
            elif crop_len == 1:
                Seq.vectors[name].val = Seq.vectors[name].val[ 1 : ]

        return Seq
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_size={self.output_size})"
        
class RandomCrop(nn.Module):
    '''
    Randomly crops a sequence to a specified length.

    Parameters
    ----------
    output_size : int
        The expected length of the cropped sequence. Must be <= length of the input sequence.

    Methods
    -------
    forward(Seq):
        Modifies the input sequence object by cropping a random slice of length `output_size`.
    '''
    def __init__(self, 
                 output_size: int, 
                ):
        super().__init__()
        if output_size <= 0:
            raise ValueError("Output size must be a positive integer.")
        self.output_size = output_size
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        if Seq.seqsize < self.output_size:
            raise ValueError(
                f"Sequence size ({Seq.seqsize}) must be greater than or equal to output_size ({self.output_size})."
            )
        crop_coordinate = torch.randint(size=(1,), low = 0, high = Seq.seqsize - self.output_size + 1).item()
            
        Seq.seq = Seq.seq[ crop_coordinate : crop_coordinate + self.output_size ] 

        #change vector feature
        for name in Seq.vectors:
            Seq.vectors[name].val = Seq.vectors[name].val[ crop_coordinate : crop_coordinate + self.output_size]
            
        return Seq
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_size={self.output_size})"

class Padding(nn.Module):
    """
    Pads a sequence to a specified length or with specified amounts on each side.

    Parameters
    ----------
    output_size : int or tuple[int, int]
        Desired output size for padding. If `int`, the sequence will be padded to this length.
        If `tuple`, it specifies padding amounts for the left and right sides respectively.
        
    Methods
    -------
    forward(Seq: SeqObj) -> SeqObj:
        Pads the sequence and its vector features according to `output_size`.
    """
    def __init__(self, 
                 output_size: int | tuple[int, int], 
                ):
        super().__init__()
        if isinstance(output_size, int) and output_size <= 0:
            raise ValueError("Output size must be a positive integer.")
        elif isinstance(output_size, tuple) and (len(output_size) != 2 or not all(isinstance(i, int) and i >= 0 for i in output_size)):
            raise ValueError("Output size tuple must contain two non-negative integers.")
        
        self.output_size = output_size
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        if isinstance(self.output_size, int) and self.output_size - Seq.seqsize > 0:
            pad_len = self.output_size - Seq.seqsize
            Seq.seq = "N"*(pad_len//2) + Seq.seq
            Seq.seq = Seq.seq + "N"*(pad_len//2) + "N"*(pad_len % 2)
            #change vector feature
            for name in Seq.vectors:
                Seq.vectors[name].val = Seq.vectors[name].pad_value*(pad_value//2) + Seq.vectors[name].val
                Seq.vectors[name].val = Seq.vectors[name].val + Seq.vectors[name].pad_value*(pad_value//2) + Seq.vectors[name].pad_value*(pad_value % 2)
        #elif isinstance(self.output_size, int) and self.output_size - Seq.seqsize < 0:
        #   self.output_size = (self.output_size, self.output_size)
        if isinstance(self.output_size, tuple):
            left_pad, right_pad = self.output_size
            Seq.seq = "N"*left_pad + Seq.seq + "N"*right_pad
            #change vector feature
            for name in Seq.vectors:
                Seq.vectors[name].val = Seq.vectors[name].pad_value*left_pad + Seq.vectors[name].val + Seq.vectors[name].pad_value*right_pad
            
        return Seq
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_size={self.output_size})"
        
class ReverseComplement(nn.Module):
    """
    Applies reverse complement transformation to a sequence with a given probability.

    Parameters
    ----------
    prob : float
        Probability of applying the reverse complement transformation (between 0 and 1).
        if prob = 1, sequence will be reverced always
        if prob = 0, sequence will not be reversed
    Methods
    -------
    forward(Seq: SeqObj) -> SeqObj:
        Modifies the input sequence if the transformation is applied.

    Example
    -------
    AATCGG -> CCGATT  # if the transformation is applied.
    """
    def __init__(self, prob):
        super().__init__()
        if not (0 <= prob <= 1):
            raise ValueError("Probability must be between 0 and 1.")
        self.prob = prob
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        
        if torch.rand((1,)).item() < self.prob:
            Seq.reverse = True
            Seq.seq = reverse_complement(Seq.seq)
            Seq.rev = 1.0
        return Seq
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(prob={self.prob})"

def reverse_complement(seq: str, mapping=None) -> str:
    if mapping is None:
        mapping = {"A": "T", "G": "C", "T": "A", "C": "G", "N": "N"}
    try:
        return "".join(mapping[base] for base in reversed(seq.upper()))
    except KeyError as e:
        raise ValueError(f"Invalid character in sequence: {e}")

class AddFeatureChannels(nn.Module):
    """
    Adds additional feature channels to the sequence.

    Parameters
    ----------
    channels : list[str]
        List of feature channel names to be added to the sequence.

    Methods
    -------
    forward(Seq: SeqObj) -> SeqObj:
        Adds the specified feature channels to the sequence object.
    """
    
    def __init__(self, channels: list[str]):
        super().__init__()
        if not all(isinstance(ch, str) for ch in channels):
            raise ValueError("All channels must be strings.")
        self.channels = channels
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        Seq.add_feature_channel = True
        
        #user must know scalar and vector features of dataframe
        for ch in self.channels:
            Seq.feature_channels.append(ch)
            
        return Seq
        
    def __repr__(self):
        return f"{self.__class__.__name__}(channels={self.channels})"

class AddReverseChannel(nn.Module):
    """
    Adds a reverse channel to the sequence to store whether the sequence was reversed.

    Methods
    -------
    forward(Seq: SeqObj) -> SeqObj:
        Marks the sequence as using a reverse channel.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, Seq: SeqObj) -> SeqObj:
        Seq.use_reverse_channel = True
        return Seq
        
    def __repr__(self):
        return f"{self.__class__.__name__}()"
