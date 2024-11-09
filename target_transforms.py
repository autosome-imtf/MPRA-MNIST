import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, T, Optional

class Compose:
    """
    Composes several transforms together. This transform does not support torchscript.

    Parameters
    ----------
    transforms : List[Callable]
        List of transformations to apply sequentially.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
    
        for transformation in self.transforms:
            target = transformation(target)
    
        return target
            
    def __repr__(self):
        transformations = '\n    '.join(repr(t) for t in self.transforms)
        return f"{self.__class__.__name__}(\n    {transformations}\n)"

class Normalize(nn.Module):
    """
    A module for normalizing a target tensor with specified mean and standard deviation.

    This class applies normalization to input data using the formula:
    normalized_target = (target - mean) / std

    Attributes:
    ----------
    mean : float or torch.Tensor
        The mean value used for normalization.
    std : float or torch.Tensor
        The standard deviation used for normalization.
    """
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, target):
        target = (target - self.mean) / self.std
        return target
        
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"