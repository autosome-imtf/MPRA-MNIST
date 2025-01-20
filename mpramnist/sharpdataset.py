import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO
import os

from .mpradataset import MpraDataset

ACTIVITY_COLUMNS = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
       'y11', 'y12']

class SharpDataset(MpraDataset):
    
    cell_types = []
    flag = "SharpDataset"
    
    def __init__(self,
                 split: str,
                 transform = None,
                 target_transform = None,
                ):
        """
        Attributes
        ----------
        split : str 
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split)
        
        self._cell_type = None  # Should this be used or removed?
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.info = INFO[self.flag]  # Ensure INFO is defined elsewhere
        try:
            file_path = os.path.join(self._data_path, f'{self.split}.tsv')
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        targets = df[ACTIVITY_COLUMNS].to_numpy()
        seq = df.seq.to_numpy()
        self.ds = {"targets": targets, "seq": seq}
        
    def split_parse(self, split: str) -> str:
        '''
        Parses the input split and returns a list of splits.
        
        Parameters
        ----------
        split : str
            Defines the data split, expected values: 'train', 'val', 'test'.
            
        Returns
        -------
        str
            A string containing the parsed split.
        '''
        
        # Default valid splits
        valid_splits = {"train", "val", "test"}
        
        # Process string input
        if split not in valid_splits:
            raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")
    
        return split
