import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, T, Union
import torch
from .info import INFO
import os

from .mpradataset import MpraDataset


class DreamDataset(MpraDataset):
    
    flag = "DreamDataset"
    task = "regression"
    cell_types = None
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
        
        self.activity_columns = "target"
        self._cell_type = None

        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.info = INFO[self.flag]  # Ensure INFO is defined elsewhere
        
        try:
            file_path = os.path.join(self._data_path, f'{self.split}.tsv')
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.df = df
        targets = self.df[self.activity_columns].to_numpy()
        seq = self.df.seq.to_numpy()
        
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
