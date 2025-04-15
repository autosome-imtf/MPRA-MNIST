import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, T, Union
import torch
import os

from mpramnist.mpradataset import MpraDataset


class DeepPromoterDataset(MpraDataset):
    
    FLAG = "DeepPromoter"
    
    def __init__(self,
                 split: str,
                 transform = None,
                 target_transform = None,
                 root = None
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
        super().__init__(split, root)
        
        self.activity_columns = "target"
        self._cell_type = None
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"

        try:
            file_name = self.prefix + 'all_seqs' + '.tsv'
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
                        
        df = df[df["split"].isin(self.split)].reset_index(drop=True)
        
        self.df = df
        targets = self.df[self.activity_columns].to_numpy()
        seq = self.df.sequence.to_numpy()
        
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
    
        return [split]
