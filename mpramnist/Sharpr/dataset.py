import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
import os

from mpramnist.mpradataset import MpraDataset

class SharprDataset(MpraDataset):
    ACTIVITY_COLUMNS  = ["k562_minp_rep1", "k562_minp_rep2", "k562_minp_avg", \
                             "k562_sv40p_rep1", "k562_sv40p_rep2", "k562_sv40p_avg", \
                             "hepg2_minp_rep1", "hepg2_minp_rep2", "hepg2_minp_avg", \
                             "hepg2_sv40p_rep1", "hepg2_sv40p_rep2", "hepg2_sv40p_avg"]
    FLAG = "Sharpr"
    
    def __init__(self,
                 split: str,
                 activity_columns: List[str],
                 transform = None,
                 target_transform = None,
                 root = None
                ):
        """
        Attributes
        ----------
        split : str 
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        activity_columns : List[str]
            List of column names with activity data to be used.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)
        
        self._cell_type = None  # Should this be used or removed?
        self.transform = transform
        self.target_transform = target_transform
        self.activity_columns = activity_columns
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"
        
        try:
            file_name = self.prefix + self.split + '.tsv'
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        targets = df[self.activity_columns].to_numpy()
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
