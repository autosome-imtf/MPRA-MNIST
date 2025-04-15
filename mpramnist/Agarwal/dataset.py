import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
import os

from mpramnist.mpradataset import MpraDataset

class AgarwalDataset(MpraDataset):

    LEFT_FLANK = "GGCCCGCTCTAGACCTGCAGG" # from human_legnet
    RIGHT_FLANK = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT" # from human_legnet
    
    CELL_TYPES = ['HepG2', 'K562', 'WTC11']
    FLAG = "Agarwal"
    
    def __init__(self,
                 split: str | List[int] | int,
                 cell_type: str,
                 averaged_target: bool = False,
                 root = None,
                 transform = None,
                 target_transform = None,
                ):
        """
        Attributes
        ----------
        split : str | List[int] | int
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        cell_type : str
            Specifies the cell type for filtering the data.
        averaged_target : bool
            Use target column with averaged expression or not
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)
        
        if cell_type not in self.CELL_TYPES:
            raise ValueError(f"Invalid cell_type: {cell_type}. Must be one of {self.CELL_TYPES}.")
        self._cell_type = cell_type
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"
        
        try:
            file_name = self.prefix + self._cell_type + '.tsv'
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
            
        target_column = "averaged_expression" if averaged_target else "expression"
        #df["target"] = df[target_column].astype(np.float32)
        #self.target = target_column
            
        self.ds = df[df.fold.isin(self.split)].reset_index(drop=True)
        
        targets = self.ds[target_column].to_numpy()
        seq = self.ds.seq.to_numpy()
        self.ds = {"targets" : targets, "seq" : seq}
        
    def split_parse(self, split: list[int] | int | str) -> list[int]:
        '''
        Parses the input split and returns a list of folds.
        '''
        
        split_default = {"train" : [1, 2, 3, 4, 5, 6, 7, 8], 
                         "val" : [9], 
                         "test" : [10]
                        } # default split of data
        
        # Process string input
        if isinstance(split, str):
            if split not in split_default:
                raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")
            split = split_default[split]
        
        # int to list for unified processing
        if isinstance(split, int):
            split = [split]
            
        # Check the range of values for a list
        if isinstance(split, list):
            for spl in split:
                if not (1 <= spl <= 10):
                    raise ValueError(f"Fold {spl} not in range 1-10.")
        
        return split