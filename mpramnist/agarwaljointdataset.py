import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO

from .mpradataset import MpraDataset

class AgarwalJointDataset(MpraDataset):

    LEFT_FLANK = "GGCCCGCTCTAGACCTGCAGG" # from human_legnet
    RIGHT_FLANK = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT" # from human_legnet
    
    cell_types = ['HepG2', 'K562', 'WTC11']
    flag = "AgarwalJointDataset"
    
    def __init__(self,
                 split: str | List[int] | int,
                 cell_type: str | List[str],
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
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split)
        
        if isinstance(cell_type, str):
            if cell_type not in self.cell_types:
                raise ValueError(f"Invalid cell_type: {cell_type}. Must be one of {self.cell_types}.")
            cell_type = [cell_type]
        if isinstance(cell_type, List):
            for i in range(len(cell_type)):
                act = cell_type[i]
                if act not in self.cell_types:
                    raise ValueError(f"Invalid cell_type: {act}. Must be one of {self.cell_types}.")
        self._cell_type = cell_type
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.info = INFO[self.flag]
        
        try:
            df = pd.read_csv(self._data_path + 'joint_data.tsv', sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self._data_path + 'joint_data'}.tsv")
            
        target_column = self._cell_type
            
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