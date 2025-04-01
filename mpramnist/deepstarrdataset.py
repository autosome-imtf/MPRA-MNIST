import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO
import os
import warnings

from .mpradataset import MpraDataset


class DeepStarrDataset(MpraDataset):
    
    flag = "DeepStarrDataset"
    task = "regression"
    cell_types = ["Developmental", "HouseKeeping"]
    list_of_chr = ['chr2L', 'chr2LHet', 'chr2RHet', 'chr3L', 'chr3LHet', 'chr3R',
       'chr3RHet', 'chr4', 'chrX', 'chrXHet', 'chrYHet', 'chr2R']
    activity_columns = ["Dev_log2", "Hk_log2"]
    
    def __init__(self,
                 split: str | List[str],
                 activity_columns: str | List[str] = ["Dev_log2", "Hk_log2"],
                 use_original_reverse_complement: bool | None = None,
                 transform = None,
                 target_transform = None,
                ):
        """
        Attributes
        ----------
        split : str | List[str]
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        activity_columns : str | List[str]
            Specifies the cell type for filtering the data.
        use_original_reverse_complement : bool
            Determines whether to generate the reverse complement of sequences using the same approach as the original study
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split)
        
        self.activity_columns = activity_columns
        self._cell_type = activity_columns
        if use_original_reverse_complement is None:
            if isinstance(split, list) or split != 'train':
                use_original_reverse_complement = False
            else:
                use_original_reverse_complement = True
        self.transform = transform
        self.target_transform = target_transform
        self.split, column = self.split_parse(split)
        self.info = INFO[self.flag]  # Ensure INFO is defined elsewhere
        try:
            file_path = os.path.join(self._data_path, f'all_chr.tsv')
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = df[df[column].isin(self.split)].reset_index(drop=True)
        
        if use_original_reverse_complement:
            """
            WARNING: 
            This function uses original paper's parameters so:
            > The training dataset (`split=train`) is pre-augmented with reverse complements
            > - 2× sequences (original + RC)  
            > - Identical labels for RC pairs  
            > Manual reverse complementing will cause data leakage!
            """
            
            if self.split == ["train"]:
                warnings.warn("WARNING! "
                              "\nNote: The training set contains reverse-complement augmentation as implemented in the original study.  "
                                "\n• Dataset size: 2N (N original + N reverse-complemented sequences)  "
                                "\n• Label consistency: y_rc ≡ y_original  "
                                "\n• Do not reapply this transformation during preprocessing. ", stacklevel=1 )
                
            # reverse_complement
            rev_aug = df.copy()
            rev_aug.sequence = rev_aug.sequence.apply(self.reverse_complement)
            df = pd.concat([df, rev_aug], ignore_index =True)
            
        targets = df[self.activity_columns].to_numpy()
        seq = df.sequence.to_numpy()
        self.ds = {"targets": targets, "seq": seq}
        
    def reverse_complement(self, seq: str, mapping=None) -> str:
        if mapping is None:
            mapping = {"A": "T", "G": "C", "T": "A", "C": "G", "N": "N"}
        
        try:
            return "".join(mapping[base] for base in reversed(seq.upper()))
        except KeyError as e:
            raise ValueError(f"Invalid character in sequence: {e}")
        
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
        column = "" # which column to consider
        
        # Process string input
        if isinstance(split, str):
            if split in valid_splits:
                column = "split"
                return [split], column
            elif split in self.list_of_chr:
                column = "chr" 
                return [split], column
            else:
                raise ValueError(f"Invalid fold value: {split}. Must be one of {self.list_of_chr}.")
        # Validate list of folds
        elif isinstance(split, list):
            result = []
            column = "chr" 
            for item in split:
                if item in self.list_of_chr:
                    result.append(item)
                else:
                    raise ValueError(f"Invalid fold value: {item}. Must be one of {self.list_of_chr}.")
            return result, column
        else:
            raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', 'test' or {self.list_of_chr}.")
