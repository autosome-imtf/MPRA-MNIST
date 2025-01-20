import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO
from .mpradataset import MpraDataset

class SureDataset(MpraDataset):
    flag = "SureDataset"
    genome_ids = ["SuRE42_HG02601",
             "SuRE43_GM18983",
             "SuRE44_HG01241",
             "SuRE45_HG03464"]
    tasks = ["classification", "regression"]
    
    def __init__(self,
                 split: str,
                 genome_id: str,
                 task: str, # regression or classification. regression is default
                 transform = None,
                 target_transform = None,
                ):
        super().__init__(split)
        self.split = self.split_parse(split)
        self.task = task
        self.transform = transform
        self.target_transform = target_transform
        
        self.info = INFO[self.flag]
        
        if genome_id not in self.genome_ids:
            raise ValueError(f"genome_id value must be one of {genome_ids}")
        self.genome_id = genome_id
        self.cell_types = self.genome_ids
        self._cell_type = genome_id
        
        path_data = self._data_path + f"{self.genome_id}_Prepocessed_data/{self.genome_id}/final_sets/combined_bins_{self.split}_set"
        
        try:
            df = pd.read_csv(path_data + '.tsv', sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path_data}.tsv")
            
        if self.task == "classification":
            self.output_names = ["K562_bin", "HepG2_bin"]
            self.num_classes_per_output = [5, 5]
            self.num_outputs = np.sum(self.num_classes_per_output)
            
        elif self.task == "regression":
            self.output_names = ["avg_K562_exp", "avg_HepG2_exp"]
            self.num_outputs = 2

        targets = df[self.output_names].to_numpy()
        seq = df.sequence.to_numpy()
        self.ds = {"targets" : targets, "seq" : seq}
        
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
        split_default = ["train", 
                         "val", 
                         "test"
                         ] # default split of data
        
        # Default valid splits
        valid_splits = {"train", "val", "test"}
        
        # Process string input
        if split not in valid_splits:
            raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")
    
        return split
