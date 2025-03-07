import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO
from torch.nn.utils.rnn import pad_sequence
from .mpradataset import MpraDataset

class SureDataset(MpraDataset):
    flag = "SureDataset"
    
    GENOME_IDS = ["SuRE42_HG02601",
             "SuRE43_GM18983",
             "SuRE44_HG01241",
             "SuRE45_HG03464"]
    tasks = ["classification", "regression"]
    
    def __init__(self,
                 split: str,
                 genome_id: str,
                 task: str, # regression or classification. regression is default
                 permute = True,
                 transform = None,
                 target_transform = None,
                ):
        super().__init__(split, permute)
        self.split = self.split_parse(split)
        self.task = task
        self.permute = permute
        self.transform = transform
        self.target_transform = target_transform
        
        self.info = INFO[self.flag]
        
        self.genome_id = genome_id
        self.cell_types = self.GENOME_IDS
        self._cell_type = genome_id
        if isinstance(genome_id, list):
            pass
        else:
            self.genome_id = [genome_id]
        self.ds = {}
        for genome in self.genome_id:
            
            if genome not in self.GENOME_IDS:
                raise ValueError(f"genome_id value must be one of {self.GENOME_IDS}")
                
            path_data = self._data_path + f"{genome}_Prepocessed_data/{genome}/final_sets/combined_bins_{self.split}_set"
            
            try:
                df = pd.read_csv(path_data + '.tsv', sep='\t')
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {path_data}.tsv")

            def compute_classification_label(k562, hepg2):
                return 5* int(k562) + int(hepg2)
                
            if self.task == "classification":
                self.output_names = ["K562_bin", "HepG2_bin"]
                self.num_classes_per_output = [5, 5]
                self.num_outputs = np.sum(self.num_classes_per_output)
                '''
                df['label'] = df.apply(lambda x: compute_classification_label(x.K562_bin, x.HepG2_bin), axis=1)
                targets = df["label"].to_numpy()  
                '''
                targets = df[self.output_names].to_numpy() 
                
            elif self.task == "regression":
                self.output_names = ["avg_K562_exp", "avg_HepG2_exp"]
                self.num_outputs = 2
                targets = df[self.output_names].to_numpy()  

            seq = df.sequence.to_numpy()
            
            if "targets" in self.ds:
                self.ds["targets"] = np.concatenate((self.ds["targets"], targets))

                self.ds["seq"] = np.concatenate((self.ds["seq"], seq))
            else:
                self.ds["targets"] = targets
                self.ds["seq"] = seq
    
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
