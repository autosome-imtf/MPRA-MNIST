import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO
import pyfastx
from .mpradataset import MpraDataset

class MassiveStarrDataset(MpraDataset):
    
    flag = "MassiveStarrSeqDataset"
    tasks = ["randomenhancer", "genomeenhancer", "genomepromoter", "differentialexpression", "capturepromoter", "atacseq", "binary"]
    
    def __init__(self,
                 task: str,
                 split: str | List[int] | int,
                 transform = None,
                 target_transform = None,
                ):
        """
        Attributes
        ----------
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split)
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.split = self.split_parse(split)
        self.info = INFO[self.flag]

        self.ds = self.load_data(self.split, self.task)

    def load_data(self, split, task):
        task = self.define_task(task)
        
        #load data files
        fa = pyfastx.Fastx(f'{self._data_path}{task}{split}.fasta.gz')
        seqs = []
        labels = []
        for name,seq in fa:
            seqs.append(seq)
        label1, label0 = [np.float32(1) for i in range(len(seqs)//2)], [np.float32(0) for i in range(len(seqs)//2)]

        return {"targets": label1 + label0, "seq": seqs}
        
    def define_task(self, name):
        tasks = ["./ranEnh/", "./genEnh/", "./genProm/", "./diffExp/", "./CaptProm/", "./ATACSeq/", "./binary/"]
        if name.lower() in self.tasks:
            return tasks[self.tasks.index(name.lower())]
        
    def split_parse(self, split: list[int] | int | str) -> list[int]:
        '''
        Parses the input split and returns a list of folds.
        '''
        split_default = {"train" : "train", 
                         "val" : "val", 
                         "test" : "test"
                        }
        if isinstance(split, str):
            if split not in split_default:
                raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")
            split = split_default[split]
        
        return split