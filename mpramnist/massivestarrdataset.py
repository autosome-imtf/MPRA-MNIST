import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO
import pyfastx
from .mpradataset import MpraDataset

class MassiveStarrDataset(MpraDataset):
    
    flag = "MassiveStarrSeqDataset"
    tasks = ["RandomEnhancer", "GenomeEnhancer", "GenomePromoter", "DifferentialExpression", "CapturePromoter", "AtacSeq", "binary"]
    
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
        #define task
        if task == "RandomEnhancer":
            task = "./ranEnh/"
        #load data files
        if split == "test":
            positive = pyfastx.Fastx(f'{self._data_path}{task}{split}_pos.fasta.gz')
            pos_seqs = []
            pos_labels = []
            for name,seq in positive:
                pos_labels.append(np.float32(1))
                pos_seqs.append(seq)
            negative = pyfastx.Fastx(f'{self._data_path}{task}{split}_neg.fasta.gz')
            neg_seqs =  []
            neg_labels = []
            for name,seq in negative:
                neg_labels.append(np.float32(0))
                neg_seqs.append(seq)
            return {"targets": pos_labels + neg_labels, "seq": pos_seqs + neg_seqs}
            
        fa = pyfastx.Fastx(f'{self._data_path}{task}{split}.fasta.gz')
        seqs = []
        labels = []
        for name,seq in fa:
            seqs.append(seq)
        label1, label0 = [np.float32(1) for i in seqs], [np.float32(0) for i in seqs]

        return {"targets": label1 + label0, "seq": seqs}
        
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