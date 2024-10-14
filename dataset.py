import pandas as pd
import numpy as np
from typing import List, T
import torch

from torch.utils.data import  Dataset

from dataclass import SeqObj, VectorDsFeature, ScalarDsFeature

class MpraDataset(Dataset):
    
    """ Sequence dataset. """

    flag = ...
    
    def __init__(self,
                 split,
                 cell_type: str = "",
                 transform = None):
        """
        Parameters
        ----------
        split (string): 'train', 'val' or 'test', required
        cell_type (string)
        transform (callable, optional): A function/transform that takes in NuclSeq dataclass object and returns a transformed version. Default: None.
        """
        #self.info = INFO[self.flag] # cool method to info about dataset
        
        self.transform = transform
        self._scalars = {}
        self._vectors = {}
        self._data_path = "./datasets/" + self.flag + "/"

        self.split = split
        self.ds = self.splitting(self._data_path, self.split, cell_type)
        
        
    def __getitem__(self, i):
        sequence = self.ds.seq.values[i]
        mean = self.ds.mean_value.values[i].astype(np.float32)
        
        scals = {name: sc[i] for name, sc in self.scalars.items()} if hasattr(self, 'scalars') else {}
        vecs = {name: vec[i] for name, vec in self.vectors.items()} if hasattr(self, 'vectors') else {}
        
        Seq = SeqObj(seq=sequence, scalars=scals, vectors=vecs)

        if self.transform is not None:
            Seq = self.transform(Seq)
            
        return Seq.seq, mean
        #return Seq
    
    def splitting(self, data_path, split, cell_type):
        split_default = {"train" : [0,8], "val" : [9], "test" : [10]}
        dataframe = pd.read_csv(data_path + cell_type + '.tsv', sep='\t')
        
        if isinstance(split,str):
            if split in split_default:
                if len(split_default[split]) == 1:
                    df = dataframe[dataframe.fold == split_default[split][0]]
                else:
                    df = dataframe[(dataframe.fold >= 0) & (dataframe.fold <= 8)]
            else:
                raise ValueError
        elif isinstance(split, list):
            for spl in split:
                assert spl < 11 and spl > 0, f"{spl} not in range 1 - 10" 
            if len(split) == 2:
                df = dataframe[(dataframe.fold >= split[0]) & (dataframe.fold <= split[1])]
            elif len(split) == 1:
                df = dataframe[dataframe.fold == split[0]]
            else:
                raise ValueError
        elif isinstance(split, int):
            assert split < 11 and split > 0, f"{split} not in range 1 - 10" 
            df = dataframe[dataframe.fold == split]
        else:
            raise ValueError
                
        return df
    @property
    def scalars(self):
        return self._scalars

    @property
    def vectors(self):
        return self._vectors

    def add_numeric_scalar(self, name: str, val: List[T]):
        self._scalars[name] = ScalarDsFeature.numeric(val=val)

    def add_categorial_scalar(self, name: str, val: List[T], levels: dict[T, int] | None = None):
        self._scalars[name] = ScalarDsFeature.categorial(val=val, levels=levels)

    def add_numeric_vector(self, name: str, val: List[List[T]], pad_value: T):
        self._vectors[name] = VectorDsFeature.numeric(val=val, pad_value=pad_value)

    def add_categorial_vector(self, name: str, val: List[List[T]], pad_value: T, levels: dict[T, int] | None = None):
        self._vectors[name] = VectorDsFeature.categorial(val=val, pad_value=pad_value, levels=levels)
    
    def __len__(self):
        return len(self.ds.seq)

class VikramDataset(MpraDataset):
    flag = "VikramDataset"
        