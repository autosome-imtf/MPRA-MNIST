import pandas as pd
import numpy as np
from typing import List, T
import torch

from torch.utils.data import  Dataset
from dataclass import SeqObj, VectorDsFeature, ScalarDsFeature

class MpraDataset(Dataset):
    
    """ SEQUENCE DATASET. """
    
    Parent_flag = "MpraDaraset"
    split: str
    cell_type: str
    transform = None
    
    def __init__(self):
        """
        Parameters
        ----------
        split (string)
        cell_type (string)
        transform (callable, optional): A function/transform that takes in NuclSeq dataclass object and returns a transformed version. Default: None.
        """
        #self.info = INFO[self.flag] # cool method to info about dataset
        
        self._scalars = {}
        self._vectors = {}
        self._data_path = "./datasets/" + self.flag + "/"        
        
    def __getitem__(self, i):
        sequence = self.ds.seq.values[i]
        mean = self.ds.expression.values[i].astype(np.float32)
        
        scals = {name: sc[i] for name, sc in self.scalars.items()} if hasattr(self, 'scalars') else {}
        vecs = {name: vec[i] for name, vec in self.vectors.items()} if hasattr(self, 'vectors') else {}
        
        Seq = SeqObj(seq=sequence, scalars=scals, vectors=vecs, split = self.split)

        if self.transform is not None:
            Seq = self.transform(Seq)
            
        return Seq.seq, mean
        #return Seq
    
    
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

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} of size {self.__len__()} ({self.Parent_flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Split fold: {self.split}")
        body.append(f"Sequence size: {len(self.__getitem__(0)[0][0])}")
        body.append(f"Number of channels: {len(self.__getitem__(0)[0])}")
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

"""
        DATASETS
"""

class VikramDataset(MpraDataset):
    """
        Parameters
        ----------
        split (string): 'train', 'val' or 'test', required
        cell_type (string): "HepG2" is a human liver cancer cell line, "K562" is myelogenous leukemia cell line, "WTC11" is pluripotent stem cell line derived from adult skin 
    """
    cell_types = ['HepG2', 'K562', 'WTC11']
    flag = "VikramDataset"
    
    def __init__(self, split, cell_type, transform = None):
        super().__init__()
        
        self._cell_type = cell_type
        self.transform = transform
        self.split = self.split_parse(split)
        
        df = pd.read_csv(self._data_path + self._cell_type + '.tsv', sep='\t')
        self.ds = df[df.fold.isin(self.split)]
        
    def split_parse(self, split: list[int] | int | str) -> list[int]:
        
        split_default = {"train" : [1, 2, 3, 4, 5, 6, 7, 8], "val" : [9], "test" : [10]} # default split of data
        
        if isinstance(split, str):
            assert split in split_default.keys(), "Non-existing split value of data" # check if string correct
            split = split_default[split]
        if isinstance(split, list):
            for spl in split:
                assert spl < 11 and spl > 0, f"{spl} not in range 1 - 10" # check if list correct
        if isinstance(split, int):
            assert split < 11 and split > 0, f"{split} not in range 1 - 10"  # check if integer correct
            split = [split]
        
        return split
