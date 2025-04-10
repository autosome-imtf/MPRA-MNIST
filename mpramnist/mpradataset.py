import pandas as pd
import numpy as np
from typing import List, T, Union, Optional, Callable
import torch

from torch.utils.data import  Dataset
from .dataclass import SeqObj, VectorDsFeature, ScalarDsFeature
from .info import INFO, HOMEPAGE, DEFAULT_ROOT

class MpraDataset(Dataset):
    
    """ SEQUENCE DATASET. """
    """
        Dataset for working with sequences and their associated features.
        This dataset provides functionality to split the data, apply transformations, and 
    access cell-type-specific information.
        
        Attributes
        ----------
        split : str | List[int] | int
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
    PARENT_FLAG = "MpraDaraset"
    
    
    def __init__(self,
                 split: str | List[int] | int | List[Union[int, str]],
                 root = None
                 permute = False,
                 root: str = DEFAULT_ROOT,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
                ):
        self.split = split
        self.permute = permute
        self.transform = transform
        self.target_transform = target_transform
        self._scalars = {}
        self._vectors = {}
        if root = None:
            self._data_path = "./../data/" + self.FLAG + "/" + self.FLAG + "_" 
        else:
            self._data_path = root
        self.info = INFO[self.FLAG]


    def __getitem__(self, idx):
        # Find all names start with 'seq' (e.g, 'seq', 'seq1', 'seq2', etc)
        seq_keys = [key for key in self.ds.keys() if key.startswith('seq')]
        
        seqs_datasets = {}
        for seq_key in seq_keys:
            sequence = self.ds[seq_key][idx]
            
            scals = {name: sc[idx] for name, sc in self.scalars.items()} if hasattr(self, 'scalars') else {}
            vecs = {name: vec[idx] for name, vec in self.vectors.items()} if hasattr(self, 'vectors') else {}
    
            Seq = SeqObj(seq=sequence, scalars=scals, vectors=vecs, split=self.split)
    
            if self.transform is not None:
                Seq = self.transform(Seq)
    
            if Seq.one_hot_encoded and self.permute:  # permute
                Seq.seq = Seq.seq.permute(1, 0)
                
            # Using original key name (seq, seq1, etc)
            seqs_datasets[seq_key] = Seq.seq
        
        target = torch.tensor(self.ds["targets"][idx].astype(np.float32))
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        if len(seqs_datasets) > 1:
            return seqs_datasets, target  # {seq : seq, seq1 : seq1, ..., targets}
        else:
            return seqs_datasets["seq"], target # sequences, targets
        
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
        #return len(self.ds) if self.ds is not None else 0
        return len(self.ds["seq"])

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} of size {self.__len__()} ({self.PARENT_FLAG})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Used split fold: {self.split}")
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
        