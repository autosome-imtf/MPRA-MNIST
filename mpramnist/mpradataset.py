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
        cell_type : str
            Specifies the cell type for filtering the data.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
    Parent_flag = "MpraDaraset"
    
    
    def __init__(self,
                 split: str | List[int] | int | List[Union[int, str]],
                 cell_type: str | List[str] = None,
                 download: bool = False,
                 root: str = DEFAULT_ROOT,
                 transform: Optional[Callable] = None,
                  target_transform: Optional[Callable] = None
                ):
        self.split = split
        self._cell_type = cell_type
        self.transform = transform
        self.target_transform = target_transform
        self._scalars = {}
        self._vectors = {}
        self._data_path = "./../datasets/" + self.flag + "/"
        self.info = INFO[self.flag]

    def __getitem__(self, i):
        
        sequence = self.ds.seq.values[i]
        target = self.ds[self.target].values[i].astype(np.float32)
        
        scals = {name: sc[i] for name, sc in self.scalars.items()} if hasattr(self, 'scalars') else {}
        vecs = {name: vec[i] for name, vec in self.vectors.items()} if hasattr(self, 'vectors') else {}

        Seq = SeqObj(seq=sequence, scalars=scals, vectors=vecs, split = self.split)

        if self.transform is not None:
            Seq = self.transform(Seq)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return Seq.seq, target
        
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
        return len(self.ds) if self.ds is not None else 0

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} of size {self.__len__()} ({self.Parent_flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Default split folds: {self.info['folds']}")
        body.append(f"Used split fold: {self.split}")
        body.append(f"Scalar features: {self.info['scalar_features']}")
        body.append(f"Vector features: {self.info['vector_features']}")
        body.append(f"Cell types: {self.cell_types}")
        body.append(f"Сell type used: {self._cell_type}")
        body.append(f"Target columns that can be used: {self.info['target_columns']}")
        body.append(f"Number of channels: {len(self.__getitem__(0)[0])}")
        body.append(f"Sequence size: {len(self.__getitem__(0)[0][0])}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
        