import pandas as pd
import numpy as np
from typing import List, T
import torch

from torch.utils.data import  Dataset

from dataclass import SeqObj, VectorDsFeature, ScalarDsFeature

class MpraDataset(Dataset):
    
    """ Sequence dataset. """
    
    def __init__(self, 
                 ds: pd.DataFrame,
                 transform = None):
        """
        Parameters
        ----------
        ds : pd.DataFrame
            Training dataset.
        transform (callable, optional): A function/transform that takes in NuclSeq dataclass object and returns a transformed version. Default: None.
        """
        self.ds = ds
        self.transform = transform
        self._scalars = {}
        self._vectors = {}
        
    def __getitem__(self, i):
        sequence = self.ds.seq.values[i]
        mean = self.ds.mean_value.values[i].astype(np.float32)
        
        scals = {name: sc[i] for name, sc in self.scalars.items()} if hasattr(self, 'scalars') else {}
        vecs = {name: vec[i] for name, vec in self.vectors.items()} if hasattr(self, 'vectors') else {}
        
        Seq = SeqObj(seq=sequence, scalars=scals, vectors=vecs)

        if self.transform is not None:
            Seq = self.transform(Seq)
            
        return Seq.seq, Seq.seqsize, mean

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
    