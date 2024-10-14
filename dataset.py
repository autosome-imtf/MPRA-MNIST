import pandas as pd
import numpy as np
from typing import List, T
import torch

from torch.utils.data import  Dataset

from dataclass import SeqObj, VectorDsFeature, ScalarDsFeature

class MpraDataset(Dataset):
    
    """ Sequence dataset. """
    
    def __init__(self, 
                 ds: pd.DataFrame = None,
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
            
        #return Seq.seq, Seq.seqsize, Seq.scalars, Seq.vectors, mean
        return Seq

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
    def __init__(self, pad_value = None, transform = None):
        super().__init__()
        
        seqs = ["agtactagagtc","atatatatatat"]
        scal = [1,2]
        vect = [[1,0,1,0,1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1,0,1,0,1]]
        self.ds = pd.DataFrame({"seq" : seqs,"scalar1": scal,"vector1": vect,"mean_value":[5,6]})
        
        if pad_value is None: #user defines pad_value
            pad_value = len(self.ds.seq.values[0])
        self.pad_value = pad_value
        
        self.add_numeric_scalar("scalar1", val = self.ds.scalar1.values)
        self.add_categorial_vector("vector1", val = self.ds.vector1.values, pad_value = self.pad_value)
        self.transform = transform
    