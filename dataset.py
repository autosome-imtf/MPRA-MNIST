import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from functools import partial

from torch.utils.data import  Dataset
from dataclass import SeqObj, VectorDsFeature, ScalarDsFeature
from info import INFO, HOMEPAGE, DEFAULT_ROOT

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
                 split: str | List[int] | int,
                 cell_type: str = None,
                 download: bool = False,
                 root: str = DEFAULT_ROOT,
                 transform = None,
                 target_transform = None
                ):
        self.split = split
        self._cell_type = cell_type
        self.transform = transform
        self.target_transform = target_transform
        self._scalars = {}
        self._vectors = {}
        self._data_path = "./datasets/" + self.flag + "/"
        self.info = INFO[self.flag]

    def __getitem__(self, i):
        
        sequence = self.ds.seq.values[i]
        target = self.ds.target.values[i].astype(np.float32)
        
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
        
"""
        DATASETS
"""

class VikramDataset(MpraDataset):
    """
        Attributes
        ----------
        split : str | List[int] | int
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        cell_type : str
            Specifies the cell type for filtering the data.
        averaged_target : bool
            Use target column with averaged expression or not
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
    """
    cell_types = ['HepG2', 'K562', 'WTC11']
    flag = "VikramDataset"
    
    def __init__(self,
                 split: str | List[int] | int,
                 cell_type: str,
                 averaged_target: bool = False,
                 transform = None,
                 target_transform = None,
                ):
        super().__init__(split, cell_type)
        if cell_type not in self.cell_types:
            raise ValueError(f"Invalid cell_type: {cell_type}. Must be one of {self.cell_types}.")
        self._cell_type = cell_type
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        
        try:
            df = pd.read_csv(self._data_path + self._cell_type + '.tsv', sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self._data_path + self._cell_type}.tsv")
            
        target_column = "averaged_expression" if averaged_target else "expression"
        df["target"] = df[target_column].astype(np.float32)
            
        self.ds = df[df.fold.isin(self.split)].reset_index(drop=True)
        
    def split_parse(self, split: list[int] | int | str) -> list[int]:
        '''
        Parses the input split and returns a list of folds.
        '''
        
        split_default = {"train" : [1, 2, 3, 4, 5, 6, 7, 8], 
                         "val" : [9], 
                         "test" : [10]
                        } # default split of data
        
        # Process string input
        if isinstance(split, str):
            if split not in split_default:
                raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")
            split = split_default[split]
        
        # int to list for unified processing
        if isinstance(split, int):
            split = [split]
            
        # Check the range of values for a list
        if isinstance(split, list):
            for spl in split:
                if not (1 <= spl <= 10):
                    raise ValueError(f"Fold {spl} not in range 1-10.")
        
        return split

class MalinoisDataset(MpraDataset):
    """
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
    cell_types = ['HepG2', 'K562', 'SKNSH']
    data_project = ['UKBB', 'GTEX', 'CRE']
    project_column = 'data_project'
    sequence_column = 'sequence'
    chr_column = 'chr'
    stderr_columns = ['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']
    flag = "MalinoisDataset"
    
    def __init__(self,
                 split: str | List[Union[int, str]] | int,
                 cell_type: str,
                 activity_columns = ['K562_log2FC', 'HepG2_log2FC', 'SKNSH_log2FC'],
                 normalize: bool = False,
                 duplication_cutoff: float = 0.5,
                 stderr_threshold: float = 1.0,
                 std_multiple_cut: float = 6.0,
                 up_cutoff_move: float = 4.0,
                 padded_seq_len: int = 600, 
                 synth_seed: int = 0,
                 transform = None,
                 target_transform = None,
                ):
        super().__init__(split, cell_type)
        
        if cell_type not in self.cell_types:
            raise ValueError(f"Invalid cell_type: {cell_type}. Must be one of {self.cell_types}.")
        self.activity_columns = activity_columns
        self.normalize = normalize
        self.duplication_cutoff = duplication_cutoff
        self.stderr_threshold = stderr_threshold
        self.std_multiple_cut = std_multiple_cut
        self.up_cutoff_move = up_cutoff_move
        self.padded_seq_len = padded_seq_len
        self._cell_type = cell_type
        self.transform = transform
        self.target_transform = target_transform

        self.split = self.split_parse(split)
        self.ds = self._load_and_filter_data()

    def _load_and_filter_data(self):
        """
        Preprocesses the dataset based on provided parameters.
        """
        columns = [self.sequence_column, *self.activity_columns, self.chr_column, self.project_column, *self.stderr_columns]
        try:
            df = pd.read_csv(self._data_path + 'Malinois.tsv', sep='\t', low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self._data_path}.tsv")

        df = df[columns].dropna()
        
        # set name of sequence column to seq
        df.rename(columns = {'sequence': 'seq'}, inplace = True) 
        self.sequence_column = "seq"
        
        # define target_column
        target_column = self._cell_type + "_log2FC" 
        df["target"] = df[target_column].astype(np.float32) 
        
        df = df[df[self.project_column].isin(self.data_project)].reset_index(drop=True)

        # filter by stderr
        # Threshold for feature stderr of examples to be used for train/val/test
        quality_filter = df[self.stderr_columns].max(axis = 1) < self.stderr_threshold 
        df = df[quality_filter].reset_index(drop=True)

        # Cut-off for extreme value filtering
        means = df[self.activity_columns].mean().to_numpy()
        stds  = df[self.activity_columns].std().to_numpy()

        # Cut-off shift for extreme value filtering
        up_cut   = means + stds * self.std_multiple_cut + self.up_cutoff_move
        down_cut = means - stds * self.std_multiple_cut 
        
        non_extremes_filter_up = (df[self.activity_columns] < up_cut).to_numpy().all(axis=1)
        df = df.loc[non_extremes_filter_up]
        
        non_extremes_filter_down = (df[self.activity_columns] > down_cut).to_numpy().all(axis=1)
        df = df.loc[non_extremes_filter_down]
        self.num_examples = len(df)
        if self.normalize:   
            df[self.activity_columns] = (df[self.activity_columns] - means) / stds
            self.activity_means = torch.Tensor(means)
            self.activity_stds = torch.Tensor(stds)

        for idx, cell in enumerate(self.activity_columns):
            cell_name = cell.rstrip('_mean')
            top_cut_value = round(up_cut[idx], 2)
            bottom_cut_value = round(down_cut[idx], 2)
            print(f'{cell_name} | top cut value: {top_cut_value}, bottom cut value: {bottom_cut_value}')
        print('')    
        num_up_cuts   = np.sum(~non_extremes_filter_up)
        num_down_cuts = np.sum(~non_extremes_filter_down)
        print(f'Number of examples discarded from top: {num_up_cuts}')
        print(f'Number of examples discarded from bottom: {num_down_cuts}')
        print('')
        print(f'Number of examples available: {self.num_examples}')
        print('-'*50)
        print('')

        df = df[df.chr.isin(self.split)].reset_index(drop=True)
        
        activities = df[self.activity_columns].to_numpy()
        activity_tensor = torch.Tensor(activities)
        sort_tensor = torch.max(activity_tensor, dim = -1).values

        #duplicating
        self.n_duplicated = 0
        if self.duplication_cutoff is not None:
            
            self.n_duplicated = (sort_tensor >= self.duplication_cutoff).sum().item()
            print(df.seq.shape[0])
            print(self.n_duplicated)
        return df

    def __len__(self):
        #dataset_len = self.ds.seq.shape[0] + self.n_duplicated
        dataset_len = self.ds.seq.shape[0]
        return dataset_len


    def split_parse(self, split: list[Union[int, str]] | int | str) -> list[str]:
        '''
        Parses the input split and returns a list of folds.
        '''
        
        split_default = {"train" : [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 22, "Y"], 
                         "val" : [19, 21, "X"], 
                         "test" : [7, 13]
                        } # default split of data
        
        # Process string input for specific keys or fold names ("X", "Y")
        if isinstance(split, str):
            if split in ["X", "Y"]:
                split = [split]
            elif split in split_default:
                split = split_default[split]
            else:
                raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")

        # int to list for unified processing
        if isinstance(split, int):
            split = [split]
            
        # Validate list of folds
        if isinstance(split, list):
            result = []
            for item in split:
                if isinstance(item, int) and 1 <= item <= 22:
                    result.append(str(item))
                elif isinstance(item, str) and item in ["X", "Y"]:
                    result.append(item)
                else:
                    raise ValueError(f"Invalid fold value: {item}. Must be an integer in range 1-22 or 'X'/'Y'.")

            split = result  # Ensure final result is clean and validated
        
        return split