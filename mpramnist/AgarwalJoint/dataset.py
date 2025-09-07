import pandas as pd
import numpy as np
from typing import List, T, Union, Dict, Optional
import torch
import os
import bioframe as bf
from mpramnist.mpradataset import MpraDataset

class AgarwalJointDataset(MpraDataset):

    CONSTANT_LEFT_FLANK = "AGGACCGGATCAACT" # required for each sequence
    CONSTANT_RIGHT_FLANK = "CATTGCGTGAACCGA" # required for each sequence
    LEFT_FLANK = "GGCCCGCTCTAGACCTGCAGG" # from human_legnet
    RIGHT_FLANK = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT" # from human_legnet
    
    CELL_TYPES = ['HepG2', 'K562', 'WTC11']
    FLAG = "AgarwalJoint"
    
    def __init__(self,
                 split: str | List[int] | int,
                 cell_type: str | List[str],
                 genomic_regions: Optional[Union[str, List[Dict]]] = None,
                 exclude_regions: bool = False,
                 root = None,
                 transform = None,
                 target_transform = None,
                ):
        """
        Attributes
        ----------
        split : str | List[int] | int
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        cell_type : str
            Specifies the cell type for filtering the data.
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool
            If True, exclude the specified regions instead of including them.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)
        
        if isinstance(cell_type, str):
            if cell_type not in self.CELL_TYPES:
                raise ValueError(f"Invalid cell_type: {cell_type}. Must be one of {self.CELL_TYPES}.")
            cell_type = [cell_type]
        if isinstance(cell_type, List):
            for i in range(len(cell_type)):
                act = cell_type[i]
                if act not in self.CELL_TYPES:
                    raise ValueError(f"Invalid cell_type: {act}. Must be one of {self.CELL_TYPES}.")
        self._cell_type = cell_type
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions

        self.prefix = self.FLAG + "_"
        
        try:
            file_name = self.prefix + 'joint_data' + '.tsv'
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Apply genomic region filtering
        df = self.filter_by_genomic_regions(df)
            
        target_column = self._cell_type

        if self.genomic_regions is None:
            self.ds = df[df.fold.isin(self.split)].reset_index(drop=True)
        else:
            self.ds = df
            self.split = "genomic region"
        
        targets = self.ds[target_column].to_numpy()
        seq = self.ds.seq.to_numpy()
        self.ds = {"targets" : targets, "seq" : seq}

    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on genomic regions using bioframe.
        
        """
        if self.genomic_regions is None:
            return df
        
        # Prepare the genomic regions for bioframe
        if isinstance(self.genomic_regions, str):
            # Load from BED file
            regions_df = bf.read_table(self.genomic_regions, schema='bed')
            regions_df['chrom'] = regions_df['chrom'].astype(str)
        else:
            # Convert list of dicts to DataFrame
            regions_df = pd.DataFrame(self.genomic_regions)
        
        # Prepare our data for bioframe intersection
        # Rename columns to match bioframe schema
        data_df = df.copy()
        data_df = data_df.rename(columns={
            'chr.hg38': 'chrom',
            'start.hg38': 'start',
            'stop.hg38': 'end'
        })
        
        # Convert to integer if possible
        for col in ['start', 'end']:
            if col in data_df.columns:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce').astype('Int64')
        
        # Find intersections
        intersections = bf.overlap(data_df, regions_df, how='inner', return_index=True)
        
        if self.exclude_regions:
            # Exclude sequences that overlap with specified regions
            filtered_df = df[~df.index.isin(intersections['index'])]
        else:
            # Include only sequences that overlap with specified regions
            filtered_df = df[df.index.isin(intersections['index'])]
        
        return filtered_df
        
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