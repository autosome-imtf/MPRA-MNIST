import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, T, Union
import torch

import os

from mpramnist.mpradataset import MpraDataset


class EvfratovDataset(MpraDataset):
    
    FLAG = "Evfratov"
    def __init__(self,
                 split: str,
                 length_of_seq: Union[str, int] = "23",  # 23 or 33
                 merge_last_classes: bool = False,
                 transform = None,
                 target_transform = None,
                 root = None
                ):
        """
        Attributes
        ----------
        split : str 
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        length_of_seq: str | int
            Defines a dataset to use: with 20 long sequences or 30 long sequences
        merge_last_classes : bool
            A flag indicating whether to merge the last two classes in the dataset.
            If True, the last two classes (typically those with the fewest examples)
            will be merged into one. This can be useful for addressing class imbalance
            or simplifying the classification task. If False, the classes remain
            unchanged. By default, it is recommended to set this to False unless the
            user is certain that merging is necessary.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)
        
        self.activity_columns = "label"
        self._cell_type = None
        self.length_of_seq = str(length_of_seq)
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"
        
        try:
            file_path = os.path.join(self._data_path, self.prefix + f'{self.length_of_seq}_{self.split}.tsv')
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
            
        sequences = df['sequence']
        df_counts = df.drop(columns=['sequence'])

        # Convert counts to distributions
        row_sums = df_counts.sum(axis=1)
        df = df_counts.div(row_sums, axis=0).fillna(0)  #  change NaN to 0 if raw sum is 0

        # Assign labels based on the column with the maximum value
        df[self.activity_columns] = df.idxmax(axis=1)
        df[self.activity_columns] = df[self.activity_columns].apply(lambda x: df.columns.get_loc(x))
                                                                    
        df["sequence"] = sequences

        # Merge last two classes if required
        if merge_last_classes:
            df[self.activity_columns] = df[self.activity_columns].replace(7, 6)
            self.n_classes = 7
        else:
            self.n_classes = 8
            
        # Replace 'U' with 'T' in sequences
        df["sequence"] = df["sequence"].str.upper().str.replace("U", "T")
        
        self.df = df
        targets = self.df[self.activity_columns].to_numpy()
        seq = self.df.sequence.to_numpy()
        
        self.ds = {"targets": targets, "seq": seq}

    def hist_plot(self):
        data = self.df.label  
        n_classes = self.n_classes  
        
        counts, bins = np.histogram(data, bins=n_classes, range=(0, n_classes))
        
        x = np.arange(n_classes) 
        plt.bar(x, counts, color='skyblue', edgecolor='black')
        
        for i, count in enumerate(counts):
            plt.text(
                x[i],  
                count,  
                int(count),  
                ha='center',  
                va='bottom', 
                fontsize=10, 
                fontweight='bold' 
            )
        
        plt.xlabel('Labels')
        plt.ylabel('Quantity')
        plt.title('Histogram count of label')
        
        plt.xticks(x)  
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.show()
    def split_parse(self, split: str) -> str:
        '''
        Parses the input split and returns a list of splits.
        
        Parameters
        ----------
        split : str
            Defines the data split, expected values: 'train', 'val', 'test'.
            
        Returns
        -------
        str
            A string containing the parsed split.
        '''
        
        # Default valid splits
        valid_splits = {"train", "val", "test"}
        
        # Process string input
        if split not in valid_splits:
            raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")
    
        return split
