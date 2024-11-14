import pandas as pd
import numpy as np
from typing import List, T, Union, Optional, Callable
import torch
import os

from .mpradataset import MpraDataset

class MalinoisDataset(MpraDataset):
    LEFT_FLANK = "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG" # from boda dataset
    RIGHT_FLANK = "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT" # from boda dataset
    
    cell_types = ['HepG2', 'K562', 'SKNSH']
    project_column = 'data_project'
    sequence_column = "sequence"
    chr_column = 'chr'
    flag = "MalinoisDataset"
    
    def __init__(self,
                 split: str | List[Union[int, str]] | int,
                 filtration: str = "original", # 'original', 'own' or 'none'
                 activity_columns: List[str] = ['K562_log2FC', 'HepG2_log2FC', 'SKNSH_log2FC'],
                  stderr_columns: List[str] = ['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE'],
                  data_project: List[str] = ['UKBB', 'GTEX', 'CRE'],
                  duplication_cutoff: Optional[float] = None,
                  stderr_threshold: float = 1.0,
                  std_multiple_cut: float = 6.0,
                  up_cutoff_move: float = 4.0,
                  transform: Optional[Callable] = None,
                  target_transform: Optional[Callable] = None,
                ):
        """
        Initializes the dataset loader with specified filtering, duplication, and transformation settings.
        
        Parameters
        ----------
        split : Union[str, List[Union[int, str]], int]
            Specifies the data split to use (e.g., 'train', 'val', 'test', or list of indices).
        filtration : str
            Specifies the filtering method. Options are 'original', 'own', or 'none'.
        activity_columns : List[str]
            List of column names with activity data to be used for filtering and duplication.
        stderr_columns : List[str]
            List of column names with standard error values used for quality filtering.
        data_project : List[str]
            Specifies the data projects to include in filtering.
        duplication_cutoff : Optional[float]
            If specified, sequences with a maximum activity value above this threshold will be duplicated.
        stderr_threshold : float
            Maximum allowed standard error threshold for filtering rows.
        std_multiple_cut : float
            The multiplier applied to standard deviations to calculate the upper cut-off threshold.
        up_cutoff_move : float
            Shift value for adjusting the upper cut-off threshold.
        transform : Optional[Callable]
            Transformation function applied to each sequence object.
        target_transform : Optional[Callable]
            Transformation function applied to the target data.
        """
        # Validate filtration parameter
        if filtration not in {"original", "own", "none"}:
            raise ValueError("filtration must be one of {'original', 'own', 'none'}")
            
        super().__init__(split)
        
        # Assign attributes
        self.stderr_columns = stderr_columns
        self.data_project = data_project
        self.duplication_cutoff = duplication_cutoff
        self.stderr_threshold = stderr_threshold
        self.std_multiple_cut = std_multiple_cut
        self.up_cutoff_move = up_cutoff_move
        self.filtration = filtration
        self.transform = transform
        self.target_transform = target_transform
        
        # Parse columns and split parameters
        activity_columns = self.activity_columns_parse(activity_columns)
        self.split = self.split_parse(split)

        # Load and process the dataset
        self.ds = self._load_and_filter_data(activity_columns)
        self._cell_type = activity_columns
        self.target = activity_columns # initialization for MpraDataset.__getitem__()

    def _load_and_filter_data(self, activity_columns):
        """
        Loads and preprocesses the dataset by selecting specific columns, handling missing values, 
        renaming columns, filtering data based on project and filtration type, splitting by chromosome, 
        and handling duplication.
    
        Parameters:
        -----------
        activity_columns : list of str
            List of columns containing activity data in the dataset.
    
        Returns:
        --------
        pd.DataFrame
            Filtered and processed DataFrame.
    
        Raises:
        -------
        FileNotFoundError
            If the specified data file is not found.
        ValueError
            If `filtration` is not one of ['original', 'own', 'none'].
        """
        
        file_path = os.path.join(self._data_path, 'Malinois.tsv')
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Select columns and drop rows with NaN in critical columns
        columns = [self.sequence_column, *activity_columns, self.chr_column, self.project_column, *self.stderr_columns]
        df = df[columns].dropna()
        
        # Rename sequence column
        self.sequence_column = "seq"
        df.rename(columns = {'sequence': self.sequence_column}, inplace = True) 

        # Filter by project column
        df = df[df[self.project_column].isin(self.data_project)].reset_index(drop=True)

        # Apply filtration
        filters = {
            "original": lambda df: self._filter_data(df, activity_columns),
            "own": lambda df: self._filter_data(df, activity_columns, self.stderr_columns, 
                                                self.data_project, self.duplication_cutoff, 
                                                self.stderr_threshold, self.std_multiple_cut, self.up_cutoff_move),
            "none": lambda df: df
        }
        if self.filtration in filters:
            df = filters[self.filtration](df)
        else:
            raise ValueError("filtration value must be 'original', 'own' or 'none'")
            
        # Split data by chromosome
        df = df[df[self.chr_column].isin(self.split)].reset_index(drop=True)
        
        # Handle duplication if specified
        if self.duplication_cutoff is not None:
            df = self.duplicate_high_activity_rows(df, activity_columns)
            
        return df

    def _filter_data(self, df: pd.DataFrame,
                     activity_columns: List[str],
                     stderr_columns = None,
                     data_project = None,
                     duplication_cutoff = 0.5,
                     stderr_threshold = 1.0,
                     std_multiple_cut = 6.0,
                     up_cutoff_move = 4.0) -> pd.DataFrame:
        '''
        Filters the DataFrame based on specified thresholds for standard error and activity metrics.
    
        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing data to be filtered.
        activity_columns : List[str]
            List of columns containing activity data for threshold-based filtering.
        stderr_columns : List[str], optional
            List of columns to assess standard error. Defaults to specific columns if not provided.
        data_project : List[str], optional
            List of projects to filter by, though not directly used here; can be customized further if needed.
        duplication_cutoff : float, optional
            The cutoff threshold for handling duplicates. Unused in this function but provided for extensibility.
        stderr_threshold : float, optional
            Threshold for standard error filtering; rows with max `stderr_columns` values above this are removed.
        std_multiple_cut : float, optional
            Multiplier for the standard deviation used to set upper and lower activity thresholds.
        up_cutoff_move : float, optional
            Value added to the upper cutoff threshold to allow further adjustment.
    
        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame based on specified thresholds.
        '''
        # Set default values for optional parameters
        if stderr_columns is None:
            stderr_columns = ['K562_lfcSE', 'HepG2_lfcSE', 'SKNSH_lfcSE']
        if data_project is None:
            data_project = ['UKBB', 'GTEX', 'CRE']

        # Standard error threshold filtering
        quality_filter = df[stderr_columns].max(axis = 1) < stderr_threshold 
        df = df[quality_filter].reset_index(drop=True)

        # Calculate means and standard deviations for activity columns
        means = df[activity_columns].mean().to_numpy()
        stds = df[activity_columns].std().to_numpy()

        # Calculate upper and lower cutoffs
        up_cut = means + stds * std_multiple_cut + up_cutoff_move
        down_cut = means - stds * std_multiple_cut 

        # Apply combined filtering for non-extreme values
        non_extremes_filter = ((df[activity_columns] < up_cut) & (df[activity_columns] > down_cut)).all(axis=1)
        df = df[non_extremes_filter].reset_index(drop=True)

        return df  

    def duplicate_high_activity_rows(self, df: pd.DataFrame, activity_columns: List[str]) -> pd.DataFrame:
        """
        Duplicates rows in the DataFrame where the maximum value across specified activity columns 
        exceeds a defined threshold.
    
        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing data to be filtered and duplicated.
        activity_columns : List[str]
            List of columns containing activity data used to determine rows for duplication.
    
        Returns:
        --------
        pd.DataFrame
            DataFrame with specified rows duplicated based on the maximum activity value.
        """
         # Calculate the maximum activity value across specified columns for each row
        max_values = df[activity_columns].max(axis=1)

        # Filter rows where the maximum activity value exceeds the duplication cutoff
        duplication_filter = max_values > self.duplication_cutoff
        duplicated_rows = df[duplication_filter].reset_index(drop=True)

        # Concatenate the duplicated rows with the original DataFrame
        df_with_duplicates = pd.concat([df, duplicated_rows], ignore_index=True)
        
        return df_with_duplicates
        
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

    def activity_columns_parse(self, activity_columns: str | List[str], 
                               default_activity_columns = ['K562', 'HepG2', 'SKNSH']) -> List[str]:
        '''
        Parses the input activity columns and returns a list of parsed activity_columns.
        '''
        if isinstance(activity_columns, str):
            activity_columns = [activity_columns]
        if isinstance(activity_columns, List):
            for i in range(len(activity_columns)):
                act = activity_columns[i]
                act = act.split('_')[0]
                if act not in default_activity_columns:
                    raise ValueError(f"Invalid activity column: {act}. Must be one of {default_activity_columns}.")
                activity_columns[i] = act + "_log2FC"
        return activity_columns