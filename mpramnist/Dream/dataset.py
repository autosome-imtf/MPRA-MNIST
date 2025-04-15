import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, T, Union
import torch
import os
import warnings

from mpramnist.mpradataset import MpraDataset

class DreamDataset(MpraDataset):
    """Dataset class for DREAM MPRA data.
    
    Handles loading and preprocessing of DREAM challenge MPRA datasets.
    Supports different dataset types and splits.
    """
    
    FLAG = "Dream"
    PLASMID = "aactctcaaggatcttaccgctgttgagatccagttcgatgtaacccactcgtgcacccaactgatcttcagcatcttttactttcaccagcgtttctgggtgagcaaaaacaggaaggcaaaatgccgcaaaaaagggaataagggcgacacggaaatgttgaatactcatactcttcctttttcaatattattgaagcatttatcagggttattgtctcatgagcggatacatatttgaatgtatttagaaaaataaacaaataggggttccgcgcacatttccccgaaaagtgccacctgacgtcatctatattaccctgttatccctagcggatctgccggtagaggtgtggtcaataagagcgacctcatactatacctgagaaagcaacctgacctacaggaaagagttactcaagaataagaattttcgttttaaaacctaagagtcactttaaaatttgtatacacttattttttttataacttatttaataataaaaatcataaatcataagaaattcgcttatttagaagtGGCGCGCCGGTCCGttacttgtacagctcgtccatgccgccggtggagtggcggccctcggcgcgttcgtactgttccacgatggtgtagtcctcgttgtgggaggtgatgtccaacttgatgttgacgttgtaggcgccgggcagctgcacgggcttcttggccttgtaggtggtcttgacctcagcgtcgtagtggccgccgtccttcagcttcagcctctgcttgatctcgcccttcagggcgccgtcctcggggtacatccgctcggaggaggcctcccagcccatggtcttcttctgcattacggggccgtcggaggggaagttggtgccgcgcagcttcaccttgtagatgaactcgccgtcctgcagggaggagtcctgggtcacggtcaccacgccgccgtcctcgaagttcatcacgcgctcccacttgaagccctcggggaaggacagcttcaagtagtcggggatgtcggcggggtgcttcacgtaggccttggagccgtacatgaactgaggggacaggatgtcccaggcgaagggcagggggccacccttggtcaccttcagcttggcggtctgggtgccctcgtaggggcggccctcgccctcgccctcgatctcgaactcgtggccgttcacggagccctccatgtgcaccttgaagcgcatgaactccttgatgatggccatgttatcctcctcgcccttgctcacCATGGTACTAGTGTTTAGTTAATTATAGTTCGTTGACCGTATATTCTAAAAACAAGTACTCCTTAAAAAAAAACCTTGAAGGGAATAAACAAGTAGAATAGATAGAGAGAAAAATAGAAAATGCAAGAGAATTTATATATTAGAAAGAGAGAAAGAAAAATGGAAAAAAAAAAATAGGAAAAGCCAGAAATAGCACTAGAAGGAGCGACACCAGAAAAGAAGGTGATGGAACCAATTTAGCTATATATAGTTAACTACCGGCTCGATCATCTCTGCCTCCAGCATAGTCGAAGAAGAATTTTTTTTTTCTTGAGGCTTCTGTCAGCAACTCGTATTTTTTCTTTCTTTTTTGGTGAGCCTAAAAAGTTCCCACGTTCTCTTGTACGACGCCGTCACAAACAACCTTATGGGTAATTTGTCGCGGTCTGGGTGTATAAATGTGTGGGTGCAACATGAATGTACGGAGGTAGTTTGCTGATTGGCGGTCTATAGATACCTTGGTTATGGCGCCCTCACAGCCGGCAGGGGAAGCGCCTACGCTTGACATCTACTATATGTAAGTATACGGCCCCATATATAggccctttcgtctcgcgcgtttcggtgatgacggtgaaaacctctgacacatgcagctcccggagacggtcacagcttgtctgtaagcggatgccgggagcagacaagcccgtcagggcgcgtcagcgggtgttggcgggtgtcggggctggcttaactatgcggcatcagagcagattgtactgagagtgcaccatatggacatattgtcgttagaacgcggctacaattaatacataaccttatgtatcatacacatacgatttaggtgacactatagaacgcggccgccagctgaagctttaactatgcggcatcagagcagattgtactgagagtgcaccataccaccttttcaattcatcattttttttttattcttttttttgatttcggtttccttgaaatttttttgattcggtaatctccgaacagaaggaagaacgaaggaaggagcacagacttagattggtatatatacgcatatgtagtgttgaagaaacatgaaattgcccagtattcttaacccaactgcacagaacaaaaacctgcaggaaacgaagataaatcatgtcgaaagctacatataaggaacgtgctgctactcatcctagtcctgttgctgccaagctatttaatatcatgcacgaaaagcaaacaaacttgtgtgcttcattggatgttcgtaccaccaaggaattactggagttagttgaagcattaggtcccaaaatttgtttactaaaaacacatgtggatatcttgactgatttttccatggagggcacagttaagccgctaaaggcattatccgccaagtacaattttttactcttcgaagacagaaaatttgctgacattggtaatacagtcaaattgcagtactctgcgggtgtatacagaatagcagaatgggcagacattacgaatgcacacggtgtggtgggcccaggtattgttagcggtttgaagcaggcggcagaagaagtaacaaaggaacctagaggccttttgatgttagcagaattgtcatgcaagggctccctatctactggagaatatactaagggtactgttgacattgcgaagagcgacaaagattttgttatcggctttattgctcaaagagacatgggtggaagagatgaaggttacgattggttgattatgacacccggtgtgggtttagatgacaagggagacgcattgggtcaacagtatagaaccgtggatgatgtggtctctacaggatctgacattattattgttggaagaggactatttgcaaagggaagggatgctaaggtagagggtgaacgttacagaaaagcaggctgggaagcatatttgagaagatgcggccagcaaaactaaaaaactgtattataagtaaatgcatgtatactaaactcacaaattagagcttcaatttaattatatcagttattaccctatgcggtgtgaaataccgcacagatgcgtaaggagaaaataccgcatcaggaaattgtaagcgttaatattttgttaaaattcgcgttaaatttttgttaaatcagctcattttttaaccaataggccgaaatcggcaaaatcccttataaatcaaaagaatagaccgagatagggttgagtgttgttccagtttggaacaagagtccactattaaagaacgtggactccaacgtcaaagggcgaaaaaccgtctatcagggcgatggcccactacgtgaaccatcaccctaatcaagtGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGAAGGCAAAGatgtctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgctacttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctagatacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagaccagagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaactataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaattagctgaccattatcaacaaaatactccaattggtgatggtccagtcttgttaccagacaaccattacttatcctatcaatctgccttatccaaagatccaaacgaaaagagagaccacatggtcttgttagaatttgttactgctgctggtattacccatggtatggatgaattgtacaaataaggcgcgccacttctaaataagcgaatttcttatgatttatgatttttattattaaataagttataaaaaaaataagtgtatacaaattttaaagtgactcttaggttttaaaacgaaaattcttattcttgagtaactctttcctgtaggtcaggttgctttctcaggtatagtatgaggtcgctcttattgaccacacctctaccggcagatccgctagggataacagggtaatataGATCTGTTTAGCTTGCCTCGTCCCCGCCGGGTCACCCGGCCAGCGACATGGAGGCCCAGAATACCCTCCTTGACAGTCTTGACGTGCGCAGCTCAGGGGCATGATGTGACTGTCGCCCGTACATTTAGCCCATACATCCCCATGTATAATCATTTGCATCCATACATTTTGATGGCCGCACGGCGCGAAGCAAAAATTACGGCTCCTCGCTGCAGACCTGCGAGCAGGGAAACGCTCCCCTCACAGACGCGTTGAATTGTCCCCACGCCGCGCCCCTGTAGAGAAATATAAAAGGTTAGGATTTGCCACTGAGGTTCTTCTTTCATATACTTCCTTTTAAAATCTTGCTAGGATACAGTTCTCACATCACATCCGAACATAAACAACCATGGGTACCACTCTTGACGACACGGCTTACCGGTACCGCACCAGTGTCCCGGGGGACGCCGAGGCCATCGAGGCACTGGATGGGTCCTTCACCACCGACACCGTCTTCCGCGTCACCGCCACCGGGGACGGCTTCACCCTGCGGGAGGTGCCGGTGGACCCGCCCCTGACCAAGGTGTTCCCCGACGACGAATCGGACGACGAATCGGACGACGGGGAGGACGGCGACCCGGACTCCCGGACGTTCGTCGCGTACGGGGACGACGGCGACCTGGCGGGCTTCGTGGTCGTCTCGTACTCCGGCTGGAACCGCCGGCTGACCGTCGAGGACATCGAGGTCGCCCCGGAGCACCGGGGGCACGGGGTCGGGCGCGCGTTGATGGGGCTCGCGACGGAGTTCGCCCGCGAGCGGGGCGCCGGGCACCTCTGGCTGGAGGTCACCAACGTCAACGCACCGGCGATCCACGCGTACCGGCGGATGGGGTTCACCCTCTGCGGCCTGGACACCGCCCTGTACGACGGCACCGCCTCGGACGGCGAGCAGGCGCTCTACATGAGCATGCCCTGCCCCTAATCAGTACTGACAATAAAAAGATTCTTGTTTTCAAGAACTTGTCATTTGTATAGTTTTTTTATATTGTAGTTGTTCTATTTTAATCAAATGTTAGCGTGATTTATATTTTTTTTCGCCTCGACATCATCTGCCCAGATGCGAAGTTAAGTGCGCAGAAAGTAATATCATGCGTCAATCGTATGTGAATGCTGGTCGCTATACTGCTGTCGATTCGATACTAACGCCGCCATCCAGTGTCGAAAACGAGCTCGaattcctgggtccttttcatcacgtgctataaaaataattataatttaaattttttaatataaatatataaattaaaaatagaaagtaaaaaaagaaattaaagaaaaaatagtttttgttttccgaagatgtaaaagactctagggggatcgccaacaaatactaccttttatcttgctcttcctgctctcaggtattaatgccgaattgtttcatcttgtctgtgtagaagaccacacacgaaaatcctgtgattttacattttacttatcgttaatcgaatgtatatctatttaatctgcttttcttgtctaataaatatatatgtaaagtacgctttttgttgaaattttttaaacctttgtttatttttttttcttcattccgtaactcttctaccttctttatttactttctaaaatccaaatacaaaacataaaaataaataaacacagagtaaattcccaaattattccatcattaaaagatacgaggcgcgtgtaagttacaggcaagcgatccgtccGATATCatcagatccactagtggcctatgcggccgcggatctgccggtctccctatagtgagtcgtattaatttcgataagccaggttaacctgcattaatgaatcggccaacgcgcggggagaggcggtttgcgtattgggcgctcttccgcttcctcgctcactgactcgctgcgctcggtcgttcggctgcggcgagcggtatcagctcactcaaaggcggtaatacggttatccacagaatcaggggataacgcaggaaagaacatgtgagcaaaaggccagcaaaaggccaggaaccgtaaaaaggccgcgttgctggcgtttttccataggctccgcccccctgacgagcatcacaaaaatcgacgctcaagtcagaggtggcgaaacccgacaggactataaagataccaggcgtttccccctggaagctccctcgtgcgctctcctgttccgaccctgccgcttaccggatacctgtccgcctttctcccttcgggaagcgtggcgctttctcaTAgctcacgctgtaggtatctcagttcggtgtaggtcgttcgctccaagctgggctgtgtgcacgaaccccccgttcagcccgaccgctgcgccttatccggtaactatcgtcttgagtccaacccggtaagacacgacttatcgccactggcagcagccactggtaacaggattagcagagcgaggtatgtaggcggtgctacagagttcttgaagtggtggcctaactacggctacactagaagAacagtatttggtatctgcgctctgctgaagccagttaccttcggaaaaagagttggtagctcttgatccggcaaacaaaccaccgctggtagcggtggtttttttgtttgcaagcagcagattacgcgcagaaaaaaaggatctcaagaagatcctttgatcttttctacggggtctgacgctcagtggaacgaaaactcacgttaagggattttggtcatgagattatcaaaaaggatcttcacctagatccttttaaattaaaaatgaagttttaaatcaatctaaagtatatatgagtaaacttggtctgacagttaccaatgcttaatcagtgaggcacctatctcagcgatctgtctatttcgttcatccatagttgcctgactccccgtcgtgtagataactacgatacgggagggcttaccatctggccccagtgctgcaatgataccgcgagacccacgTtcaccggctccagatttatcagcaataaaccagccagccggaagggccgagcgcagaagtggtcctgcaactttatccgcctccatccagtctattaattgttgccgggaagctagagtaagtagttcgccagttaatagtttgcgcaacgttgttgccattgctacaggcatcgtggtgtcacgctcgtcgtttggtatggcttcattcagctccggttcccaacgatcaaggcgagttacatgatcccccatgttgtgcaaaaaagcggttagctccttcggtcctccgatcgttgtcagaagtaagttggccgcagtgttatcactcatggttatggcagcactgcataattctcttactgtcatgccatccgtaagatgcttttctgtgactggtgagtactcaaccaagtcattctgagaatagtgtatgcggcgaccgagttgctcttgcccggcgtcaatacgggataataccgcgccacatagcagaactttaaaagtgctcatcattggaaaacgttcttcggggcgaa"

    LEFT_FLANK = "TGCATTTTTTTCACATC" 
    RIGHT_FLANK = "GGTTACGGCTGTT"
    # Supported dataset types
    TYPES = [
        "high", "low", "yeast", "challenging", "random", "all",  # single types
        "snv", "perturbation", "tiling"  # paired types
    ]
    
    def __init__(self,
                 split: str,
                 data_type: str | List[str] = None,
                 transform = None,
                 target_transform = None,
                 root = None
                ):
        """
        Attributes
        ----------
        split : str 
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        data_type : str
            Dataset type (high, low, yeast, challenging, random, all, snv, perturbation, tiling)
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)

        # Initialize transformations
        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        
        # Parse and validate inputs
        self.split = self.split_parse(split)
        self.target_column = "label"

        # Load and prepare dataset
        self.dataset, self.data_type = self._define_dataset(self.split, data_type)
        self.df = self._load_and_prepare_data(self.dataset)
        
        # Prepare data structure based on split type
        self._prepare_data_structure()
            
    def _load_and_prepare_data(self, dataset: str) -> pd.DataFrame:
        """Load data from file and prepare the dataset."""
        try:
            file_name = self.prefix + dataset + '.tsv'
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
            
        return self._filter_dataset(df, self.split, self.data_type)

    def _prepare_data_structure(self):
        """Prepare the data structure based on the split type."""
        if self.dataset in ["train", "single"]:
            self.ds = {
                "targets": self.df[self.target_column].to_numpy(),
                "seq": self.df.seq.to_numpy()
            }
        elif self.dataset == "paired":
            self.ds = {
                "targets": self.df[self.target_column].to_numpy(),
                "seq": self.df.seq.to_numpy(),
                "seq_alt": self.df.seq_alt.to_numpy()
            }
            
    def _filter_dataset(
        self,
        df: pd.DataFrame,
        split: str,
        dataset_type: str | List[str]
    ) -> pd.DataFrame:
        """Filter the dataset based on split and type."""
        if split == "train":
            return df
        else:
            public_or_private = df[df[split] == 1]
            if isinstance(dataset_type, str):
                dataset_type = [dataset_type]
            selected_rows = public_or_private[
                (public_or_private[dataset_type] == 1).any(axis=1)
            ]
            return selected_rows
            
    def _define_dataset(
        self,
        split: str,
        dataset_type: str | List[str]) -> tuple[str, str | List[str]]:
        """Determine which dataset to load based on type and split.
        
        Returns:
            A tuple of (dataset_name, processed_type)
        """
        single_types = ["high", "low", "yeast", "challenging", "random", "all"]
        paired_types = ["snv", "perturbation", "tiling"]
        
        if split == "train":
            if dataset_type is not None:
                warnings.warn(
                    "WARNING! The training set was selected, "
                    "\nso the 'type' parameter is ignored.",
                    stacklevel=1
                )
            return "train", None
            
        if isinstance(dataset_type, str):
            return self._handle_single_type(dataset_type, single_types, paired_types)
        elif isinstance(dataset_type, List):
            return self._handle_multiple_types(dataset_type, single_types, paired_types)
        else:
            raise ValueError(f"Invalid type: {dataset_type}")

    def _handle_single_type(
        self,
        dataset_type: str,
        single_types: list,
        paired_types: list) -> tuple[str, str]:
        """Handle single type specification."""
        lower_type = dataset_type.lower()
        if lower_type in single_types:
            return "single", lower_type
        elif lower_type in paired_types:
            return "paired", lower_type
        raise ValueError(
            f"Invalid type: {dataset_type}. "
            f"Expected one of: {self.types}"
        )

    def _handle_multiple_types(
        self,
        dataset_types: List[str],
        single_types: list,
        paired_types: list) -> tuple[str, List[str]]:
        """Handle multiple types specification."""
        lower_types = [t.lower() for t in dataset_types]
        
        if all(t in single_types for t in lower_types):
            return "single", lower_types
        elif all(t in paired_types for t in lower_types):
            return "paired", lower_types
        raise ValueError(
            f"Invalid types: {dataset_types}. "
            f"All types must be from {single_types} or {paired_types}"
        )

    def split_parse(self, split: str) -> str:
        """Parse and validate the split parameter.
        
        Args:
            split: Input split specification
            
        Returns:
            Validated split string
            
        Raises:
            ValueError: If split is invalid
        """
        valid_splits = {
            "train": "train",
            "val": "public",
            "public": "public",
            "test": "private",
            "private": "private"
        }
        
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: {split}. "
                "Expected 'train', 'val'/'public', or 'test'/'private'."
            )
    
        return valid_splits[split]