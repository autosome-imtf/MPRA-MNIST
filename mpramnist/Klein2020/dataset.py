import pandas as pd
import os
from typing import List, Union, Dict, Optional
from collections import defaultdict
import pyfaidx
import subprocess
import bioframe as bf

from mpramnist.mpradataset import MpraDataset


class KleinDataset(MpraDataset):
    
    
    FLAG = "Klein2020"

    CELL_TYPE = "HepG2" # hepg2 only
    

    def __init__(
        self,
        experiment: str,
        split: str = "test_all",
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        
        # Initialize parent class
        super().__init__(split, root)

        self.transform = transform
        self.target_transform = target_transform
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions
        self.experiment = self.experiment_parse(experiment)
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"  # Prefix for file names
        
        try:
            # Load the data file
            file_name = self.prefix + "all_chr" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        if self.genomic_regions is None:
            self.ds = df[df.fold.isin(self.split)].reset_index(drop=True)
        else:
            self.ds = df
            self.split = "genomic region"

        # Prepare final dataset structure
        targets = self.ds[self.experiment].to_numpy()
        seq = self.ds.sequence.to_numpy()
        self.ds = {"targets": targets, "seq": seq}

        # Identifier for split information
        self.name_for_split_info = self.prefix

    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on genomic regions using bioframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'Chromosome': chromosome name (hg38)
            - 'Position': variant position (0-based, hg38)

        Returns
        -------
        pd.DataFrame
            Filtered dataframe containing only sequences that overlap (or don't overlap)
            with the specified genomic regions

        Notes
        -----
        - Uses bioframe library for genomic interval operations
        - All genomic coordinates use hg38 assembly with 0-based indexing
        - Sequences are defined as regions centered on variant positions
        - Input regions should be provided in hg38 coordinates with 0-based indexing
        """
        if self.genomic_regions is None:
            return df

        # Prepare the genomic regions for bioframe
        if isinstance(self.genomic_regions, str):
            # Load from BED file
            regions_df = bf.read_table(self.genomic_regions, schema="bed")
            regions_df["chrom"] = regions_df["chrom"].astype(str)
        else:
            # Convert list of dicts to DataFrame
            regions_df = pd.DataFrame(self.genomic_regions)

        # Prepare our data for bioframe intersection
        # Create start and end positions based on the mutation position and desired length
        data_df = df.copy()
        half_length = self.length // 2
        
        # Calculate start and end positions for each sequence
        data_df["start"] = data_df["Position"] - half_length
        data_df["end"] = data_df["Position"] + half_length
        data_df["chrom"] = data_df["Chromosome"]
        
        # Convert to integer if possible
        for col in ["start", "end"]:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce").astype("Int64")

        # Find intersections
        intersections = bf.overlap(data_df, regions_df, how="inner", return_index=True)
        
        if self.exclude_regions:
            # Exclude sequences that overlap with specified regions
            filtered_df = df[~df.index.isin(intersections["index"])]
        else:
            # Include only sequences that overlap with specified regions
            filtered_df = df[df.index.isin(intersections["index"])]

        return filtered_df

    def experiment_parse(self, exp: Union[str, List[int], int]) -> list[int]:
        """
        Parse experiment parameter and return list of columns to use.
        May be one or more from the list below
        """

        experiments_default = ['pGL4', 'HSS',
       'ORI', '5/3_WT',
       '5/3_MT', '5/5_WT',
       '5/5_MT', '3/3_WT',
       '3/3_MT', 'HSS_full',
       'ORI_full', 'HSS_b2',
       'ORI_b2']

        # Process string input
        if isinstance(exp, str):
            exp = [exp]

        # Check the range of values for a list
        if isinstance(exp, list):
            for i in range(len(exp)):
                e = exp[i].split("_log2")[0]
                if e not in experiments_default:
                    raise ValueError(
                        f"Invalid split value: {e}. Must be one of {experiments_default}."
                    )
                exp[i] = e + "_log2"

        return exp
        
    def split_parse(self, split: Union[str, List[int], int]) -> list[int]:
        """
        Parse split parameter and return list of fold numbers.

        Parameters
        ----------
        split : Union[str, List[int], int]
            Split specification to parse

        Returns
        -------
        list[int]
            List of fold numbers (1-10)

        Raises
        ------
        ValueError
            If split string is invalid or fold numbers are out of range

        Examples
        --------
        >>> split_parse('train')
        [1, 2, 3, 4, 5, 6, 7, 8]
        >>> split_parse([1, 2, 3])
        [1, 2, 3]
        >>> split_parse(5)
        [5]
        """

        split_default = {
            "train": [1, 2, 3, 4, 5, 6, 7, 8],
            "val": [9],
            "test": [10],
            "test_all": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }  # default split of data

        # Process string input
        if isinstance(split, str):
            if split not in split_default:
                raise ValueError(
                    f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
                )
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
