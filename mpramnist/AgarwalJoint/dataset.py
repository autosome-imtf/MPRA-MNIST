import pandas as pd
from typing import List, Union, Dict, Optional
import os
import bioframe as bf
from mpramnist.mpradataset import MpraDataset


class AgarwalJointDataset(MpraDataset):
    """
    Dataset class for joint Agarwal MPRA (Massively Parallel Reporter Assay) data across multiple cell types.
    
    This class handles loading, filtering, and processing of genomic sequence data
    from the Agarwal et al. study, with support for multiple cell types simultaneously
    and genomic region-based filtering.

    The dataset uses human genome assembly hg38 with 0-based coordinate indexing.
    All genomic positions (start, end) follow 0-based indexing convention.

    Inherits from:
        MpraDataset: Base class for MPRA datasets

    Constants:
        CONSTANT_LEFT_FLANK (str): Constant left flanking sequence required for each sequence
        CONSTANT_RIGHT_FLANK (str): Constant right flanking sequence required for each sequence
        LEFT_FLANK (str): Left flanking sequence from human_legnet
        RIGHT_FLANK (str): Right flanking sequence from human_legnet
        CELL_TYPES (list): Available cell types: ['HepG2', 'K562', 'WTC11']
        FLAG (str): Dataset identifier flag: 'AgarwalJoint'

    Examples:
        >>> # Load training data for HepG2 cell type only
        >>> dataset = AgarwalJointDataset(split='train', cell_type='HepG2')
        >>> 
        >>> # Load data for multiple cell types
        >>> dataset = AgarwalJointDataset(
        ...     split='train',
        ...     cell_type=['HepG2', 'K562']
        ... )
        >>> 
        >>> # Load data filtered by genomic regions from BED file
        >>> dataset = AgarwalJointDataset(
        ...     split='train',
        ...     cell_type=['HepG2', 'K562', 'WTC11'],
        ...     genomic_regions='path/to/regions.bed'
        ... )
        >>> 
        >>> # Load data excluding specific genomic regions
        >>> regions = [{'chrom': 'chr1', 'start': 1000, 'end': 2000}]
        >>> dataset = AgarwalJointDataset(
        ...     split=[1, 2, 3],
        ...     cell_type='WTC11',
        ...     genomic_regions=regions,
        ...     exclude_regions=True
        ... )
    """

    CONSTANT_LEFT_FLANK = "AGGACCGGATCAACT"  # required for each sequence
    CONSTANT_RIGHT_FLANK = "CATTGCGTGAACCGA"  # required for each sequence
    LEFT_FLANK = "GGCCCGCTCTAGACCTGCAGG"  # from human_legnet
    RIGHT_FLANK = (
        "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"  # from human_legnet
    )

    CELL_TYPES = ["HepG2", "K562", "WTC11"]
    FLAG = "AgarwalJoint"

    def __init__(
        self,
        split: str | List[int] | int,
        cell_type: str | List[str],
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        root=None,
        transform=None,
        target_transform=None,
    ):
        """
        Initialize AgarwalJointDataset instance.

        Parameters
        ----------
        split : str | List[int] | int
            Defines which data split to use. Can be:
            - String: 'train', 'val', 'test' (uses predefined fold sets)
            - List[int]: List of specific fold numbers (1-10)
            - int: Single fold number (1-10)
        cell_type : str | List[str]
            Cell type(s) for filtering the data. Can be:
            - str: Single cell type ('HepG2', 'K562', or 'WTC11')
            - List[str]: Multiple cell types
        genomic_regions : Optional[Union[str, List[Dict]]], optional
            Genomic regions to include or exclude. Can be:
            - str: Path to BED file containing genomic regions (hg38, 0-based)
            - List[Dict]: List of dictionaries with 'chrom', 'start', 'end' keys (hg38, 0-based)
            - None: No genomic region filtering
        exclude_regions : bool, default=False
            If True, exclude the specified genomic regions instead of including them
        root : optional
            Root directory for data storage
        transform : callable, optional
            Transformation function applied to each sequence
        target_transform : callable, optional
            Transformation function applied to target values

        Raises
        ------
        ValueError
            - If cell_type is not in CELL_TYPES
            - If split string is not 'train', 'val', or 'test'
            - If fold numbers are not in range 1-10
        FileNotFoundError
            If the required joint data file is not found

        Notes
        -----
        - The dataset uses 10-fold cross-validation by default
        - Training folds: 1-8, Validation fold: 9, Test fold: 10
        - When genomic_regions is specified, the split parameter is ignored for filtering
          but the split information is stored as 'genomic region'
        - All genomic coordinates use hg38 assembly with 0-based indexing
        - For multiple cell types, the target will be a multi-column array
        """
        super().__init__(split, root)

        if isinstance(cell_type, str):
            if cell_type not in self.CELL_TYPES:
                raise ValueError(
                    f"Invalid cell_type: {cell_type}. Must be one of {self.CELL_TYPES}."
                )
            cell_type = [cell_type]
        if isinstance(cell_type, List):
            for i in range(len(cell_type)):
                act = cell_type[i]
                if act not in self.CELL_TYPES:
                    raise ValueError(
                        f"Invalid cell_type: {act}. Must be one of {self.CELL_TYPES}."
                    )
        self._cell_type = cell_type
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions

        self.prefix = self.FLAG + "_"

        try:
            file_name = self.prefix + "joint_data" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
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
        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on genomic regions using bioframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'chromosome': chromosome name (hg38)
            - 'start': start position (0-based, hg38)
            - 'end': end position (0-based, hg38)

        Returns
        -------
        pd.DataFrame
            Filtered dataframe containing only sequences that overlap (or don't overlap)
            with the specified genomic regions

        Notes
        -----
        - Uses bioframe library for genomic interval operations
        - Converts chromosome names to strings for compatibility
        - Handles both BED files and list of region dictionaries
        - All genomic coordinates use hg38 assembly with 0-based indexing
        - Input regions should be provided in hg38 coordinates with 0-based indexing
        - BED files typically use 0-based coordinates, which matches this implementation
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
        # Rename columns to match bioframe schema
        data_df = df.copy()
        data_df = data_df.rename(
            columns={"chromosome": "chrom", "start": "start", "end": "end"}
        )

        # Convert to integer if possible
        for col in ["start", "end"]:
            if col in data_df.columns:
                data_df[col] = pd.to_numeric(data_df[col], errors="coerce").astype(
                    "Int64"
                )

        # Find intersections
        intersections = bf.overlap(data_df, regions_df, how="inner", return_index=True)

        if self.exclude_regions:
            # Exclude sequences that overlap with specified regions
            filtered_df = df[~df.index.isin(intersections["index"])]
        else:
            # Include only sequences that overlap with specified regions
            filtered_df = df[df.index.isin(intersections["index"])]

        return filtered_df

    def split_parse(self, split: list[int] | int | str) -> list[int]:
        """
        Parse split parameter and return list of fold numbers.

        Parameters
        ----------
        split : list[int] | int | str
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
