import pandas as pd
from typing import List, Union, Optional, Dict
import os
import bioframe as bf
import warnings

from mpramnist.mpradataset import MpraDataset


class DeepStarrDataset(MpraDataset):
    """
    Dataset class for DeepSTARR MPRA data from Drosophila S2 cells.

    This class handles loading and preprocessing of MPRA data from the DeepSTARR study,
    which measures transcriptional activity of regulatory sequences in Drosophila
    melanogaster S2 cells. The dataset includes measurements for both developmental
    and housekeeping gene regulatory activities.

    The dataset supports chromosome-based splits, genomic region filtering, and
    reverse complement augmentation as implemented in the original study.

    The dataset uses Drosophila melanogaster genome assembly BDGP R5/dm3 with 0-based coordinate indexing.
    All genomic positions (start, end) follow 0-based indexing convention.

    Attributes
    ----------
    FLAG : str
        Identifier flag for DeepSTARR datasets ("DeepStarr")
    ACTIVITY_COLUMNS : List[str]
        Available activity measurement columns: ["Dev_log2", "Hk_log2"]
    LIST_OF_CHR : List[str]
        Valid chromosome names for chromosome-based splitting:
        ["2L", "2LHet", "2RHet", "3L", "3LHet", "3R", "3RHet", "4", "X", "XHet", "YHet", "2R"]

    Parameters
    ----------
    split : str | List[str]
        Data split specification. Can be:
        - Standard splits: 'train', 'val', 'test'
        - Chromosome names: any from LIST_OF_CHR
        - List of chromosome names for custom splits
    cell_type : str | List[str], default: ["Dev_log2", "Hk_log2"]
        Cell type(s) for target data. Can be:
        - "Dev_log2": Developmental activity
        - "Hk_log2": Housekeeping activity
        - List containing both for multi-task learning
    use_original_reverse_complement : bool | None, optional
        Whether to apply reverse complement augmentation as in original study.
        If None, automatically set to True for training split and False otherwise.
    genomic_regions : str | List[Dict], optional
        Genomic regions to include or exclude. Can be:
        - Path to BED file
        - List of dictionaries with 'chrom', 'start', 'end' keys
    exclude_regions : bool, default: False
        If True, exclude the specified genomic regions instead of including them
    transform : callable, optional
        Transformation function applied to each sequence.
    target_transform : callable, optional
        Transformation function applied to target values.
    root : str, optional
        Root directory for data storage.

    Raises
    ------
    FileNotFoundError
        If the required data file cannot be found or downloaded
    ValueError
        If provided split, cell_type, or chromosome parameters are invalid

    Examples
    --------
    >>> # Load training data with reverse complement augmentation
    >>> train_data = DeepStarrDataset(split="train", cell_type="Dev_log2")
    >>>
    >>> # Load validation data from specific chromosome
    >>> val_data = DeepStarrDataset(split="2L", cell_type="Hk_log2")
    >>>
    >>> # Load multi-task data with both activities
    >>> multi_data = DeepStarrDataset(split="test", cell_type=["Dev_log2", "Hk_log2"])
    >>>
    >>> # Load data filtered by genomic regions
    >>> region_data = DeepStarrDataset(
    ...     split="train",
    ...     genomic_regions="path/to/regions.bed",
    ...     exclude_regions=True
    ... )
    >>>
    >>> # Load custom chromosome split
    >>> custom_data = DeepStarrDataset(split=["2L", "2R", "3L"], cell_type="Dev_log2")

    Notes
    -----
    - Cell type: Drosophila melanogaster S2 cells
    - Activity measurements: log2-transformed reporter activity
    - Reverse complement augmentation: 
        * Training set is pre-augmented in original study
        * Applying manually may cause data leakage
    - Chromosome-based splits: Useful for cross-chromosome validation
    - Genomic region filtering: Uses bioframe for efficient genomic operations

    """

    FLAG = "DeepStarr"

    ACTIVITY_COLUMNS = ["Dev_log2", "Hk_log2"]
    LIST_OF_CHR = [
        "2L",
        "2LHet",
        "2RHet",
        "3L",
        "3LHet",
        "3R",
        "3RHet",
        "4",
        "X",
        "XHet",
        "YHet",
        "2R",
    ]
    

    def __init__(
        self,
        split: str | List[str],
        cell_type: str | List[str] = ["Dev_log2", "Hk_log2"],
        use_original_reverse_complement: bool | None = None,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize DeepStarrDataset for Drosophila regulatory sequence analysis.

        Supports multiple split strategies including standard train/val/test splits
        and chromosome-based splits for genomic cross-validation.

        Parameters
        ----------
        split : str | List[str]
            Data split specification. Can be:
            - Standard splits: 'train', 'val', 'test'
            - Chromosome names: any from LIST_OF_CHR
            - List of chromosome names for custom splits
        cell_type : str | List[str], default: ["Dev_log2", "Hk_log2"]
            Cell type(s) for target data. Can be:
            - "Dev_log2": Developmental activity
            - "Hk_log2": Housekeeping activity
            - List containing both for multi-task learning
        use_original_reverse_complement : bool | None, optional
            Whether to apply reverse complement augmentation as in original study.
            If None, automatically set to True for training split and False otherwise.
        genomic_regions : str | List[Dict], optional
            Genomic regions to include or exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool, default: False
            If True, exclude the specified genomic regions instead of including them
        transform : callable, optional
            Transformation function applied to each sequence.
        target_transform : callable, optional
            Transformation function applied to target values.
        root : str, optional
            Root directory for data storage.
        """
        super().__init__(split, root)
        self.cell_type = ["Drosophila S2 Developmental", 
                          "Drosophila S2 Housekeeping"] # for parent class compatibility

        self.activity_column = cell_type
        if use_original_reverse_complement is None:
            if isinstance(split, list) or split != "train":
                use_original_reverse_complement = False
            else:
                use_original_reverse_complement = True

        self.transform = transform
        self.target_transform = target_transform
        self.split, column = self.split_parse(split)
        self.prefix = self.FLAG + "_"
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions

        try:
            file_name = self.prefix + "all_chr" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Apply genomic region filtering
        df = self.filter_by_genomic_regions(df)

        if self.genomic_regions is None:
            df = df[df[column].isin(self.split)].reset_index(drop=True)
        else:
            self.split = "genomic region"

        if use_original_reverse_complement:
            """
            WARNING: 
            This function uses original paper's parameters so:
            > The training dataset (`split=train`) is pre-augmented with reverse complements
            > - 2× sequences (original + RC)  
            > - Identical labels for RC pairs  
            > Manual reverse complementing will cause data leakage!
            """

            if self.split == ["train"]:
                warnings.warn(
                    "WARNING! "
                    "\nNote: The training set contains reverse-complement augmentation as implemented in the original study.  "
                    "\n• Dataset size: 2N (N original + N reverse-complemented sequences)  "
                    "\n• Label consistency: y_rc ≡ y_original  "
                    "\n• Do not reapply this transformation during preprocessing. ",
                    stacklevel=1,
                )

            # reverse_complement
            rev_aug = df.copy()
            rev_aug.sequence = rev_aug.sequence.apply(self.reverse_complement)
            df = pd.concat([df, rev_aug], ignore_index=True)

        targets = df[self.activity_column].to_numpy()
        seq = df.sequence.to_numpy()
        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

    def reverse_complement(self, seq: str, mapping=None) -> str:
        """
        Generate reverse complement of a DNA sequence.

        Parameters
        ----------
        seq : str
            Input DNA sequence
        mapping : dict, optional
            Custom base mapping dictionary. Defaults to standard DNA complement:
            {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

        Returns
        -------
        str
            Reverse complement of the input sequence

        Raises
        ------
        ValueError
            If sequence contains invalid characters not in mapping

        Examples
        --------
        >>> dataset.reverse_complement("ATCG")
        'CGAT'
        >>> dataset.reverse_complement("ATCG", mapping={'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'})
        'CGAT'
        """
        if mapping is None:
            mapping = {"A": "T", "G": "C", "T": "A", "C": "G", "N": "N"}

        try:
            return "".join(mapping[base] for base in reversed(seq.upper()))
        except KeyError as e:
            raise ValueError(f"Invalid character in sequence: {e}")

    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame based on specified genomic regions.

        Uses bioframe for efficient genomic interval operations. Can either include
        or exclude sequences overlapping with the specified regions.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'chromosome': chromosome name (BDGP R5/dm3)
            - 'start': start position (0-based, BDGP R5/dm3)
            - 'end': end position (0-based, BDGP R5/dm3)

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only sequences that overlap (or don't overlap)
            with the specified genomic regions

        Notes
        -----
        - Requires bioframe package for genomic operations
        - Input DataFrame must have 'chr', 'start', 'end' columns
        - Converts chromosome names to strings for compatibility
        - Handles both BED files and list of region dictionaries
        - All genomic coordinates use BDGP R5/dm3 assembly with 0-based indexing
        - Input regions should be provided in BDGP R5/dm3 coordinates with 0-based indexing


        See Also
        --------
        bioframe.overlap : Underlying genomic interval overlap function
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
            columns={"chr": "chrom", "start": "start", "end": "end"}
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

    def split_parse(self, split: str) -> str:
        """
        Parse and validate split parameter.

        Converts various split specifications into standardized format and
        determines the appropriate column for filtering.

        Parameters
        ----------
        split : str | List[str]
            Split specification. Can be:
            - Standard split: 'train', 'val', 'test'
            - Chromosome name: any from LIST_OF_CHR
            - List of chromosome names

        Returns
        -------
        tuple[List[str], str]
            Tuple containing:
            - parsed_split: List of split values
            - column: Name of column to use for filtering ('split' or 'chr')

        Raises
        ------
        ValueError
            If split contains invalid values or mixed types

        Examples
        --------
        >>> dataset.split_parse('train')
        (['train'], 'split')
        >>> dataset.split_parse('2L')
        (['2L'], 'chr')
        >>> dataset.split_parse(['2L', '2R'])
        (['2L', '2R'], 'chr')
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}
        column = ""  # which column to consider

        # Process string input
        if isinstance(split, str):
            if split in valid_splits:
                column = "split"
                return [split], column
            elif split in self.LIST_OF_CHR:
                column = "chr"
                return [split], column
            else:
                raise ValueError(
                    f"Invalid fold value: {split}. Must be one of {self.LIST_OF_CHR}."
                )
        # Validate list of folds
        elif isinstance(split, list):
            result = []
            column = "chr"
            for item in split:
                if item in self.LIST_OF_CHR:
                    result.append(item)
                else:
                    raise ValueError(
                        f"Invalid fold value: {item}. Must be one of {self.LIST_OF_CHR}."
                    )
            return result, column
        else:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', 'test' or {self.LIST_OF_CHR}."
            )
