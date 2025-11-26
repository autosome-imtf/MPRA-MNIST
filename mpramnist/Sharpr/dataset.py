import pandas as pd
from typing import List, Union, Optional, Dict
import os
import bioframe as bf

from mpramnist.mpradataset import MpraDataset


class SharprDataset(MpraDataset):
    """
    Dataset class for Sharpr MPRA (Massively Parallel Reporter Assay) data.
    
    This class extends MpraDataset to handle Sharpr MPRA data with
    multiple cell types and promoters. It provides access to sequence data
    and corresponding activity measurements from K562 and HepG2 cell lines
    with minP and SV40 promoters.

    This implementation and preprocessed data are adapted from:
    https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/Sharpr_MPRA.py

    Important Notes
    ---------------
    - All genomic coordinates are based on the hg19 reference genome assembly
    - Uses 0-based indexing for genomic coordinates
    
    Attributes
    ----------
    CELL_TYPES : list of str
        Predefined list of all available activity measurement columns in the dataset.
        Includes replicate measurements and averages for:
        - K562 cell line with minP promoter (rep1, rep2, avg)
        - K562 cell line with SV40 promoter (rep1, rep2, avg) 
        - HepG2 cell line with minP promoter (rep1, rep2, avg)
        - HepG2 cell line with SV40 promoter (rep1, rep2, avg)
    FLAG : str
        Dataset identifier flag, set to "Sharpr".
    
    Parameters
    ----------
    split : str
        Defines which data split to use. Expected values: 'train', 'val', 'test'.
    cell_type : List[str]
        List of column names with activity data to be used as targets.
        Must be a subset of CELL_TYPES.
    transform : callable, optional
        Transformation function applied to each sequence sample.
        If provided, will be applied to sequences during data loading.
    genomic_regions : str | List[Dict], optional
        Genomic regions to include or exclude. Can be specified as:
        - Path to BED file (str)
        - List of dictionaries with 'chrom', 'start', 'end' keys
        All coordinates must be in hg19 0-based format.
    exclude_regions : bool, optional, default=False
        If True, exclude the specified genomic regions instead of including them.
    target_transform : callable, optional
        Transformation function applied to target activity values.
        If provided, will be applied to targets during data loading.
    root : str, optional
        Root directory where dataset is stored. If None, uses default location.
    
    Raises
    ------
    ValueError
        If provided split value is not 'train', 'val', or 'test'.
    FileNotFoundError
        If the requested data file cannot be found in the dataset directory.
    
    Examples
    --------
    >>> dataset = SharprDataset(
    ...     split='train',
    ...     cell_type=['k562_minp_avg', 'hepg2_minp_avg']
    ... )
    >>> len(dataset)
    15000
    >>> sequence, target = dataset[0]
    >>> print(sequence.shape, target.shape)
    (1000,) (2,)

    >>> # Load data excluding specific genomic regions
    >>> regions = [{"chrom": "chr1", "start": 1000000, "end": 2000000}]
    >>> dataset = SharprDataset(
    ...     split="val",
    ...     cell_type=['k562_minp_avg', 'hepg2_minp_avg']
    ...     genomic_regions=regions,
    ...     exclude_regions=True
    ... )
    
    Notes
    -----
    - The dataset automatically downloads data files if not present
    - Activity measurements are typically log-transformed values
    - Multiple activity columns can be used for multi-label learning
    - Genomic coordinates follow hg19 reference genome and 0-based indexing
    - Genomic region filtering uses bioframe for efficient interval operations
    """

    CELL_TYPES = [
        "k562_minp_rep1", "k562_minp_rep2", "k562_minp_avg",
        "k562_sv40p_rep1", "k562_sv40p_rep2", "k562_sv40p_avg",
        "hepg2_minp_rep1", "hepg2_minp_rep2", "hepg2_minp_avg",
        "hepg2_sv40p_rep1", "hepg2_sv40p_rep2", "hepg2_sv40p_avg",
    ]
    FLAG = "Sharpr"

    def __init__(
        self,
        split: str,
        cell_type: List[str] = None,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize SharprDataset instance.

        Attributes
        ----------
        split : str
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        cell_type : List[str], default None
            Cell types or List of column names with activity data to be used as targets.
            Must be a subset of CELL_TYPES.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)

        self.transform = transform
        self.target_transform = target_transform
        self.cell_type = cell_type
        self.split = self.split_parse(split)
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions
        self.prefix = self.FLAG + "_"

        if self.cell_type is None:
            self.cell_type = self.CELL_TYPES
            
        try:
            file_name = self.prefix + "all_chr"+ ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t", low_memory = False)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Apply genomic region filtering
        full_df = self.filter_by_genomic_regions(df)

        # Filter sequence duplicates between splits
        if self.split[0] in ["val", "test"]:
            # For val/test splits: remove sequences that appear in train split
            train_sequences = set(full_df[full_df["split"] == "train"]["seq"])
            full_df = full_df[full_df["split"].isin(self.split)]
            
            # Remove sequences from val/test that are present in train
            mask = ~full_df["seq"].isin(train_sequences)
            df = full_df[mask].reset_index(drop=True)
            
            print(f"{self.split[0]}: After filtering duplicates: {len(full_df)} sequences in {self.split[0]}")
        else:
            # For train split: simply select train split as before
            df = full_df[full_df["split"].isin(self.split)].reset_index(drop=True)

        targets = df[self.cell_type].to_numpy()
        seq = df.seq.to_numpy()
        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on genomic regions using bioframe.

        This method applies genomic region filtering to the dataset using either
        inclusion or exclusion logic. Regions are specified in hg19 0-based coordinates.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic variants with columns:
            - 'chr': chromosome name
            - 'start': start position (0-based, hg19)
            - 'end': end position (0-based, hg19)
            - Other variant metadata and measurements

        Returns
        -------
        pd.DataFrame
            Filtered dataframe containing only variants that pass the region criteria.

        Notes
        -----
        - Uses bioframe for efficient genomic interval operations
        - Input regions must be in hg19 0-based coordinate system
        - Handles both BED files and list of region dictionaries
        - Automatically converts coordinate columns to appropriate data types
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


    def split_parse(self, split: str) -> str:
        """
        Parse and validate the split parameter.
        
        Parameters
        ----------
        split : str
            Data split identifier. Must be one of: 'train', 'val', 'test'.
        
        Returns
        -------
        str
            Validated split string.
        
        Raises
        ------
        ValueError
            If split is not one of the allowed values.
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}

        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )

        return [split]
