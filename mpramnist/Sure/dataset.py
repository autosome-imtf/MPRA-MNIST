import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict
import os
import bioframe as bf
from mpramnist.mpradataset import MpraDataset

class SureDataset(MpraDataset):
    """
    A dataset class for SuRE (Survey of Regulatory Elements) MPRA data.
    
    This class handles loading and processing of SuRE experimental data, which provides
    functional measurements of genetic variants across multiple cell types and genomes.
    The dataset supports both classification and regression tasks with genomic region filtering.

    This implementation and preprocessed data are adapted from:
    https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/SuRE.py

    Important Notes
    ---------------
    - All genomic coordinates are based on the hg19 reference genome assembly
    - Uses 0-based indexing for genomic coordinates
    - Supports data from four different genomes with variant-level measurements
    - Provides both binary classification and continuous regression targets

    Attributes
    ----------
    FLAG : str
        Constant identifier for the dataset type ("Sure").
    GENOME_IDS : list of str
        Available genome identifiers for SuRE data.
    TASKS : list of str
        Supported machine learning tasks.
    CELL_TYPES : list of str
        Available cell types with experimental measurements.

    Parameters
    ----------
    split : str
        Defines which split to use. Must be one of: 'train', 'val', 'test'.
    genome_id : str
        Identifier of the genome to use. Must be one of:
        - "SuRE42_HG02601"
        - "SuRE43_GM18983"
        - "SuRE44_HG01241" 
        - "SuRE45_HG03464"
        Specifies which individual's genomic variant data to load.
    task : str
        Type of machine learning task. Must be one of:
        - "classification": Binary classification with binned expression levels
        - "regression": Continuous regression with average expression values
    permute : bool, optional, default=True
        Whether to transpose one-hot encoded sequence matrices from 
        (sequence_length, 4) to (4, sequence_length) format.
        This converts from sequence-first to channels-first format for CNN compatibility.
    genomic_regions : str | List[Dict], optional
        Genomic regions to include or exclude. Can be specified as:
        - Path to BED file (str)
        - List of dictionaries with 'chrom', 'start', 'end' keys
        All coordinates must be in hg19 0-based format.
    exclude_regions : bool, optional, default=False
        If True, exclude the specified genomic regions instead of including them.
    transform : callable, optional
        Function to apply transformations to the input sequences.
    target_transform : callable, optional
        Function to apply transformations to the target labels.
    root : str, optional
        Root directory where data is stored. If None, uses default data path.

    Raises
    ------
    ValueError
        If invalid genome_id, task, or split values are provided.
    FileNotFoundError
        If the required data file cannot be found.

    Examples
    --------
    >>> # Load training data for classification from one genome
    >>> dataset = SureDataset(
    ...     split="train",
    ...     genome_id="SuRE42_HG02601", 
    ...     task="classification"
    ... )
    
    >>> # Load regression data with genomic region filtering
    >>> dataset = SureDataset(
    ...     split="test",
    ...     genome_id="SuRE43_GM18983",
    ...     task="regression",
    ...     genomic_regions="promoters.bed"
    ... )
    
    >>> # Load data excluding specific genomic regions
    >>> regions = [{"chrom": "chr1", "start": 1000000, "end": 2000000}]
    >>> dataset = SureDataset(
    ...     split="val",
    ...     genome_id="SuRE44_HG01241",
    ...     task="classification", 
    ...     genomic_regions=regions,
    ...     exclude_regions=True
    ... )

    Notes
    -----
    - Genomic coordinates follow hg19 reference genome and 0-based indexing
    - The dataset contains variant sequences with functional measurements in K562 and HepG2 cells
    - For classification: targets are binned into 5 categories per cell type
    - For regression: targets are continuous average expression values
    - Genomic region filtering uses bioframe for efficient interval operations
    """

    FLAG = "Sure"

    GENOME_IDS = [
        "SuRE42_HG02601",
        "SuRE43_GM18983",
        "SuRE44_HG01241",
        "SuRE45_HG03464",
    ]
    TASKS = ["classification", "regression"]
    CELL_TYPES = ["K562", "HepG2"]

    def __init__(
        self,
        split: str,
        genome_id: str,
        task: str,  # regression or classification. regression is default
        permute=True,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Attributes
        ----------
        split : str | List[str]
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        genome_id : str
            Identifier of the genome to use. Must be one of:
            - "SuRE42_HG02601"
            - "SuRE43_GM18983" 
            - "SuRE44_HG01241"
            - "SuRE45_HG03464"
            Specifies which genomic dataset to load.
        task : str
            Type of machine learning task. Must be one of:
            - "classification": for binary classification tasks
            - "regression": for continuous value prediction tasks
            Determines how target values are processed and interpreted.
        permute : bool, optional, default=True
            Whether to transpose one-hot encoded sequence matrices from 
            (sequence_length, 4) to (4, sequence_length) format.
            This is done for compatibility with certain model architectures
            that expect channels-first input format.
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool
            If True, exclude the specified regions instead of including them
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split=split, permute=permute, root=root)
        self.split = self.split_parse(split)
        self.task = task
        self.permute = permute
        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions
        self.genome_id = genome_id
        self._cell_type = genome_id
        if isinstance(genome_id, list):
            pass
        else:
            self.genome_id = [genome_id]
        self.ds = {}
        for genome in self.genome_id:
            if genome not in self.GENOME_IDS:
                raise ValueError(f"genome_id value must be one of {self.GENOME_IDS}")

            try:
                file_name = self.prefix + genome + ".tsv"
                self.download(self._data_path, file_name)
                file_path = os.path.join(self._data_path, file_name)
                df = pd.read_csv(file_path, sep="\t")
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")

            # Apply genomic region filtering
            df = self.filter_by_genomic_regions(df)
    
            if self.genomic_regions is None:
                df = df[df["split"].isin(self.split)].reset_index(drop=True)
            else:
                self.split = "genomic region"
            
            if self.task == "classification":
                self.output_names = ["K562_bin", "HepG2_bin"]
                self.num_classes_per_output = [5, 5]
                self.num_outputs = np.sum(self.num_classes_per_output)
                targets = df[self.output_names].to_numpy()

            elif self.task == "regression":
                self.output_names = ["avg_K562_exp", "avg_HepG2_exp"]
                self.num_outputs = 2
                targets = df[self.output_names].to_numpy()

            seq = df.sequence.to_numpy()

            if "targets" in self.ds:
                self.ds["targets"] = np.concatenate((self.ds["targets"], targets))

                self.ds["seq"] = np.concatenate((self.ds["seq"], seq))
            else:
                self.ds["targets"] = targets
                self.ds["seq"] = seq

            self.name_for_split_info = self.prefix + genome + "_"

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
        Parse and validate the input split parameter.

        Parameters
        ----------
        split : str
            Defines the data split. Must be one of: 'train', 'val', 'test'.

        Returns
        -------
        list of str
            A list containing the validated split identifier.

        Raises
        ------
        ValueError
            If split is not one of the allowed values.

        Examples
        --------
        >>> split_parse("train")
        ['train']
        >>> split_parse("test")  
        ['test']
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}

        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )

        return [split]
