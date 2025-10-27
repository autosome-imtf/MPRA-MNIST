import pandas as pd
from typing import Literal
import os
import warnings

from mpramnist.mpradataset import MpraDataset


class VaishnavDataset(MpraDataset):
    """
    Dataset class for Vaishnav et al. MPRA data with environmental context support.

    This class handles loading and preprocessing of MPRA data from Vaishnav et al. study,
    which investigates transcriptional regulation across different environmental contexts
    and sequence conditions. The dataset supports multiple experimental environments
    and test scenarios for comprehensive model evaluation.

    The data includes measurements from yeast strains (Y8205, S288C::ura3, etc.)
    under different environmental conditions, enabling study of context-dependent
    regulatory effects.

    Attributes
    ----------
    FLAG : str
        Identifier flag for Vaishnav datasets ("Vaishnav")
    PLASMID : str
        Constant plasmid backbone sequence used in the MPRA constructs
    LEFT_FLANK : str
        Left flanking sequence used for sequence extraction and alignment
    RIGHT_FLANK : str
        Right flanking sequence used for sequence extraction and alignment

    Parameters
    ----------
    split : Literal["train", "val", "test"]
        Data split specification:
        - "train": Training data
        - "val": Validation data
        - "test": Test data
    dataset_env_type : Literal["defined", "complex"]
        Environmental context type:
        - "defined": Synthetic defined medium lacking uracil (SD-Ura)
        - "complex": yeast extract, peptone and dextrose (YPD)
    test_dataset_type : Literal["drift", "native", "paired"], optional
        Required for test split only. Specifies test scenario:
        - "drift": The sequences designed by the genetic algorithm 
        - "native": Native yeast promoter test sequences
        - "paired": Paired reference (native)/alternative (single mutations into each native sequence) sequences
    transform : callable, optional
        Transformation function applied to each sequence. Useful for data augmentation
        or sequence encoding. Should accept a sequence string and return transformed data.
    target_transform : callable, optional
        Transformation function applied to target values. Useful for normalization
        or target processing.
    root : str, optional
        Root directory for data storage. If None, uses default data directory.

    Raises
    ------
    FileNotFoundError
        If the required data file cannot be found or downloaded
    ValueError
        If provided split, dataset_env_type, or test_dataset_type parameters are invalid

    Examples
    --------
    >>> # Load training data from defined environment
    >>> train_dataset = VaishnavDataset(
    ...     split="train",
    ...     dataset_env_type="defined"
    ... )
    >>>
    >>> # Load validation data from complex environment
    >>> val_dataset = VaishnavDataset(
    ...     split="val", 
    ...     dataset_env_type="complex"
    ... )
    >>>
    >>> # Load test data for distribution drift scenario
    >>> test_drift = VaishnavDataset(
    ...     split="test",
    ...     dataset_env_type="defined",
    ...     test_dataset_type="drift"
    ... )
    >>>
    >>> # Load test data for paired sequence analysis
    >>> test_paired = VaishnavDataset(
    ...     split="test",
    ...     dataset_env_type="complex", 
    ...     test_dataset_type="paired"
    ... )

    Notes
    -----
    - Yeast strain information: Uses Y8205, S288C::ura3 and related strains
    - Environmental contexts:
        * "defined": defined medium (synthetic defined medium lacking uracil (SD-Ura))
        * "complex": complex medium (yeast extract, peptone and dextrose (YPD))
    - Test scenarios:
        * "native": Native yeast promoter test sequences
        * "drift": The sequences designed by the genetic algorithm 
        * "paired": Paired reference (native)/alternative (single mutations into each native sequence) sequences
    - Target values: 
        * For "paired" test type: delta_measured (difference in activity)
        * For other types: label (absolute activity measurement)

    See Also
    --------
    DreamDataset : Related class for DREAM challenge MPRA data
    """

    FLAG = "Vaishnav"
    PLASMID = "aactctcaaggatcttaccgctgttgagatccagttcgatgtaacccactcgtgcacccaactgatcttcagcatcttttactttcaccagcgtttctgggtgagcaaaaacaggaaggcaaaatgccgcaaaaaagggaataagggcgacacggaaatgttgaatactcatactcttcctttttcaatattattgaagcatttatcagggttattgtctcatgagcggatacatatttgaatgtatttagaaaaataaacaaataggggttccgcgcacatttccccgaaaagtgccacctgacgtcatctatattaccctgttatccctagcggatctgccggtagaggtgtggtcaataagagcgacctcatactatacctgagaaagcaacctgacctacaggaaagagttactcaagaataagaattttcgttttaaaacctaagagtcactttaaaatttgtatacacttattttttttataacttatttaataataaaaatcataaatcataagaaattcgcttatttagaagtGGCGCGCCGGTCCGttacttgtacagctcgtccatgccgccggtggagtggcggccctcggcgcgttcgtactgttccacgatggtgtagtcctcgttgtgggaggtgatgtccaacttgatgttgacgttgtaggcgccgggcagctgcacgggcttcttggccttgtaggtggtcttgacctcagcgtcgtagtggccgccgtccttcagcttcagcctctgcttgatctcgcccttcagggcgccgtcctcggggtacatccgctcggaggaggcctcccagcccatggtcttcttctgcattacggggccgtcggaggggaagttggtgccgcgcagcttcaccttgtagatgaactcgccgtcctgcagggaggagtcctgggtcacggtcaccacgccgccgtcctcgaagttcatcacgcgctcccacttgaagccctcggggaaggacagcttcaagtagtcggggatgtcggcggggtgcttcacgtaggccttggagccgtacatgaactgaggggacaggatgtcccaggcgaagggcagggggccacccttggtcaccttcagcttggcggtctgggtgccctcgtaggggcggccctcgccctcgccctcgatctcgaactcgtggccgttcacggagccctccatgtgcaccttgaagcgcatgaactccttgatgatggccatgttatcctcctcgcccttgctcacCATGGTACTAGTGTTTAGTTAATTATAGTTCGTTGACCGTATATTCTAAAAACAAGTACTCCTTAAAAAAAAACCTTGAAGGGAATAAACAAGTAGAATAGATAGAGAGAAAAATAGAAAATGCAAGAGAATTTATATATTAGAAAGAGAGAAAGAAAAATGGAAAAAAAAAAATAGGAAAAGCCAGAAATAGCACTAGAAGGAGCGACACCAGAAAAGAAGGTGATGGAACCAATTTAGCTATATATAGTTAACTACCGGCTCGATCATCTCTGCCTCCAGCATAGTCGAAGAAGAATTTTTTTTTTCTTGAGGCTTCTGTCAGCAACTCGTATTTTTTCTTTCTTTTTTGGTGAGCCTAAAAAGTTCCCACGTTCTCTTGTACGACGCCGTCACAAACAACCTTATGGGTAATTTGTCGCGGTCTGGGTGTATAAATGTGTGGGTGCAACATGAATGTACGGAGGTAGTTTGCTGATTGGCGGTCTATAGATACCTTGGTTATGGCGCCCTCACAGCCGGCAGGGGAAGCGCCTACGCTTGACATCTACTATATGTAAGTATACGGCCCCATATATAggccctttcgtctcgcgcgtttcggtgatgacggtgaaaacctctgacacatgcagctcccggagacggtcacagcttgtctgtaagcggatgccgggagcagacaagcccgtcagggcgcgtcagcgggtgttggcgggtgtcggggctggcttaactatgcggcatcagagcagattgtactgagagtgcaccatatggacatattgtcgttagaacgcggctacaattaatacataaccttatgtatcatacacatacgatttaggtgacactatagaacgcggccgccagctgaagctttaactatgcggcatcagagcagattgtactgagagtgcaccataccaccttttcaattcatcattttttttttattcttttttttgatttcggtttccttgaaatttttttgattcggtaatctccgaacagaaggaagaacgaaggaaggagcacagacttagattggtatatatacgcatatgtagtgttgaagaaacatgaaattgcccagtattcttaacccaactgcacagaacaaaaacctgcaggaaacgaagataaatcatgtcgaaagctacatataaggaacgtgctgctactcatcctagtcctgttgctgccaagctatttaatatcatgcacgaaaagcaaacaaacttgtgtgcttcattggatgttcgtaccaccaaggaattactggagttagttgaagcattaggtcccaaaatttgtttactaaaaacacatgtggatatcttgactgatttttccatggagggcacagttaagccgctaaaggcattatccgccaagtacaattttttactcttcgaagacagaaaatttgctgacattggtaatacagtcaaattgcagtactctgcgggtgtatacagaatagcagaatgggcagacattacgaatgcacacggtgtggtgggcccaggtattgttagcggtttgaagcaggcggcagaagaagtaacaaaggaacctagaggccttttgatgttagcagaattgtcatgcaagggctccctatctactggagaatatactaagggtactgttgacattgcgaagagcgacaaagattttgttatcggctttattgctcaaagagacatgggtggaagagatgaaggttacgattggttgattatgacacccggtgtgggtttagatgacaagggagacgcattgggtcaacagtatagaaccgtggatgatgtggtctctacaggatctgacattattattgttggaagaggactatttgcaaagggaagggatgctaaggtagagggtgaacgttacagaaaagcaggctgggaagcatatttgagaagatgcggccagcaaaactaaaaaactgtattataagtaaatgcatgtatactaaactcacaaattagagcttcaatttaattatatcagttattaccctatgcggtgtgaaataccgcacagatgcgtaaggagaaaataccgcatcaggaaattgtaagcgttaatattttgttaaaattcgcgttaaatttttgttaaatcagctcattttttaaccaataggccgaaatcggcaaaatcccttataaatcaaaagaatagaccgagatagggttgagtgttgttccagtttggaacaagagtccactattaaagaacgtggactccaacgtcaaagggcgaaaaaccgtctatcagggcgatggcccactacgtgaaccatcaccctaatcaagtGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGAAGGCAAAGatgtctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgctacttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctagatacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagaccagagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaactataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaattagctgaccattatcaacaaaatactccaattggtgatggtccagtcttgttaccagacaaccattacttatcctatcaatctgccttatccaaagatccaaacgaaaagagagaccacatggtcttgttagaatttgttactgctgctggtattacccatggtatggatgaattgtacaaataaggcgcgccacttctaaataagcgaatttcttatgatttatgatttttattattaaataagttataaaaaaaataagtgtatacaaattttaaagtgactcttaggttttaaaacgaaaattcttattcttgagtaactctttcctgtaggtcaggttgctttctcaggtatagtatgaggtcgctcttattgaccacacctctaccggcagatccgctagggataacagggtaatataGATCTGTTTAGCTTGCCTCGTCCCCGCCGGGTCACCCGGCCAGCGACATGGAGGCCCAGAATACCCTCCTTGACAGTCTTGACGTGCGCAGCTCAGGGGCATGATGTGACTGTCGCCCGTACATTTAGCCCATACATCCCCATGTATAATCATTTGCATCCATACATTTTGATGGCCGCACGGCGCGAAGCAAAAATTACGGCTCCTCGCTGCAGACCTGCGAGCAGGGAAACGCTCCCCTCACAGACGCGTTGAATTGTCCCCACGCCGCGCCCCTGTAGAGAAATATAAAAGGTTAGGATTTGCCACTGAGGTTCTTCTTTCATATACTTCCTTTTAAAATCTTGCTAGGATACAGTTCTCACATCACATCCGAACATAAACAACCATGGGTACCACTCTTGACGACACGGCTTACCGGTACCGCACCAGTGTCCCGGGGGACGCCGAGGCCATCGAGGCACTGGATGGGTCCTTCACCACCGACACCGTCTTCCGCGTCACCGCCACCGGGGACGGCTTCACCCTGCGGGAGGTGCCGGTGGACCCGCCCCTGACCAAGGTGTTCCCCGACGACGAATCGGACGACGAATCGGACGACGGGGAGGACGGCGACCCGGACTCCCGGACGTTCGTCGCGTACGGGGACGACGGCGACCTGGCGGGCTTCGTGGTCGTCTCGTACTCCGGCTGGAACCGCCGGCTGACCGTCGAGGACATCGAGGTCGCCCCGGAGCACCGGGGGCACGGGGTCGGGCGCGCGTTGATGGGGCTCGCGACGGAGTTCGCCCGCGAGCGGGGCGCCGGGCACCTCTGGCTGGAGGTCACCAACGTCAACGCACCGGCGATCCACGCGTACCGGCGGATGGGGTTCACCCTCTGCGGCCTGGACACCGCCCTGTACGACGGCACCGCCTCGGACGGCGAGCAGGCGCTCTACATGAGCATGCCCTGCCCCTAATCAGTACTGACAATAAAAAGATTCTTGTTTTCAAGAACTTGTCATTTGTATAGTTTTTTTATATTGTAGTTGTTCTATTTTAATCAAATGTTAGCGTGATTTATATTTTTTTTCGCCTCGACATCATCTGCCCAGATGCGAAGTTAAGTGCGCAGAAAGTAATATCATGCGTCAATCGTATGTGAATGCTGGTCGCTATACTGCTGTCGATTCGATACTAACGCCGCCATCCAGTGTCGAAAACGAGCTCGaattcctgggtccttttcatcacgtgctataaaaataattataatttaaattttttaatataaatatataaattaaaaatagaaagtaaaaaaagaaattaaagaaaaaatagtttttgttttccgaagatgtaaaagactctagggggatcgccaacaaatactaccttttatcttgctcttcctgctctcaggtattaatgccgaattgtttcatcttgtctgtgtagaagaccacacacgaaaatcctgtgattttacattttacttatcgttaatcgaatgtatatctatttaatctgcttttcttgtctaataaatatatatgtaaagtacgctttttgttgaaattttttaaacctttgtttatttttttttcttcattccgtaactcttctaccttctttatttactttctaaaatccaaatacaaaacataaaaataaataaacacagagtaaattcccaaattattccatcattaaaagatacgaggcgcgtgtaagttacaggcaagcgatccgtccGATATCatcagatccactagtggcctatgcggccgcggatctgccggtctccctatagtgagtcgtattaatttcgataagccaggttaacctgcattaatgaatcggccaacgcgcggggagaggcggtttgcgtattgggcgctcttccgcttcctcgctcactgactcgctgcgctcggtcgttcggctgcggcgagcggtatcagctcactcaaaggcggtaatacggttatccacagaatcaggggataacgcaggaaagaacatgtgagcaaaaggccagcaaaaggccaggaaccgtaaaaaggccgcgttgctggcgtttttccataggctccgcccccctgacgagcatcacaaaaatcgacgctcaagtcagaggtggcgaaacccgacaggactataaagataccaggcgtttccccctggaagctccctcgtgcgctctcctgttccgaccctgccgcttaccggatacctgtccgcctttctcccttcgggaagcgtggcgctttctcaTAgctcacgctgtaggtatctcagttcggtgtaggtcgttcgctccaagctgggctgtgtgcacgaaccccccgttcagcccgaccgctgcgccttatccggtaactatcgtcttgagtccaacccggtaagacacgacttatcgccactggcagcagccactggtaacaggattagcagagcgaggtatgtaggcggtgctacagagttcttgaagtggtggcctaactacggctacactagaagAacagtatttggtatctgcgctctgctgaagccagttaccttcggaaaaagagttggtagctcttgatccggcaaacaaaccaccgctggtagcggtggtttttttgtttgcaagcagcagattacgcgcagaaaaaaaggatctcaagaagatcctttgatcttttctacggggtctgacgctcagtggaacgaaaactcacgttaagggattttggtcatgagattatcaaaaaggatcttcacctagatccttttaaattaaaaatgaagttttaaatcaatctaaagtatatatgagtaaacttggtctgacagttaccaatgcttaatcagtgaggcacctatctcagcgatctgtctatttcgttcatccatagttgcctgactccccgtcgtgtagataactacgatacgggagggcttaccatctggccccagtgctgcaatgataccgcgagacccacgTtcaccggctccagatttatcagcaataaaccagccagccggaagggccgagcgcagaagtggtcctgcaactttatccgcctccatccagtctattaattgttgccgggaagctagagtaagtagttcgccagttaatagtttgcgcaacgttgttgccattgctacaggcatcgtggtgtcacgctcgtcgtttggtatggcttcattcagctccggttcccaacgatcaaggcgagttacatgatcccccatgttgtgcaaaaaagcggttagctccttcggtcctccgatcgttgtcagaagtaagttggccgcagtgttatcactcatggttatggcagcactgcataattctcttactgtcatgccatccgtaagatgcttttctgtgactggtgagtactcaaccaagtcattctgagaatagtgtatgcggcgaccgagttgctcttgcccggcgtcaatacgggataataccgcgccacatagcagaactttaaaagtgctcatcattggaaaacgttcttcggggcgaa"

    LEFT_FLANK = "TGCATTTTTTTCACATC"
    RIGHT_FLANK = "GGTTACGGCTGTT"

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        dataset_env_type: Literal["defined", "complex"],
        test_dataset_type: Literal["drift", "native", "paired"] = None,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize VaishnavDataset for environmental context MPRA analysis.

        The dataset is designed for studying transcriptional regulation across
        different environmental contexts in yeast, providing insights into
        context-dependent regulatory mechanisms.

        Parameters
        ----------
        split : Literal["train", "val", "test"]
            Data split specification:
            - "train": Training data
            - "val": Validation data
            - "test": Test data
        dataset_env_type : Literal["defined", "complex"]
            Environmental context type:
            - "defined": Synthetic defined medium lacking uracil (SD-Ura)
            - "complex": yeast extract, peptone and dextrose (YPD)
        test_dataset_type : Literal["drift", "native", "paired"], optional
            Required for test split only. Specifies test scenario:
            - "drift": The sequences designed by the genetic algorithm 
            - "native": Native yeast promoter test sequences
            - "paired": Paired reference (native)/alternative (single mutations into each native sequence) sequences
        transform : callable, optional
            Transformation function applied to each sequence. Useful for data augmentation
            or sequence encoding. Should accept a sequence string and return transformed data.
        target_transform : callable, optional
            Transformation function applied to target values. Useful for normalization
            or target processing.
        root : str, optional
            Root directory for data storage. If None, uses default data directory.
        """
        super().__init__(split, root)
        self.cell_type = "strains Y8205, S288C::ura3, etc"
        # Initialize transformations
        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"

        # Parse and validate inputs
        self.split, self.target_column = self.parse_dataset_config(
            split, dataset_env_type, test_dataset_type
        )

        # Load and prepare dataset
        file_name = self._define_dataset(
            self.split, dataset_env_type, test_dataset_type
        )
        self.df = self._load_and_prepare_data(file_name)

        # Prepare data structure based on split type
        self._prepare_data_structure(test_dataset_type)

        self.name_for_split_info = self.prefix + dataset_env_type + "_"

    def _define_dataset(
        self, split: str, dataset_env_type: str, test_dataset_type: str
    ) -> str:
        """
        Determine which dataset file to load based on split and environment type.

        Constructs the appropriate filename according to the dataset configuration:
        - Training/validation: {env_type}_train_val
        - Test: {env_type}_{test_type}

        Parameters
        ----------
        split : str
            Data split ("train", "val", or "test")
        dataset_env_type : str
            Environmental context type ("defined" or "complex")
        test_dataset_type : str
            Test scenario type ("drift", "native", or "paired")

        Returns
        -------
        str
            Dataset filename without extension

        Warns
        -----
        UserWarning
            If test_dataset_type is provided for train/val splits
        """
        if split in ["train", "val"]:
            if test_dataset_type is not None:
                warnings.warn(
                    f"WARNING! A {self.split} set has been selected."
                    "\nIn this case, the parameter 'test_dataset_type' is ignored.",
                    stacklevel=1,
                )
            file_name = f"{dataset_env_type}_train_val"
        else:
            file_name = f"{dataset_env_type}_{test_dataset_type}"

        return file_name

    def _load_and_prepare_data(self, dataset: str) -> pd.DataFrame:
        """
        Load dataset from TSV file and return as DataFrame.

        Downloads the data file if not already present, then loads it into pandas.

        Parameters
        ----------
        dataset : str
            Dataset filename without extension

        Returns
        -------
        pd.DataFrame
            Loaded dataset

        Raises
        ------
        FileNotFoundError
            If the data file cannot be found after attempted download
        """
        try:
            file_name = self.prefix + dataset + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        return df

    def _prepare_data_structure(self, test_dataset_type):
        """
        Prepare internal data structure based on dataset configuration.

        For train/val splits, filters data to the specified split.
        For test split with "paired" type, includes alternative sequences.

        Parameters
        ----------
        test_dataset_type : str
            Test scenario type, used to determine if alternative sequences
            should be included
        """
        is_alt = False

        if self.split in ["train", "val"]:
            self.df = self.df[self.df.split.isin([self.split])].reset_index(drop=True)

        if self.split == "test" and test_dataset_type == "paired":
            is_alt = True

        self.ds = {
            "targets": self.df[self.target_column].to_numpy(),
            "seq": self.df.seq.to_numpy(),
        }

        if is_alt:
            self.ds["seq_alt"] = self.df.seq_alt.to_numpy()

    def parse_dataset_config(
        self, split: str, complex_or_defined: str, test_type: str
    ) -> str:
        """
        Parse and validate dataset configuration parameters.

        Validates all input parameters and determines the appropriate target column
        based on the test scenario.

        Parameters
        ----------
        split : str
            Data split specification
        complex_or_defined : str
            Environmental context type
        test_type : str
            Test scenario type

        Returns
        -------
        tuple[str, str]
            Validated split and target column name

        Raises
        ------
        ValueError
            If any parameter is invalid or test_type is None for test split
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}
        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )
        if complex_or_defined not in ["complex", "defined"]:
            raise ValueError(
                f"Invalid dataset_env_type value: {complex_or_defined}. Expected 'complex' or 'defined'."
            )
        if test_type is not None and test_type not in ["native", "drift", "paired"]:
            raise ValueError(
                f"Invalid test_dataset_type value: {test_type}. Expected 'native', 'drift' or 'paired'."
            )
        elif test_type is None and split == "test":
            raise ValueError(
                "Parameter 'test_type' cannot be None for test split. "
                "Expected one of: 'native', 'drift', 'paired'."
            )

        target_column = "delta_measured" if test_type == "paired" else "label"

        return split, target_column
