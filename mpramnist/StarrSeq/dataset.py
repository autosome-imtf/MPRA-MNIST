import pandas as pd
import numpy as np
from typing import List, Union, Literal
import pyfastx
from mpramnist.mpradataset import MpraDataset
import os


class StarrSeqDataset(MpraDataset):
    """
    A dataset class for StarrSeq MPRA (Massively Parallel Reporter Assay) data.
    
    This class handles loading and processing of various StarrSeq experimental tasks
    including random enhancers, genomic promoters, capture promoters, genomic enhancers,
    ATAC-seq data, and binary classification tasks.

    The dataset supports different splitting strategies:
    - Default splits (train/val/test) for: randomenhancer, genomicpromoter, capturepromoter, binary
    - Chromosome-based splits for: genomicenhancer, atacseq

    Key Features:
    - Automatic download of missing data files
    - Flexible split configurations
    - Support for binary classification with different enhancer types
    - Integration with PyTorch transforms
    - Efficient FASTA parsing with pyfastx

    Constants
    ----------
    FLAG : str
        Constant identifier for the dataset type ("StarrSeq").
    TASKS : dict
        Mapping of available task names to their file prefixes.

    Notes
    -----
    Data files are expected to be in compressed FASTA format (.fasta.gz) with
    labels stored in the comment section of each sequence record.

    Examples
    --------
    >>> # Load random enhancer training data
    >>> dataset = StarrSeqDataset(task="randomenhancer", split="train")
    
    >>> # Load genomic enhancer data for specific chromosomes
    >>> dataset = StarrSeqDataset(task="genomicenhancer", split=["chr1", "chr2"])
    
    >>> # Load binary classification data with specific class
    >>> dataset = StarrSeqDataset(
    ...     task="binary", 
    ...     split="train", 
    ...     binary_class="enhancer_from_input"
    ... )
    """

    FLAG = "StarrSeq"
    TASKS = {
        "randomenhancer": "ranEnh_",  # Splits are available for train, val, and test only.
        "genomicpromoter": "genProm_",  # Splits are available for train, val, and test only.
        "capturepromoter": "CaptProm_",  # Splits are available for train, val, and test only.
        "genomicenhancer": "genEnh_",  # Splitting is based on chromosomes, train/val/test available too.
        "atacseq": "ATACSeq_",  # Splitting is based on chromosomes, train/val/test available too.
        "binary": "binary_",  # Splits are available for train, val, and test only.
    }

    def __init__(
        self,
        task: str,
        split: str | List[str] | List[int] | int,
        binary_class: Literal[
            "enhancer_from_input", "promoter_from_input", "enhancer_permutated"
        ] = None,  # (optional), supportable only for binary promoter-enhancer experiment
        root=None,
        transform=None,
        target_transform=None,
    ):
        """
        Initialize the STARR-seq MPRA dataset.

        Parameters
        ----------
        task : str
            The name of the task to load. Must be one of:
            - "randomenhancer": Random enhancer data
            - "genomicpromoter": Genomic promoter data  
            - "capturepromoter": Capture promoter data
            - "genomicenhancer": Genomic enhancer data (chromosome-based splits)
            - "atacseq": ATAC-seq data (chromosome-based splits)
            - "binary": Binary classification task
        split : str | List[str] | List[int] | int
            Specifies how to split the data:
            - For default split tasks: "train", "val", or "test"
            - For chromosome-based tasks: chromosome names/numbers or predefined splits
        binary_class : Literal["enhancer_from_input", "promoter_from_input", "enhancer_permutated"], optional
            Specifies the binary class type (only supported for binary task with train split).
            If None, uses default promoter and enhancer files.
        root : str, optional
            Root directory where data is stored. If None, uses default data path.
        transform : callable, optional
            Function to apply transformations to the input sequences.
        target_transform : callable, optional
            Function to apply transformations to the target labels.

        Raises
        ------
        ValueError
            If an invalid task name is provided.
            If an invalid binary_class is provided for binary task.
            If an invalid split value is provided.
        Notes
        -----
        - For tasks "randomenhancer", "genomicpromoter", "capturepromoter", and "binary",
        only predefined splits ("train", "val", "test") are available.
        - For tasks "genomicenhancer" and "atacseq", chromosome-based splits are supported
        using chromosome names (e.g., "chr1", "chrX") or numbers (e.g., 1, 22).
        - The binary task requires special handling with separate promoter and enhancer files.
        """
        super().__init__(split, root)

        if task.lower() not in self.TASKS:
            raise ValueError(
                f"incorrect task '{task}'. Expected one of {list(self.TASKS.keys())}."
            )
        self.task = task.lower()

        self.binary_class = binary_class

        self.cell_types = ["GP5d", "HepG2"]
        self._cell_type = None
        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"

        self.ds = self.load_data(split, self.task)
        self.name_for_split_info = self.prefix+self.TASKS[self.task]

    def load_data(self, split, task):
        """
        Load data based on the task type and split configuration.

        This method routes to the appropriate data loading strategy based on the task type:
        - For tasks with default splits: uses predefined train/val/test splits
        - For chromosome-based tasks: uses chromosome-based splitting
        - For binary tasks: handles special binary classification data

        Parameters
        ----------
        split : str | List[str] | List[int] | int
            Defines how to split the dataset. Can be:
            - For default splits: "train", "val", "test"
            - For chromosome splits: chromosome names/numbers or predefined splits
        task : str
            The name of the task. Must be one of the keys in self.TASKS.

        Returns
        -------
        ds : dict
            Dictionary containing the loaded dataset with keys:
            - "targets": numpy array of labels/values
            - "seq": numpy array of sequences
            - "seq_enh": numpy array of enhancer sequences (for binary task only)

        Raises
        ------
        ValueError
            If task is not recognized or invalid split configuration is provided.
        """
        # Default split tasks
        if task in ["randomenhancer", "genomicpromoter", "capturepromoter", "binary"]:
            is_split_default = True

            if task == "binary":
                ds = self.task_binary(self.TASKS[task], self.binary_class, self.split)
            else:
                ds = self.task_with_default_split(self.TASKS[task], self.split)

        # Chromosome-based split tasks
        elif task in ["genomicenhancer", "atacseq"]:
            is_split_default = False
            self.split = self.split_parse(split, is_split_default)

            ds = self.task_with_various_split(self.TASKS[task], self.split)

        return ds

    ###################### Task data preparation ##############################
    def read_fasta(self, file_path, file_name, return_names=False):
        """
        Read sequences and labels from a FASTA file.

        This method handles the reading of compressed (.gz) FASTA files and extracts
        sequences, their names, and corresponding labels. The labels are expected to
        be in the comment section of each FASTA record.

        Parameters
        ----------
        file_path : str
            Path to the directory containing the FASTA file.
        file_name : str
            Name of the FASTA file (must be compressed with .gz).
        return_names : bool, optional
            If True, returns sequence names along with sequences and labels.
            Default is False.

        Returns
        -------
        tuple
            Depending on return_names:
            - If return_names=False: (seqs, labels)
            - If return_names=True: (names, seqs, labels)
            
            Where:
            - names : list of str
                Sequence identifiers (chromosome names or other IDs)
            - seqs : list of str
                DNA sequences
            - labels : list of float
                Corresponding activity values/labels

        Raises
        ------
        FileNotFoundError
            If the specified FASTA file cannot be found.
        
        Notes
        -----
        - The method automatically downloads the file if it doesn't exist locally.
        - Sequence names are extracted by splitting the FASTA header at the first colon.
        - Labels are converted to numpy.float32 for consistency.
        """
        try:
            self.download(file_path, file_name)
            file_path = os.path.join(file_path, file_name)
            fa = pyfastx.Fastx(file_path, comment=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        names, seqs, labels = [], [], []
        for name, seq, label in fa:
            seqs.append(seq)
            names.append(name.split(":")[0])
            labels.append(np.float32(label))

        if return_names:
            return names, seqs, labels
        else:
            return seqs, labels

    # for random enhancer, genomic promoter and capture promoter data
    def task_with_default_split(self, task, split):
        """
        Load data for tasks that use predefined train/val/test splits.

        This method handles tasks where data is already split into standard
        training, validation, and test sets stored in separate files.

        Parameters
        ----------
        task : str
            Task prefix from self.TASKS (e.g., "ranEnh_", "genProm_")
        split : str
            Split identifier: "train", "val", or "test"

        Returns
        -------
        dict
            Dictionary with keys:
            - "targets": list of float
                Activity values for each sequence
            - "seq": list of str
                DNA sequences

        Notes
        -----
        File naming convention: {prefix}{task}{split}.fasta.gz
        Example: "StarrSeq_ranEnh_train.fasta.gz"
        """
    
        file_name = f"{self.prefix}{task}{split}.fasta.gz"
        seqs, labels = self.read_fasta(self._data_path, file_name)
        return {"targets": labels, "seq": seqs}

    # for genomic Enhancer and ATACseq data
    def task_with_various_split(self, task, split):
        """
        Load data for tasks that use chromosome-based splits.

        This method loads a combined FASTA file containing all chromosomes
        and then filters sequences based on the specified chromosome split.

        Parameters
        ----------
        task : str
            Task prefix from self.TASKS (e.g., "genEnh_", "ATACSeq_")
        split : list of str
            List of chromosome names to include (e.g., ["chr1", "chr2", "chrX"])

        Returns
        -------
        dict
            Dictionary with keys:
            - "targets": numpy array of float
                Activity values for each sequence
            - "seq": numpy array of str
                DNA sequences

        Notes
        -----
        - The input file is expected to contain all chromosomes.
        - Sequences are filtered based on the 'chr' column matching the split list.
        - Returns numpy arrays for better performance with large datasets.
        """
        file_name = f"{self.prefix}{task}all_chr_file.fasta.gz"
        names, seqs, labels = self.read_fasta(self._data_path, file_name, True)
        data = pd.DataFrame({"chr": names, "seq": seqs, "targets": labels})
        data = data[data.chr.isin(split)].reset_index(drop=True)
        return {"targets": data.targets.to_numpy(), "seq": data.seq.to_numpy()}

    def task_binary(self, task, binary_class, split):
        """
        Load data for binary classification tasks.

        This method handles the special case of binary classification where
        both promoter and enhancer sequences need to be loaded. The method
        supports different configurations for training and evaluation.

        Parameters
        ----------
        task : str
            Task prefix for binary tasks ("binary_")
        binary_class : str or None
            Specifies the binary class configuration:
            - "promoter_from_input": Use default enhancers but inactive promoters
            - "enhancer_from_input": Use default promoters but inactive enhancers
            - "enhancer_permutated": Use active permutated enhancers
            - None: Use default promoter and enhancer files
        split : str
            Split identifier: "train", "val", or "test"

        Returns
        -------
        dict
            Dictionary with keys:
            - "targets": list of float
                Promoter activity values
            - "seq": list of str
                Promoter sequences
            - "seq_enh": list of str
                Enhancer sequences

        Raises
        ------
        ValueError
            If invalid binary_class is provided for training split.

        Notes
        -----
        - For training split with binary_class specified, uses specialized files.
        - For other cases, uses default promoter and enhancer files.
        - File naming follows pattern: {prefix}{task}{split}_{class}.fasta.gz
        - For more information, please read the original article https://doi.org/10.1038/s41588-021-01009-4
        """

        binary_train = [
            "promoter_from_input",
            "enhancer_permutated",
            "enhancer_from_input",
        ]
        if split == "train":
            if binary_class is not None:
                if binary_class not in binary_train:
                    raise ValueError(
                        f"'binary_class' must be one of {binary_train} for training"
                    )
                else:
                    if binary_class.split("_")[0] == "promoter":
                        file_name_prom = (
                            f"{self.prefix}{task}{split}_{binary_class}.fasta.gz"
                        )
                        file_name_enh = f"{self.prefix}{task}{split}_enhancer.fasta.gz"
                    else:
                        file_name_prom = f"{self.prefix}{task}{split}_promoter.fasta.gz"
                        file_name_enh = (
                            f"{self.prefix}{task}{split}_{binary_class}.fasta.gz"
                        )

                    seqs_prom, labels_prom = self.read_fasta(
                        self._data_path, file_name_prom
                    )
                    seqs_enh, labels_enh = self.read_fasta(
                        self._data_path, file_name_enh
                    )

                    print(f"using train {binary_class}")

                    return {
                        "targets": labels_prom,
                        "seq": seqs_prom,
                        "seq_enh": seqs_enh,
                    }

            elif binary_class is None:
                pass

        file_name_prom = f"{self.prefix}{task}{split}_promoter.fasta.gz"
        seqs_prom, labels_prom = self.read_fasta(self._data_path, file_name_prom)

        file_name_enh = f"{self.prefix}{task}{split}_enhancer.fasta.gz"
        seqs_enh, labels_enh = self.read_fasta(self._data_path, file_name_enh)

        print(f"using {split}")

        return {"targets": labels_prom, "seq": seqs_prom, "seq_enh": seqs_enh}

    ################################ Split Parsing #################################
    def split_parse(
        self, split: list[int | str] | int | str, is_split_default: bool
    ) -> str | list[str]:
        """
        Parses the input split and returns a list of chromosome names or default splits.

        This method provides flexible input handling for different split configurations:
        - Converts chromosome numbers to standardized chromosome names
        - Validates split inputs against allowed values
        - Handles both default splits and chromosome-based splits

        Parameters
        ----------
        split : list[int | str] | int | str
            The split identifier(s). Can be:
            - str: 
                * For default splits: 'train', 'val', 'test'
                * For chromosome splits: chromosome name ('chrX') or number ('X', '1-22')
            - int: Chromosome number (1-22)
            - list[int | str]: List of chromosome identifiers
        is_split_default : bool
            If True, expects default splits ('train', 'val', 'test').
            If False, expects chromosome-based splits.

        Returns
        -------
        list[str]
            A list of chromosome names in format 'chrX' or split identifiers.

        Raises
        ------
        ValueError
            If invalid split value is provided or chromosome is out of range.

        Examples
        --------
        >>> split_parse("train", True)
        'train'
        
        >>> split_parse(["chr1", "chr2"], False)
        ['chr1', 'chr2']
        
        >>> split_parse([1, 2, "X"], False)  
        ['chr1', 'chr2', 'chrX']

        Notes
        -----
        - Predefined chromosome splits:
            train: chr1,3,5,7,9,13-22,X
            val: chr4,6,8  
            test: chr2,10,11
        - Chromosomes 1-22 and X are supported
        """

        def convert_to_chr(value: Union[int, str]) -> str:
            """Helper function to convert integer or string to 'chrX' format."""
            if isinstance(value, int):
                return f"chr{value}"
            return value

        # Default split logic (train, val, test)
        if is_split_default:
            split_default = {"train": "train", "val": "val", "test": "test"}

            if isinstance(split, str):
                if split not in split_default:
                    raise ValueError(
                        f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
                    )
                split = split  # Make it a list to maintain uniformity
            else:
                raise ValueError(
                    "Invalid split value. Expected 'train', 'val', or 'test'."
                )

        else:
            # Chromosomal split logic (specific chromosomes for train, val, test)
            split_chr = {
                "train": [
                    "chr1",
                    "chr3",
                    "chr5",
                    "chr7",
                    "chr9",
                    "chr13",
                    "chr14",
                    "chr15",
                    "chr16",
                    "chr17",
                    "chr18",
                    "chr19",
                    "chr20",
                    "chr21",
                    "chr22",
                    "chrX",
                ],
                "val": ["chr4", "chr6", "chr8"],
                "test": ["chr2", "chr10", "chr11"],
            }

            list_of_chr = [str(i) for i in range(1, 23)] + ["X"]
            list_of_named_chr = ["chr" + i for i in list_of_chr]

            if isinstance(split, str):
                if split == "X":
                    split = ["chrX"]
                elif split in split_chr:
                    split = split_chr[split]
                elif split in list_of_chr:
                    split = [f"chr{split}"]
                elif split in list_of_named_chr:
                    split = [split]
                else:
                    raise ValueError(
                        f"Invalid split value: {split}. Expected 'train', 'val', or 'test', range 1-22, 'X' or name 'chrx', where x is number 1-22 or X."
                    )

            elif isinstance(split, int):
                split = [convert_to_chr(split)]

            elif isinstance(split, list):
                split = [convert_to_chr(item) for item in split]

                # Validate all items in the list
                for item in split:
                    if not item.startswith("chr") or item[3:] not in list_of_chr:
                        raise ValueError(
                            f"Invalid split value: {item}. Must be in range 1-22 or 'X'."
                        )

        return split
