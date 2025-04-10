import pandas as pd
import numpy as np
from typing import List, T, Union, Literal
import torch
import pyfastx
import pyfaidx
from mpramnist.mpradataset import MpraDataset
import os
import warnings

class StarrSeqDataset(MpraDataset):
    
    FLAG = "StarrSeq"
    TASKS = {
             "randomenhancer"  : "ranEnh_",  # Splits are available for train, val, and test only.
             "genomicpromoter" : "genProm_", # Splits are available for train, val, and test only.
             "capturepromoter" : "CaptProm_",# Splits are available for train, val, and test only.
             
             "genomicenhancer" : "genEnh_",  # Splitting is based on chromosomes, train/val/test available too.
             "atacseq"         : "ATACSeq_", # Splitting is based on chromosomes, train/val/test available too.
             
             "binary"          : "binary_"   # Splits are available for train, val, and test only.
            }
    
    def __init__(self,
                 task: str,
                 split: str | List[str] | List[int] | int,
                 binary_class: Literal["enhancer_from_input", "promoter_from_input", "enhancer_permutated"] = None, # (optional), supportable only for binary promoter-enhancer experiment
                 root = None,
                 transform = None,
                 target_transform = None,
                ):
        """
        Attributes
        ----------
        task : str
            The name of the task, one of ["randomenhancer", "genomicenhancer", "genomicpromoter", 
            "capturepromoter", "atacseq", "binary"].
        split : str | List[int] | int
            Specifies how to split the data (e.g., into training and testing sets).
        binary_class : str, optional
            Specifies enhancer_from_input/promoter_from_input/enhancer_permutated. allowed only for train split. Defailt None
        transform : callable, optional
            Function to apply transformations to the input sequences.
        target_transform : callable, optional
            Function to apply transformations to the target labels.
        """
        super().__init__(split, root)
        
        if task.lower() not in self.TASKS:
            raise ValueError(f"incorrect task '{task}'. Expected one of {list(self.TASKS.keys())}.")
        self.task = task.lower()

        self.binary_class = binary_class

        self.cell_types = None
        self._cell_type = None
        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"

        self.ds = self.load_data(split, self.task)

    def load_data(self, split, task):
        """
        Load data based on the task type and split configuration.
    
        Parameters
        ----------
        split : str | List[str] | List[int] | int
            Defines how to split the dataset (e.g., train/val/test or chromosome-based).
        task : str
            The name of the task (e.g., "randomenhancer", "binary").
    
        Returns
        -------
        ds : dict
            The loaded dict corresponding to the specified task and split.
        """
        # Default split tasks
        if task in ["randomenhancer", "genomicpromoter", "capturepromoter", "binary"]:
            is_split_default = True
            self.split = self.split_parse(split, is_split_default)
            if task == "genomicpromoter" and split in ["val", "test"]:
                warnings.warn(
                    "WARNING! The test dataset released by the authors of the study contains an error causing positive sequences to duplicate negative ones."
                    "We suggest using the validation dataset as the test instead.",
                    stacklevel=1
                )
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
    def read_fasta(self, file_path, return_names = False):
        """
        Read sequences and labels from a FASTA file.
    
        Parameters
        ----------
        file_path : str
            Path to the FASTA file.
        return_names : bool, optional
            If True, returns sequence names along with sequences and labels.
    
        Returns
        -------
        tuple
            A tuple containing:
            - seqs (list of str): Sequences from the FASTA file.
            - labels (list of float): Corresponding labels.
            - names (list of str, optional): Names of sequences (if return_names is True).
        """
        fa = pyfastx.Fastx(file_path, comment=True)
        names, seqs, labels = [], [], []
        for name, seq, label in fa:
            seqs.append(seq)
            names.append(name.split(":")[0])
            labels.append(np.float32(label))

        if return_names:
            return names, seqs, labels
        else:
            return seqs, labels
        
    #for random enhancer, genomic promoter and capture promoter data
    def task_with_default_split(self, task, split):
        file_path = f"{self._data_path}{self.prefix}{task}{split}.fasta.gz"
        seqs, labels = self.read_fasta(file_path)
        return {"targets": labels, "seq": seqs}

    # for genomic Enhancer and ATACseq data
    def task_with_various_split(self, task, split):
        file_path = f"{self._data_path}{self.prefix}{task}all_chr_file.fasta.gz"
        names, seqs, labels = self.read_fasta(file_path, True)
        data = pd.DataFrame({"chr": names, "seq": seqs, "targets": labels})
        data = data[data.chr.isin(split)].reset_index(drop=True)
        return {"targets": data.targets.to_numpy(), "seq": data.seq.to_numpy()}

    def task_binary(self, task, binary_class, split): 
        binary_train = ["promoter_from_input", "enhancer_permutated", "enhancer_from_input"]
        if split == "train":
            if binary_class is not None:
                if binary_class not in binary_train:
                    raise ValueError(f"'binary_class' must be one of {binary_train} for training")
                else:
                    if binary_class.split("_")[0] == "promoter":
                        file_path_prom = f"{self._data_path}{self.prefix}{task}{split}_{binary_class}.fasta.gz"
                        file_path_enh = f"{self._data_path}{task}{split}_enhancer.fasta.gz"
                    else: 
                        file_path_prom = f"{self._data_path}{self.prefix}{task}{split}_promoter.fasta.gz"
                        file_path_enh = f"{self._data_path}{task}{split}_{binary_class}.fasta.gz"
                                                
                    seqs_prom, labels_prom = self.read_fasta(file_path_prom)
                    seqs_enh, labels_enh = self.read_fasta(file_path_enh)
                    
                    print(f"using train {binary_class}")
                    
                    return {"targets": labels_prom, "seq": seqs_prom, "seq_enh": seqs_enh}
                    
            elif binary_class is None:
                pass 
                
        file_path_prom = f"{self._data_path}{self.prefix}{task}{split}_promoter.fasta.gz"
        seqs_prom, labels_prom = self.read_fasta(file_path_prom)
        
        file_path_enh = f"{self._data_path}{self.prefix}{task}{split}_enhancer.fasta.gz"
        seqs_enh, labels_enh = self.read_fasta(file_path_enh)
        
        print(f"using {split}")
        
        return {"targets": labels_prom, "seq": seqs_prom, "seq_enh": seqs_enh}
        
############################### Task diff expr + Extract sequences for diff expression ###################################
    ########################### DEPRECATED #############################################################
    def task_diff_exp(self, task, split, length, bed_file="RNA-seq_diffExpr_GP5dvsHepG2_from_tpmMeanMin1_with_coords_ENSG.bed.gz"):
        file_path = os.path.join(self._data_path, task, bed_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path, sep="\t", header=None, names=["chr", "start", "end", "name", "targets"])
        df.loc[df['chr'] == "chrMT", "chr"] = "chrM" # replace chrMT wit chrM
        
        # Download FASTA if necessary
        fasta_file = os.path.join(self._data_path, task, "hg19.fa")
        if not os.path.exists(fasta_file):
            download_file("https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz", fasta_file + ".gz")
            subprocess.run(["gunzip", fasta_file + ".gz"], check=True)

        # Load FASTA
        try:
            ref = pyfaidx.Fasta(fasta_file)
        except Exception as e:
            raise IOError(f"Error loading FASTA file: {fasta_file}. {str(e)}")
    
        # Get chromosome sizes
        chromosome_sizes = {k: len(v) for k, v in ref.items()}

        # Extract sequences
        def calculate_sequence(row):
            if row.chr not in chromosome_sizes:
                raise KeyError(f"Chromosome {row.chr} not found in chromosome sizes.")
            return self.get_sequence(ref, row.chr, chromosome_sizes[row.chr], length, row.start, row.end)
    
        df["seq"] = df.apply(calculate_sequence, axis=1)
        df = df[df.chr.isin(split)].reset_index(drop=True)
    
        return {"targets": df.targets.to_numpy(), "seq": df.seq.to_numpy()}

    def get_sequence(self, file_pyfaidx, chromosome, chr_size, length, start, end):
        """
        Extract sequence from a FASTA file with padding to a fixed length.
    
        Parameters
        ----------
        file_pyfaidx : pyfaidx.Fasta
            Opened FASTA file.
        chromosome : str
            Chromosome name.
        chr_size : int
            Size of the chromosome.
        length : int
            Desired length of the sequence.
        start : int
            Start position.
        end : int
            End position.
    
        Returns
        -------
        str
            Extracted sequence.
        """
        diff = length - (end - start + 1)
        start = max(0, start - diff//2 - diff%2)
        end = min(end + diff//2, chr_size) 
        
        try:
            seq = file_pyfaidx.get_seq(chromosome, start, end)
        except Exception as e:
            raise ValueError(f"Error extracting sequence for {chromosome}:{start}-{end}. {str(e)}")
    
        return str(seq)
################################ Split Parsing #################################
    def split_parse(self, split: list[int | str] | int | str, is_split_default: bool) -> str | list[str]:
        """
        Parses the input split and returns a list of chromosome names or default splits.
    
        Parameters
        ----------
        split : list[int | str] | int | str
            The split identifier(s). Can be:
            - str: 'train', 'val', 'test', chromosome name ('chrX') or number ('X', '1-22').
            - int: Chromosome number (1-22).
            - list[int | str]: List of chromosome identifiers.
        is_split_default : bool
            If True, uses predefined splits ('train', 'val', 'test').
    
        Returns
        -------
        list[str]
            A list of chromosome names or split identifiers.
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
                    raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")
                split = split  # Make it a list to maintain uniformity
            else:
                raise ValueError("Invalid split value. Expected 'train', 'val', or 'test'.")
        
        else:
            # Chromosomal split logic (specific chromosomes for train, val, test)
            split_chr = {
                "train": ['chr1', 'chr3', 'chr5', 'chr7', 'chr9', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
                "val": ["chr4", "chr6", "chr8"],
                "test": ["chr2", "chr10", "chr11"]
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
                    raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test', range 1-22, 'X' or name 'chrx', where x is number 1-22 or X.")
            
            elif isinstance(split, int):
                split = [convert_to_chr(split)]
            
            elif isinstance(split, list):
                split = [convert_to_chr(item) for item in split]
    
                # Validate all items in the list
                for item in split:
                    if not item.startswith("chr") or item[3:] not in list_of_chr:
                        raise ValueError(f"Invalid split value: {item}. Must be in range 1-22 or 'X'.")
    
        return split

        