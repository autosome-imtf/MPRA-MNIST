import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO
import pyfastx
from .mpradataset import MpraDataset

class MassiveStarrDataset(MpraDataset):
    
    flag = "MassiveStarrSeqDataset"
    tasks = ["randomenhancer", "genomicenhancer", "genomicpromoter", "differentialexpression", "capturepromoter", "atacseq", "binary"]
    
    def __init__(self,
                 task: str,
                 split: str | List[int] | int,
                 transform = None,
                 target_transform = None,
                ):
        """
        Attributes
        ----------
        task:
        split:
        transform : callable, optional
            Transformation applied to each sequence objects.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        self.cell_types = None
        self._cell_type = None
        
        super().__init__(split)
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.info = INFO[self.flag]

        self.ds = self.load_data(split, self.task)

    def load_data(self, split, task):
        task = self.define_task(task)
        
        # these tasks have default split, so we can not change it
        if self.task.lower() in ["randomenhancer", "genomicpromoter", "capturepromoter"]:
            is_split_default = True
            self.split = self.split_parse(split, is_split_default)
            ds = self.task_with_default_split(task, self.split)
            
        # these tasks could be splitted to chr so they have another file structure
        elif self.task.lower() in ["genomicenhancer", "atacseq"]:
            is_split_default = False
            self.split = self.split_parse(split, is_split_default)
            ds = self.task_with_various_split(task, self.split)
        return ds

    #for random enhancer, genomic promoter and capture promoter data
    def task_with_default_split(self, task, split):
        fa = pyfastx.Fastx(f'{self._data_path}{task}{split}.fasta.gz', comment=True)
        seqs = []
        labels = []
        for name,seq,label in fa:
            seqs.append(seq)
            labels.append(np.float32(label))
        
        return {"targets": labels, "seq": seqs}

    # for genomic Enhancer and ATACseq data
    def task_with_various_split(self, task, split):
        fa = pyfastx.Fastx(f'{self._data_path}{task}all_chr_file.fasta.gz', comment=True)
        seqs = []
        names = []
        labels = []
        for name,seq,label in fa:
            seqs.append(seq)
            names.append(name.split(":")[0])
            labels.append(np.float32(label))
        data = {"chr" : names, "seq" : seqs}
        df = pd.DataFrame(data)
        df["targets"] = labels
        
        df = df[df.chr.isin(split)].reset_index(drop=True)
        
        seqs = df.seq.to_numpy()
        targets = df.targets.to_numpy()

        return {"targets": targets, "seq": seqs}

    def task_diff_exp(self, task, split): # Not ready
        return None

    def task_binary(self, task, split): # Not ready
        return None
        
    def define_task(self, name):
        tasks_path = ["./ranEnh/", "./genEnh/", "./genProm/", "./diffExp/", "./CaptProm/", "./ATACSeq/", "./binary/"]
        if name.lower() in self.tasks:
            return tasks_path[self.tasks.index(name.lower())]
        
    def split_parse(self, split: list[int] | int | str, is_split_default) -> list[int]:
        '''
        Parses the input split and returns a list of folds.
        '''
        
        if is_split_default:

            split_default = {"train" : "train", 
                         "val" : "val", 
                         "test" : "test"
                        }
            
            if isinstance(split, str):
                if split not in split_default:
                    raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test'.")
            else:
                raise ValueError("Invalid split value. Expected 'train', 'val', or 'test'")
                
        else:

            split_chr = {"train" : ['chr1','chr3','chr5','chr7','chr9','chr11','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX'],
                     "val": ["chr4", "chr6", "chr8"],
                     "test" : ["chr2", "chr10", "chr11"]
}

            list_of_chr = [str(i) for i in range(23)] + ["X"]
            list_of_named_chr = ["chr" + i for i in list_of_chr]
            
            if isinstance(split, str):
                if split in split_chr:
                    split = split_chr[split]
                elif split in list_of_chr:
                    split = ["chr"+split]
                elif split in list_of_named_chr:
                    split = [split]
                else:
                    raise ValueError(f"Invalid split value: {split}. Expected 'train', 'val', or 'test', range 1-22, 'X' or name 'chrx', where x is number 1-2 or X.")
                
            # int to list for unified processing
            elif isinstance(split, int):
                split = ["chr"+str(split)]
                
            # Check the range of values for a list
            elif isinstance(split, list):
                result = []
                for item in split:
                    if isinstance(item, int) and 1 <= item <= 22:
                        result.append("chr" + str(item))
                    elif isinstance(item, str) and item in list_of_chr:
                        result.append("chr" + item)
                    elif isinstance(item, str) and item in list_of_named_chr:
                        result.append(item)
                    else:
                        raise ValueError(f"Invalid split value: {item}. Must be in range 1-22 or 'X'.")
                        
                split = result  # Ensure final result is clean and validated
                
        return split