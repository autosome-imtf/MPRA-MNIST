import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
from .info import INFO

from .mpradataset import MpraDataset

class MassiveStarrDataset(MpraDataset):
    
    cell_types = ['HepG2', 'K562', 'WTC11']
    flag = "MassiveStarrDataset"
    
    def __init__(self,
                 task: str
                 split: str | List[int] | int,
                 transform = None,
                 target_transform = None,
                ):
        """
        Attributes
        ----------
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split)
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.info = INFO[self.flag]
        self.task = task

        self.ds = self.load_data(self.split, self.task)

    def load_data(self, split, task):
        tasks = ["RandomEnhancer", "GenomeEnhancer", "GenomePromoter", "DifferentialExpression", "CapturePromoter", "AtacSeq", "binary"]
        files = {"RandomEnhancer":[
            "GP5d_ranEnh_GRE1.170.noDup.fixednames.bothclasses.train.fasta.gz",
            "GP5d_ranEnh_GRE1.170.noDup.fixednames.bothclasses.val.fasta.gz",
            "GP5d_ranEnh_GRE1.170.noDup.fixednames.test.fasta.gz",
            "InputLib_SynEnh.170.noDup.fixednames.test.fasta"],
                "GenomeEnhancer":[
                    "train_bothstrands_bothclasses_091120.fasta.gz",
                    "val_bothclasses_091120.fasta.gz",
                    "test_positive_280120.fasta.gz",
                    "test_negative_coverage_balanced.fasta.gz"],
                 "GenomePromoter":[
                     "Hs_EPDnew_006_hg19_100+20_train_bothclasses.fasta.gz",
                     "Hs_EPDnew_006_hg19_100+20_val_bothclasses.fasta.gz",
                     "Hs_EPDnew_006_hg19_100+20_test_positive.fasta.gz",
                     "Hs_EPDnew_006_hg19_100+20_test_negative.fasta.gz",
                     "EPD_overlap_CAGE_test_positive_slop_b500.fasta"], #what is it?
                 "DifferentialExpression":[
                     "RNA-seq_diffExpr_GP5dvsHepG2_from_tpmMeanMin1_with_coords_ENSG.bed.gz"],
                 "CapturePromoter":[
                     "promoter_upstream_train_bothclasses.fasta.gz",
                     "promoter_upstream_val_bothclasses.fasta.gz",
                     "test_positive.fasta.gz",
                     "test_negative.fasta.gz"],
                 "AtacSeq":[
                     "GP5d_ATAC_train_bothclasses.fasta.gz",
                     "GP5d_ATAC_val_bothclasses.fasta.gz",
                     "test_positive.fasta.gz",
                     "test_negative.fasta.gz"],
                 "binary":[
                     "paired_R1_train.fasta.gz",
                     "paired_R1_val.fasta.gz",
                     "GP5d_prom_enh_paired_R1_test.fasta.gz",
                     "Input_GP5d_prom_enh_paired_R1_test.fasta.gz",
                     "paired_R2_train.fasta.gz",
                     "paired_R2_val.fasta.gz",
                     "GP5d_prom_enh_paired_R2_test.fasta.gz",
                     "Input_GP5d_prom_enh_paired_R2_test.fasta.gz"]
                    }
        
    def split_parse(self, split: list[int] | int | str) -> list[int]:
        '''
        Parses the input split and returns a list of folds.
        '''
        
        split_default = {"train" : 0.7, 
                         "val" : 0.15, 
                         "test" : 0.15
                        } # default split of data
        
        return split