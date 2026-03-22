import pandas as pd
import numpy as np
import os

from mpramnist.dataclass import seqobj, ScalarFeature, Categorial
from typing import Callable, ClassVar

from mpramnist.mpradataset import MpraDataset

class FromelDataset(MpraDataset):
    
    CONSTANT_LEFT_FLANK: ClassVar[str] = "AGGACCGGATCAACT"  # required for each sequence
    CONSTANT_RIGHT_FLANK: ClassVar[str] = "CATTGCGTGAACCGA"  # required for each sequence
    LEFT_FLANK: ClassVar[str] = "GGCCCGCTCTAGACCTGCAGG" 
    RIGHT_FLANK: ClassVar[str] = (
        "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"
    )

    SUBDATASETS: ClassVar[dict[str, list[str]]] = {
        "HSPC_TRAINVALID" : [
            'HSPC.libB.DATA',
            'HSPC.libB.CONTROLS.GENERAL',
            'HSPC.libB.CONTROLS.TP53',
            'HSPC.libA.DATA',
            'HSPC.libC.DATA',
            'HSPC.libC.CONTROLS.GENERAL',
            'HSPC.libC.CONTROLS.TP53',
            'HSPC.libF.DATA',
            'HSPC.libF.CONTROLS.GENERAL'
            'HSPC.libF.CONTROLS.TP53'],
        "K562_TRAINVALID": [
            'K562.libC.minP.tra.DATA',
            'K562.libC.minP.tra.CONTROLS.GENERAL',
            'K562.libC.minP.tra.CONTROLS.TP53',
            'K562.libA.minP.tra.DATA',
            'K562.libB.minP.tra.DATA',
            'K562.libB.minP.tra.CONTROLS.GENERAL',
            'K562.libB.minP.tra.CONTROLS.TP53'],
        "TEST_GENOMIC": ['HSPC.libG.DATA'],
        "TEST_SYNTHETIC": ['HSPC.libH.DATA'],
        "TEST_GENERATED": ['HSPC.libD.DATA'],
        "K562_INT": [
            'K562.libB.minP.int.DATA',
            'K562.libB.minP.int.CONTROLS.GENERAL',
            'K562.libB.minP.int.CONTROLS.TP53'
        ],
        "K562_minCMV": [
            'K562.libB.minCMV.tra.DATA',
            'K562.libB.minCMV.tra.CONTROLS.GENERAL',
            'K562.libB.minCMV.tra.CONTROLS.TP53'
        ]
    }

    HSPC_TARGETS: list[str] = [
        'State_1M',
        'State_2D',
        'State_3E',
        'State_4M',
        'State_5M',
        'State_6N',
        'State_7M',
    ]

    HSPC_BATCHES = {
        'libA': 0,
        'libB': 1,
        'libC': 2,
        'libH': 2,
        'libD': 3,
        'libF': 4,
        'libG': 4
    }


    CELL_TYPES = ["HSPC", "K562"]
    FLAG: ClassVar[str] = 'Fromel2025'

    def __init__(self,
                 split: list[int] | str | int, # folds 
                 cell_type: str = 'HSPC',
                 upper_seq: bool = True, # return sequence in upper-case or return in mixed-case format showing motif placement for most sequences in the dataset
                 targets: list[str] | str | None = None,
                 add_batch_info: bool = True,
                 state_level_value: str = 'mean.norm.adj',
                 transform: Callable | None = None,
                 target_transform: Callable | None = None,
                 root: str | None = None
                ):

        super().__init__(split, root)
        self.cell_type = cell_type 
        self.split = split
        self.state_level_value = state_level_value
        self.upper_seq = upper_seq
        self.targets = self.process_targets(targets)
        subdataset_name, folds = self.split_parse(split)

        subdataset_cols = self.SUBDATASETS[subdataset_name]

        self.prefix = self.FLAG + "_"

        try:
            file_name = self.prefix + "Fromel" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            data = pd.read_csv(file_path, sep="\t", dtype={'fold': 'string'})
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.upper_seq:
            data['Seq'] = data['Seq'].str.upper()

        if self.cell_type == 'HSPC':
            data = data[data['source'].str.startswith('HSPC')]
            data['batch'] = data['source'].apply(lambda x: self.HSPC_BATCHES[x.split('.', 2)[1]])
            self.add_batch_info = add_batch_info
        else:
            self.add_batch_info = False

        data = data[data['source'].isin(subdataset_cols)]
        data = data[data['fold'].isin(folds)]

        index = ["Seq", "CRS", "Library"]
        if self.add_batch_info:
            index.append('batch')

        data = data.pivot(index=index,
           columns="clusterID",
           values=self.state_level_value).reset_index()
        
        if self.split == 'generated' or self.split == 'genome':
            data['State_5M'] = np.nan
            # Library F and two additional libraries (D and G, see below)
            # were measured in six cell states, since differences between 
            # early and late monocyte precursors in Library A and B were minimal

        self.data = data
        self.seqs = data['Seq'].values
        if self.add_batch_info:
            self.batch = data['batch'].values
        
        self.target = data[self.targets].values

        self.name_for_split_info = ''
        self.info = {'task': 'regression', 'description': 'TODO'}
        self.transform = transform
        self.target_transform = target_transform

    def process_targets(self, targets: list[str] | str | None):
        if targets is None or (isinstance(targets, str) and targets == 'all'):
            if self.cell_type == 'HSPC':
                targets = list(self.HSPC_TARGETS)

            elif self.cell_type == 'K562':
                targets = ['State_9K']
            else:
                raise Exception(f'Wrong cell type: {self.cell_type}')
        else:            
            if isinstance(targets, str):
                targets = [targets]
            for ta in targets:
                if ta not in self.HSPC_TARGETS:
                    raise Exception(f'Wrong target {ta} for cell type {cell_type}')
        return targets 

    def split_parse(self, split: str | list[int] | int) -> tuple[str, list[str]]:
        """
        Parse split parameter
        """
        # Process string input
        if split == 'train':
            folds = [0,1,2,3,4,5,6,7,8]
        elif split == 'valid':
            folds = [9]
        elif split == 'test':
            folds = [10]
        elif split == 'genome':
            folds = 'test'
            if self.cell_type == 'HSPC':
                subdataset = 'TEST_GENOMIC'
            elif self.cell_type == 'K562':
                raise Exception(f'Genomic sequences were not measured for {self.cell_type}')
            else:
                raise Exception(f'Wrong {self.cell_type}')
        elif split == 'synthetic':
            folds = 'test'
            if self.cell_type == 'HSPC':
                subdataset = 'TEST_SYNTHETIC'
            elif self.cell_type == 'K562':
                raise Exception(f'Synthetic sequences were not measured for {self.cell_cell_typeline}')
            else:
                raise Exception(f'Wrong {self.cell_type}')
        elif split == 'generated':
            folds = 'test'
            if self.cell_type == 'HSPC':
                subdataset = 'TEST_GENERATED'
            elif self.cell_type == 'K562':
                raise Exception(f'Generated sequences were not measured for {self.cell_type}')
            else:
                raise Exception(f'Wrong {self.cell_type}')
        elif isinstance(split, int):
            folds = [split]
        elif isinstance(split, list):
            for i in split:
                if not isinstance(i, int):
                    raise Exception(f'Wrong fold value: {i}')
            folds = split
        else:
            raise Exception(f'Wrong split: {split}')
        
        if isinstance(folds, list): # list of ints 
            if self.cell_type == 'HSPC':
                subdataset = 'HSPC_TRAINVALID'
            elif self.cell_type == 'K562':
                subdataset = 'K562_TRAINVALID'
            else:
                raise Exception(f'Wrong {self.cell_type}')
            folds = [str(i) for i in folds]
        else: # str
            folds = [folds]
            
        return subdataset, folds

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):  
        sequence = self.seqs[idx]
        seq = seqobj(seq=sequence,
                     scalars={},
                     vectors={}, 
                     add_feature_channel=self.add_batch_info)
        if self.add_batch_info:
            seq.scalars['batch'] = ScalarFeature(self.batch[idx], tp=Categorial(levels=list(self.HSPC_BATCHES.keys())))

        if self.transform is not None:
            seq = self.transform(seq)

        target = self.target[idx].astype(np.float32)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return seq.seq, target
