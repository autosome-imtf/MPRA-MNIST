import torch
import torch.nn as nn

from mpramnist.mpradataset import MpraDataset
from mpramnist.dataclass import seqobj, VectorDsFeature, ScalarDsFeature
from mpramnist.genome import download_genome
from typing import Callable, ClassVar
import numpy as np
import pandas as pd
import os
import pyfaidx

class BarbadillaMartinez2026(MpraDataset):
    LEFT_FLANK: ClassVar[str] = (
        "CAGTGAT" # for focused dataset, part of adapter
    )
    RIGHT_FLANK: ClassVar[str] = (
        "CACGACG" # for focused dataset, part of adapter
    )

    FEATURE_TYPES: ClassVar[dict[str, int]] = {
        'promoter': 0,
        'enchancer': 1
    }

    CELLLINE_2_LIBRARY: dict[str, str] = {
       'AGS': 'focused_v1', 
       'HAP1': 'focused_v1',
       'K562': 'focused_v2',
       'HepG2': 'focused_v2', 
       'HEK293': 'focused_v2',
       'HCT116': 'focused_v2', 
       'LNCaP': 'focused_v2',
       'MCF7': 'focused_v2', 
       'U2OS': 'focused_v2', 
       'K562_HeatShock': 'focused_v2', 
       'K562_Control_HeatShock': 'focused_v2',
       'HepG2_Nutlin3a': 'focused_v2',
       'HepG2_Control_Nutlin3a': 'focused_v2',
       'K562_PMA': 'focused_v2',
       'K562_Control_PMA': 'focused_v2',
       'K562_genomewide': 'genomewide',
       'HepG2_genomewide': 'genomewide',
    }
    
    LIBRARY_2_NORMALIZATION: dict[str, str] = {
        'focused_v1': 'pDNA_pHY3_T2',
        'focused_v2': 'pDNA_T1_T2_sum',
        'genomewide': 'iPCR'
    }
    
    NORMALIZATION_2_THRESHOLD: dict[str, str] = {
        'pDNA_pHY3_T2': 10,
        'pDNA_T1_T2_sum': 10,
        'iPCR': 0,
    }

    GENOME_MAPPING: ClassVar[dict[str, int]] = {'GM18983': 0, 'HG03464': 1, 'HG02601': 2, 'HG01241': 3}
    
    FLAG: ClassVar[str] = 'BarbadillaMartinez2026'

    def __init__(self,
                 split: list[int], # folds 
                 library_type: str = 'focused',
                 cell_line: str = 'AGS', 
                 transform: Callable | None = None,
                 target_transform: Callable | None = None,
                 normalization_treshold: int | None = None,
                 feature_types: list[str] =['promoter'],
                 genomes: list[str] | str = 'all', # which genomes should be used, applicable for genomewide libraries only
                 filter_ambiguous: bool = True, # whether to filter out snps with ambiguous bases, applicable for genomewide libraries only
                 filter_heterozygotes: bool = True, # whether to filter out snps with diffent alleles, applicable for genomewide libraries only
                 root: str | None = None
                ):

        super().__init__(split, root)
        self.split = split
        folds = self.split_parse(split)

        version = self.CELLLINE_2_LIBRARY.get(cell_line, None)
        if version is None:
            raise Exception(f'Wrong cell line provided: {cell_line}')
        normalization_column = self.LIBRARY_2_NORMALIZATION[version]
        normalization_threshold = self.NORMALIZATION_2_THRESHOLD[normalization_column]
        
        data = self.retrieve_data(folds, version)
        self.normalization_column = normalization_column
        self.normalization_threshold = normalization_threshold
        
        data = data[data[normalization_column] >= normalization_threshold]

        feature_ids = self.map_feature_types(feature_types)
        data = data[data['FEATtype'].isin(feature_ids)]

        if self.library_is_genomewide(cell_line):
            self.account_for_snps = True
            genome_ids = self.map_genomes(genomes)
            data = data[data['genome'].isin(genome_ids)]

            data['SNPrelpos'] = data['SNPrelpos'].str.split(',').apply(lambda x: [int(c) for c in x] if isinstance(x, list) else x)
            data['SNPbase'] = data['SNPbase'].str.split(',')
            
            if filter_ambiguous:
                mask = data['SNPbase'].apply(lambda x: all(c in 'ATGC' for c in x) if isinstance(x, list) else True)
                data = data[mask]

            data = self.infer_heterozytes(data)
            if filter_heterozygotes:
                data = data[~data['hetero']]

        else:
            self.account_for_snps = False # where is no information on snps in focused libraries 

        data['start'] = data['start'] # to zero-indexed
        self._data = data
        self.chr = data['chr'].values
        self.start = data['start'].values 
        self.end = data['end'].values
        self.strand = data['strand'].values
        if self.library_is_genomewide(cell_line):
            self.snps = data['SNPbase'].values
            self.snps_poses = data['SNPrelpos'].values

        target_column = self.get_target_column(cell_line)
        self.target = data[target_column].values

        self.lengths = self.end - self.start
        self.genome_path = download_genome('hg19')
        self.faidx_genome = None # to be initialized in __getitem__ to avoid race condition 
        
        self.name_for_split_info = ''
        self.info = {'task': 'regression', 'description': 'TODO'}
        self.transform = transform
        self.target_transform = target_transform
        self.feat = data['FEAT']

    def library_is_genomewide(self, cell_line) -> bool:
        return 'genomewide' in cell_line

    def map_genomes(self, genomes: str | list[str]) -> list[str]:
        if isinstance(genomes, str):
            if genomes == 'all':
                genomes = self.get_available_genomes()
            else:
                genomes = [genomes]

        gen_ids = []
        for gen in genomes:
            gid = self.GENOME_MAPPING.get(gen, -1)
            if gid == -1:
                raise Exception(f'Wrong genome id: {gen}')
            gen_ids.append(gid)
        return gen_ids

    def get_default_normalization_threshold(self, column) -> int:
        if column == 'pDNA_pHY3_T2' or column == 'pDNA_T1_T2_sum':
            normalization_treshold = 10
        elif column == 'iPCR':
            normalization_treshold = 0
        else:
            raise Exception(f'No defaults for normalization column: {normalization_column}')
        return normalization_treshold

    def get_target_column(self, cell_line) -> str:
        if cell_line.endswith('_genomewide'):
           cell_line = cell_line.replace('_genomewide', '')
        return  f"Log2RPM_{cell_line}"


    @classmethod
    def get_available_genomes(cls):
        return list(cls.GENOME_MAPPING.keys())

    @classmethod
    def get_available_cell_lines(self) -> list[str]:
        return list(self.CELLLINE_2_LIBRARY.keys())

    def retrieve_data(self, folds: list[int], version):
        data = []
        for fold in folds:
            if isinstance(fold, int):
                fold_name = f'fold{fold}'
            else:
                fold_name = fold
            file_name = f"{self.FLAG}_{version}_library_{fold_name}.bed"
            try:         
                self.download(self._data_path, file_name)   
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {file_path}")
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
            data.append(df)
        data = pd.concat(data, axis=0)
        return data 

    def split_parse(self, split: str | list[int] | int) -> list[int]:
        """
        Parse split parameter and return list of fold numbers.

        Parameters
        ----------
        split : Union[str, List[int], int]
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
        [0, 1, 2, 3]
        >>> split_parse([1, 2, 3])
        [1, 2, 3]
        >>> split_parse(5)
        [5]
        """
        # Process string input
        if isinstance(split, str):
            split_default = self.info['split']
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
                if not (0 <= spl <= 4):
                    raise ValueError(f"Fold {spl} not in range 0-4.")

        if isinstance(split, str):
            split = [split]
                    
        return split

    def map_feature_types(self, feature_types: list[str]) -> list[int]:
        assert len(feature_types) >= 1, 'Must select at least one feature type'
        feature_type_ids = []
        for ft in feature_types:
            ft_id = self.FEATURE_TYPES.get(ft, -1)
            if ft_id == -1:
                raise Exception('Wrong feature type')
            feature_type_ids.append(ft_id)
        return feature_type_ids

    def infer_heterozytes(self, data: pd.DataFrame) -> pd.DataFrame:
        from collections import defaultdict

        hetero = []
        snp_relpos = []
        snp_base = []
        for _, xs, ys in data[['SNPrelpos', 'SNPbase']].itertuples():
            if not isinstance(xs, list) and np.isnan(xs):
                het = False
            else:
                d = defaultdict(list)
                for p, n in zip(xs,ys):
                    d[p].append(n)
                if len(d) == len(xs):
                    het = False
                elif all( len(set(x)) == 1 for x in d.values()):
                    het = False
                else:
                    het = True
                    xs = []
                    ys = []
                    for k in sorted(d.keys()):
                        key = tuple(sorted(set(d[k])))
                        iupac = IUPAC_MAP_REV[key]
                        xs.append(k)
                        ys.append(iupac)
            hetero.append(het)
            snp_relpos.append(xs)
            snp_base.append(ys)
        data = data.copy()
        data['SNPrelpos'] = snp_relpos
        data['SNPbase'] = snp_base
        data['hetero'] = hetero
        return data 

    def __len__(self):
        return len(self.start)

    def __getitem__(self, idx):

        if self.faidx_genome is None:
            self.faidx_genome = pyfaidx.Fasta(self.genome_path, 
                                              as_raw=False, 
                                              sequence_always_upper=False,
                                              one_based_attributes=False,
                                              strict_bounds=True,
                                              rebuild=True)

        ch = self.chr[idx]
        s = self.start[idx]
        e = self.end[idx]
        strand = self.strand[idx]

        sequence = self.faidx_genome.get_seq(ch, s, e)
        if self.account_for_snps:
            snps = self.snps[idx]      
            poses = self.snps_poses[idx]
            #print(snps, poses)
            if isinstance(snps, list):
                sequence.seq = self.apply_snps(sequence.seq, poses, snps) 
            
        if strand == '-':
            sequence = sequence.reverse.complement
        sequence = sequence.seq.upper()
        
        seq = seqobj(seq=sequence,
                     scalars={},
                     vectors={})

        if self.transform is not None:
            seq = self.transform(seq)

        target = self.target[idx].astype(np.float32)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return seq.seq, target

    def apply_snps(self, seq: str, poses: list[int], snps: list[str]) -> str:
        """
        Apply SNP modifications to a sequence.

        Parameters
        ----------
        seq : str
            Original sequence.
        poses : list[int]
            Positions where SNPs occur (0-based).
        snps : list[str]
            Replacement nucleotides.

        Returns
        -------
        str
            Modified sequence.
        """

        if len(poses) != len(snps):
            raise ValueError("poses and snps_str must have the same length")
        seq_list = list(seq)

        for pos, snp in zip(poses, snps):
            seq_list[pos] = snp

        return "".join(seq_list)

