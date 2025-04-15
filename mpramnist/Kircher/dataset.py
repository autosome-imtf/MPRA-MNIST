import pandas as pd
import numpy as np
from typing import List, T, Union
import torch
import os
import pyfaidx
import subprocess

from mpramnist.mpradataset import MpraDataset

class KircherDataset(MpraDataset):
    
    FLAG = "Kircher"
    ALL_PROMOTERS = ['BCL11A', 'F9', 'FOXE1', 'GP1BA', 'HBB', 'HBG1', 'HNF4A', 'IRF4',
       'IRF6', 'LDLR', 'LDLR.2', 'MSMB', 'MYCrs11986220', 'MYCrs6983267',
       'PKLR-24h', 'PKLR-48h', 'RET', 'SORT1', 'SORT1-flip', 'SORT1.2',
       'TCF7L2', 'TERT-GAa', 'TERT-GBM', 'TERT-GSc', 'TERT-HEK', 'UC88',
       'ZFAND3', 'ZRSh-13', 'ZRSh-13h2']
    def __init__(self,
                 split: str = "test",
                 length: int = 230, # length of cutted sequence 
                 promoters: list[str] | str =  ["F9","LDLR.2","LDLR","PKLR-24h","PKLR-48h","SORT1.2","SORT1"],
                 transform = None,
                 target_transform = None,
                 root = None,
                ):
        """
        Attributes
        ----------
        split : str
            Specifies how to split the data. Currently only "test" is supported.
        length : int, optional
            Length of the sequence for the differential expression experiment. Must be positive.
        promoters : Union[list[str], str], optional
            List of promoter elements to include, or "all" for all promoters.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)
        
        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"

        if (isinstance(promoters, str) and promoters.lower() != "all") or \
           (isinstance(promoters, list) and not all(p in self.ALL_PROMOTERS for p in promoters)):
            raise ValueError("Invalid promoters list")
    
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"Parameter 'length' must be natural integer, not {length}.")
        self.length = length
        
        try:
            file_name = self.prefix + "GRCh38_ALL" + '.tsv'
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Process data
        df.Chromosome = 'chr' + df.Chromosome.astype(str)
        df.Position = df.Position.astype(int)
        self.ds = df.dropna().reset_index(drop=True)
        target_column = "Value"
        
        # Handle FASTA file
        fasta_file = self._setup_fasta_file()
        
        # Load FASTA
        try:
            ref = pyfaidx.Fasta(fasta_file)
        except Exception as e:
            raise IOError(f"Error loading FASTA file {fasta_file}: {str(e)}") from e

        # Extract sequences
        self.ds["seq_alt"] = self.ds.apply(
            lambda row: self.get_sequence(
                file_pyfaidx=ref,
                chromosome=row.Chromosome,
                length=self.length,
                pos=row.Position,
                ref=row.Ref,
                alt=row.Alt
            ),
            axis=1
        )
        
        self.ds["seq_ref"] = self.ds.apply(
            lambda row: self.get_sequence( # get reference sequence
                file_pyfaidx=ref,
                chromosome=row.Chromosome,
                length=self.length,
                pos=row.Position,
                ref=row.Ref,
                alt=row.Ref # here not alternative variant
            ),
            axis=1
        )
            
        # Filter by promoters
        if isinstance(promoters, str) and promoters.lower() == "all":
            pass  # include all promoters
        else:
            self.ds = self.ds[self.ds.Element.isin(promoters)]
            
        self.ds = self.ds.dropna().reset_index(drop=True)
        
        targets = self.ds[target_column].to_numpy()
        seq_alt = self.ds.seq_alt.to_numpy()
        seq_ref = self.ds.seq_ref.to_numpy()
        self.ds = {"targets" : targets, "seq" : seq_ref, "seq_alt" : seq_alt} 

    def _setup_fasta_file(self) -> str:
        """Ensure FASTA file exists and is ready for use."""
        fasta_file = os.path.join(self._data_path, "hg38.fa")
        
        if not os.path.exists(fasta_file):
            url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
            try:
                # Download and decompress
                subprocess.run(["wget", url, "-O", f"{fasta_file}.gz"], check=True)
                subprocess.run(["gunzip", fasta_file + ".gz"], check=True)
            except subprocess.CalledProcessError as e:
                raise IOError(f"Failed to download/decompress FASTA file: {str(e)}") from e
        else:
            #print("FASTA file already exists. Skipping download.")
            pass
        
        return fasta_file

    def get_sequence(self, file_pyfaidx, chromosome: str, length: int, pos: int, ref: str, alt: str) -> str:
        """
        Extract sequence from a FASTA file with padding to a fixed length.
        """
        # Input validation
        if not isinstance(ref, str) or len(ref) != 1:
            raise ValueError(f"Reference nucleotide should be single character, got {ref}")
        
        # Verify reference nucleotide matches expected
        observed_ref = str(file_pyfaidx.get_seq(chromosome, pos, pos)).upper()
        if observed_ref != ref.upper():
            return None
        
        half_len = length // 2
        start = pos - half_len 
        end = pos + half_len 
        
        if length % 2 == 0:
            end -= 1
    
        try:        
            ref_pos_in_seq = half_len
            
            if alt == "-":
                # Handle deletion
                seq = str(file_pyfaidx.get_seq(chromosome, start, end + 1))
                modified_seq = seq[:ref_pos_in_seq] + seq[ref_pos_in_seq + 1:]
            else:
                # Handle substitution or insertion
                seq = str(file_pyfaidx.get_seq(chromosome, start, end))
                modified_seq = seq[:ref_pos_in_seq] + alt + seq[ref_pos_in_seq + 1:]
            
            return modified_seq
        except Exception as e:
            raise ValueError(f"Error processing {chromosome}:{pos}-{ref}>{alt}: {str(e)}") from e
            