import pandas as pd
import os
from typing import List, Union, Dict, Optional
import pyfaidx
import subprocess
import bioframe as bf

from mpramnist.mpradataset import MpraDataset


class KircherDataset(MpraDataset):
    """
    Attributes
    ----------
    FLAG : str
        Identifier for this dataset type
    PROMOTERS : list[str]
        List of available promoter elements in the dataset
    PROMOTERS_CELL_TYPES : dict
        Mapping of promoters to their corresponding cell types
    ENHANCERS : list[str]
        List of available enhancer elements in the dataset  
    ENHANCERS_CELL_TYPES : dict
        Mapping of enhancers to their corresponding cell types
    ALL_ELEMENTS : list[str]
        Combined list of all promoters and enhancers
    """
    
    FLAG = "Kircher"
    # List of promoter elements available in the dataset
    PROMOTERS = [
        "F9", "FOXE1", "GP1BA", "HBB", "HBG1", "HNF4A", "LDLR", "LDLR.2", 
        "MSMB", "PKLR-24h", "PKLR-48h", "TERT-GAa", "TERT-GBM", "TERT-GSc", "TERT-HEK"
    ]
    
    # Mapping of promoters to their corresponding cell types
    PROMOTERS_CELL_TYPES = {
        "F9": "HepG2", "FOXE1": "HeLa", "GP1BA": "HEL92.1.7", 
        "HBB": "HEL92.1.7", "HBG1": "HEL92.1.7", "HNF4A": "HEK293T", 
        "LDLR": "HepG2", "LDLR.2": "HepG2", "MSMB": "HEK293T", 
        "PKLR-24h": "K562", "PKLR-48h": "K562", 
        "TERT-GAa": ["HEK293T", "SF7996"], "TERT-GBM": ["HEK293T", "SF7996"], 
        "TERT-GSc": ["HEK293T", "SF7996"], "TERT-HEK": ["HEK293T", "SF7996"]
    }
    
    # List of enhancer elements available in the dataset
    ENHANCERS = [
        "BCL11A", "IRF4", "IRF6", "MYCrs6983267", "MYCrs11986220", "RET", 
        "SORT1", "SORT1-flip", "SORT1.2", "TCF7L2", "UC88", "ZFAND3", 
        "ZRSh-13", "ZRSh-13h2"
    ]
    
    # Mapping of enhancers to their corresponding cell types
    ENHANCERS_CELL_TYPES = {
        "BCL11A": "HEL92.1.7", "IRF4": "SK-MEL-28", "IRF6": "HaCaT", 
        "MYCrs6983267": "HEK293T", "MYCrs11986220": "LNCaP", 
        "RET": "Neuro-2a", "SORT1": "HepG2", "SORT1-flip": "HepG2", 
        "SORT1.2": "HepG2", "TCF7L2": "MIN6", "UC88": "Neuro-2a", 
        "ZFAND3": "MIN6", "ZRSh-13": "NIH/3T3", "ZRSh-13h2": "NIH/3T3"
    }

    def __init__(
        self,
        split: str = "test",
        length: int = 230,  # length of cutted sequence
        promoter_enhancer: list[str] | str = None,
        cell_types: list[str] | str = None,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize the Kircher MPRA dataset.
        
        Attributes
        ----------
        split : str, optional
            Specifies how to split the data. Currently only "test" is supported.
            Default is "test".
        length : int, optional  
            Length of the sequence for the differential expression experiment. 
            Must be positive integer. Default is 230.
        promoter_enhancer : Union[list[str], str], optional
            List of promoter-enhancer elements to include. If None, includes all elements.
            Can be a single string or list of strings.
        cell_types : Union[list[str], str], optional
            List of cell types to filter by. If None, includes all elements.
            Can be a single string or list of strings.
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool
            If True, exclude the specified regions instead of including them.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data (expression values).
        root : str, optional
            Root directory where data is stored. If None, uses default data directory.
        """
        # Initialize parent class
        super().__init__(split, root)

        self.transform = transform
        self.target_transform = target_transform
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions
        self.prefix = self.FLAG + "_"  # Prefix for file names

        # Combine all available elements for validation
        self.ALL_ELEMENTS = self.PROMOTERS + self.ENHANCERS
        
        # Validate promoter-enhancer input
        if (isinstance(promoter_enhancer, str) and promoter_enhancer not in self.ALL_ELEMENTS) or (
            isinstance(promoter_enhancer, list)
            and not all(p in self.ALL_ELEMENTS for p in promoter_enhancer)
        ):
            raise ValueError("Invalid promoter-enhancer list")

        # Validate sequence length parameter
        if not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"Parameter 'length' must be natural integer, not {length}."
            )
        self.length = length

        try:
            # Load the data file
            file_name = self.prefix + "GRCh38_ALL" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process data - ensure proper chromosome formatting
        df.Chromosome = df.Chromosome.astype(str)
        df.Position = df.Position.astype(int)
        self.ds = df.dropna().reset_index(drop=True)
        target_column = "Value"  # Column containing expression values

        if self.genomic_regions is None:
            # Filter by cell type or promoter-enhancer
            # Preprocess cell type column to handle multiple cell types in one cell
            # Convert to list of strings for easier filtering
            self.ds['Cell_type_processed'] = self.ds['Cell_Type'].apply(
                lambda x: [ct.strip() for ct in str(x).split(',')] if pd.notna(x) else []
            )
    
            # Filter by cell types if specified
            if cell_types is not None:
                # Convert single cell type to list for consistency
                if isinstance(cell_types, str):
                    cell_types = [cell_types]
                
                # Filter rows where any of the processed cell types matches the requested cell types
                self.ds = self.ds[
                    self.ds['Cell_type_processed'].apply(
                        lambda cell_list: any(cell in cell_types for cell in cell_list)
                    )
                ]
    
            # Filter by promoters and enhancers if specified
            if promoter_enhancer is not None:
                # Convert single element to list for consistency
                if isinstance(promoter_enhancer, str):
                    promoter_enhancer = [promoter_enhancer]
                
                self.ds = self.ds[self.ds.Element.isin(promoter_enhancer)]
            else:
                # Include all promoters and enhancers if none specified
                self.ds = self.ds[self.ds.Element.isin(self.ALL_ELEMENTS)]
        else:
            
            # If self.genomic_regions is not None filter by genomic regions 
            self.ds = self.filter_by_genomic_regions(self.ds)
            
        # Clean up and reset index after filtering
        self.ds = self.ds.dropna().reset_index(drop=True)
        
        # Set up FASTA reference file for sequence extraction
        fasta_file = self._setup_fasta_file()

        # Load FASTA reference genome
        try:
            ref = pyfaidx.Fasta(fasta_file)
        except Exception as e:
            raise IOError(f"Error loading FASTA file {fasta_file}: {str(e)}") from e
        
        # Extract alternative sequences (with SNP/varaint)
        self.ds["seq_alt"] = self.ds.apply(
            lambda row: self.get_sequence(
                file_pyfaidx=ref,
                chromosome=row.Chromosome,
                length=self.length,
                pos=row.Position,
                ref=row.Ref,
                alt=row.Alt,
            ),
            axis=1,
        )

        # Extract reference sequences (without variant)
        self.ds["seq_ref"] = self.ds.apply(
            lambda row: self.get_sequence(
                file_pyfaidx=ref,
                chromosome=row.Chromosome,
                length=self.length,
                pos=row.Position,
                ref=row.Ref,
                alt=row.Ref,  # Use reference allele instead of alternative
            ),
            axis=1,
        )

        # Prepare final dataset structure
        targets = self.ds[target_column].to_numpy()
        seq_alt = self.ds.seq_alt.to_numpy()
        seq_ref = self.ds.seq_ref.to_numpy()
        self.ds = {"targets": targets, "seq": seq_ref, "seq_alt": seq_alt}

        # Identifier for split information
        self.name_for_split_info = self.prefix

    def filter_by_genomic_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe based on genomic regions using bioframe.
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
        # Create start and end positions based on the mutation position and desired length
        data_df = df.copy()
        half_length = self.length // 2
        
        # Calculate start and end positions for each sequence
        data_df["start"] = data_df["Position"] - half_length
        data_df["end"] = data_df["Position"] + half_length
        data_df["chrom"] = data_df["Chromosome"]
        
        # Convert to integer if possible
        for col in ["start", "end"]:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce").astype("Int64")

        # Find intersections
        intersections = bf.overlap(data_df, regions_df, how="inner", return_index=True)
        
        if self.exclude_regions:
            # Exclude sequences that overlap with specified regions
            filtered_df = df[~df.index.isin(intersections["index"])]
        else:
            # Include only sequences that overlap with specified regions
            filtered_df = df[df.index.isin(intersections["index"])]

        return filtered_df
        
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
                raise IOError(
                    f"Failed to download/decompress FASTA file: {str(e)}"
                ) from e
        else:
            # print("FASTA file already exists. Skipping download.")
            pass

        return fasta_file

    def get_sequence(
        self, file_pyfaidx, chromosome: str, length: int, pos: int, ref: str, alt: str
    ) -> str:
        """
        Extract sequence from a FASTA file with padding to a fixed length.
        """
        chromosome = "chr" + chromosome
        # Input validation
        if not isinstance(ref, str) or len(ref) != 1:
            raise ValueError(
                f"Reference nucleotide should be single character, got {ref}"
            )

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
                modified_seq = seq[:ref_pos_in_seq] + seq[ref_pos_in_seq + 1 :]
            else:
                # Handle substitution or insertion
                seq = str(file_pyfaidx.get_seq(chromosome, start, end))
                modified_seq = seq[:ref_pos_in_seq] + alt + seq[ref_pos_in_seq + 1 :]

            return modified_seq
        except Exception as e:
            raise ValueError(
                f"Error processing {chromosome}:{pos}-{ref}>{alt}: {str(e)}"
            ) from e
