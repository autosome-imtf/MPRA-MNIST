import pandas as pd
import os
from typing import List, Union, Dict, Optional
import pyfaidx
import subprocess
import bioframe as bf

from mpramnist.mpradataset import MpraDataset


class KircherDataset(MpraDataset):
    """
    Dataset class for Kircher MPRA (Massively Parallel Reporter Assay) data.
    
    This class handles loading, filtering, and processing of genomic sequence data
    from the Kircher et al. study, which contains both promoter and enhancer elements
    across multiple cell types with SNP/variant information.

    The dataset uses human genome assembly hg38 with 0-based coordinate indexing.
    All genomic positions (start, end) follow 0-based indexing convention.

    Inherits from:
        MpraDataset: Base class for MPRA datasets

    Constants:
        FLAG (str): Dataset identifier flag: 'Kircher'
        CELL_TYPE (dict): Mapping of elements to their corresponding cell types

    Examples:
        >>> # Load all promoter elements
        >>> dataset = KircherDataset(elements=['F9', 'HBB', 'LDLR'])
        >>> 
        >>> # Load data for specific cell types
        >>> dataset = KircherDataset(cell_type=['HepG2', 'K562'])
        >>> 
        >>> # Load data with custom sequence length
        >>> dataset = KircherDataset(length=300, elements='HBB')
        >>> 
        >>> # Load data filtered by genomic regions
        >>> dataset = KircherDataset(
        ...     genomic_regions='path/to/regions.bed',
        ...     elements=['BCL11A', 'IRF4']
        ... )
    """
    
    FLAG = "Kircher"
    
    # Mapping of elements to their corresponding cell types
    CELL_TYPE = {
        # promoters
        "F9": "HepG2", "FOXE1": "HeLa", "GP1BA": "HEL92.1.7", 
        "HBB": "HEL92.1.7", "HBG1": "HEL92.1.7", "HNF4A": "HEK293T", 
        "LDLR": "HepG2", "LDLR.2": "HepG2", "MSMB": "HEK293T", 
        "PKLR-24h": "K562", "PKLR-48h": "K562", 
        "TERT-GAa": "SF7996", "TERT-GBM": "SF7996", 
        "TERT-GSc": "SF7996", "TERT-HEK": "HEK293T",

        # enhancers
        "BCL11A": "HEL92.1.7", "IRF4": "SK-MEL-28", "IRF6": "HaCaT", 
        "MYCrs6983267": "HEK293T", "MYCrs11986220": "LNCaP", 
        "RET": "Neuro-2a", "SORT1": "HepG2", "SORT1-flip": "HepG2", 
        "SORT1.2": "HepG2", "TCF7L2": "MIN6", "UC88": "Neuro-2a", 
        "ZFAND3": "MIN6", "ZRSh-13": "NIH-3T3", "ZRSh-13h2": "NIH-3T3"
    }

    def __init__(
        self,
        split: str = "test",
        length: int = 200,  # length of cutted sequence
        elements: list[str] | str = None,
        cell_type: list[str] | str = None,
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
            Must be positive integer. Default is 200.
        elements : Union[list[str], str], optional
            List of promoter-enhancer elements to include. If None, includes all elements.
            Can be a single string or list of strings.
        cell_type : Union[list[str], str], optional
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
        
        # Validate promoter-enhancer input
        if (isinstance(elements, str) and elements not in self.CELL_TYPE) or (
            isinstance(elements, list)
            and not all(p in self.CELL_TYPE for p in elements)
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
    
            # Filter by cell types if specified
            if cell_type is not None:
                # Convert single cell type to list for consistency
                if isinstance(cell_type, str):
                    cell_type = [cell_type]
                
                # Filter rows where any of the processed cell types matches the requested cell types
                self.ds = self.ds[self.ds['Cell_Type'].isin(cell_type)]
    
            # Filter by promoters and enhancers if specified
            if elements is not None:
                # Convert single element to list for consistency
                if isinstance(elements, str):
                    elements = [elements]
                
                self.ds = self.ds[self.ds.Element.isin(elements)]
            else:
                # Include all promoters and enhancers if none specified
                self.ds = self.ds[self.ds.Element.isin(self.CELL_TYPE.keys())]
        else:
            
            # If self.genomic_regions is not None filter by genomic regions 
            self.ds = self.filter_by_genomic_regions(self.ds)
        
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
                ref_genome=ref,
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
                ref_genome=ref,
                chromosome=row.Chromosome,
                length=self.length,
                pos=row.Position,
                ref=row.Ref,
                alt=row.Ref,  # Use reference allele instead of alternative
            ),
            axis=1,
        )

        # Clean up and reset index after filtering
        self.ds = self.ds.dropna().reset_index(drop=True)

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

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'Chromosome': chromosome name (hg38)
            - 'Position': variant position (0-based, hg38)

        Returns
        -------
        pd.DataFrame
            Filtered dataframe containing only sequences that overlap (or don't overlap)
            with the specified genomic regions

        Notes
        -----
        - Uses bioframe library for genomic interval operations
        - All genomic coordinates use hg38 assembly with 0-based indexing
        - Sequences are defined as regions centered on variant positions
        - Input regions should be provided in hg38 coordinates with 0-based indexing
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
        """
        Ensure FASTA file exists and is ready for use.

        Returns
        -------
        str
            Path to the FASTA file

        Raises
        ------
        IOError
            If the FASTA file cannot be downloaded or decompressed

        Notes
        -----
        - Downloads hg38 reference genome from UCSC if not present
        - Uses 0-based coordinate system for sequence extraction
        """

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
        self, ref_genome, chromosome: str, length: int, pos: int, ref: str, alt: str
    ) -> str:
        """
        Extract sequence from a FASTA file with padding to a fixed length.

        Parameters
        ----------
        ref_genome : pyfaidx.Fasta
            FASTA file object for sequence extraction
        chromosome : str
            Chromosome name (without 'chr' prefix, will be added automatically)
        length : int
            Total length of sequence to extract
        pos : int
            Variant position (0-based, hg38)
        ref : str
            Reference allele (single character)
        alt : str
            Alternative allele (single character or '-' for deletion)

        Returns
        -------
        str
            Extracted sequence with variant incorporated

        Raises
        ------
        ValueError
            - If reference nucleotide doesn't match expected
            - If sequence extraction fails

        Notes
        -----
        - Uses hg38 reference genome with 0-based coordinates
        - Sequences are centered on the variant position
        - Handles both substitutions and deletions
        - For deletions, the sequence length is maintained by removing the deleted base
        """
        chromosome = "chr" + chromosome
        # Input validation
        if not isinstance(ref, str) or len(ref) != 1:
            raise ValueError(
                f"Reference nucleotide should be single character, got {ref}"
            )

        # Verify reference nucleotide matches expected
        observed_ref = str(ref_genome[chromosome][pos : pos+1]).upper()
        if observed_ref != ref.upper():
            return None

        half_len = length // 2
        start = pos - half_len - 1
        end = pos + half_len
        
        if length % 2 == 0:
            end -= 1

        try:
            ref_pos_in_seq = half_len

            if alt == "-":
                # Handle deletion
                seq = str(ref_genome[chromosome][start : end + 1])
                modified_seq = seq[:ref_pos_in_seq] + seq[ref_pos_in_seq + 1 :]
            else:
                # Handle substitution or insertion
                seq = str(ref_genome[chromosome][start : end])
                modified_seq = seq[:ref_pos_in_seq] + alt + seq[ref_pos_in_seq + 1:]

            return modified_seq
        except Exception as e:
            raise ValueError(
                f"Error processing {chromosome}:{pos}-{ref}>{alt}: {str(e)}"
            ) from e
