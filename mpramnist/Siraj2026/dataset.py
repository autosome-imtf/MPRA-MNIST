import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Union, Dict, Optional
import pyfaidx
import subprocess
import bioframe as bf

from mpramnist.mpradataset import MpraDataset


class SirajMPRADataset(MpraDataset):
    """
    Dataset class for Siraj MPRA (Massively Parallel Reporter Assay) data.

    This class handles loading, filtering, and processing of genomic sequence data
    from the Siraj et al. study, which focuses on functional dissection of complex
    trait variants at single-nucleotide resolution across relevant cell types.

    The dataset uses human genome assembly hg19 with 0-based coordinate indexing.
    All genomic positions (start, end) follow 0-based indexing convention.

    Attributes
    ----------
    FLAG : str
        Identifier flag for Siraj datasets ("Siraj")
    CELL_TYPE : list
        Available cell types ['K562', 'HEPG2', 'A549', 'SKNSH', 'HCT116']
    PLASMID : str
        Constant plasmid backbone sequence used in the MPRA constructs
    LEFT_FLANK : str
        Left flanking sequence used for sequence extraction and alignment
    RIGHT_FLANK : str
        Right flanking sequence used for sequence extraction and alignment

    Examples
    --------
    >>> # Load data for specific cell type
    >>> dataset = SirajMPRADataset(cell_type='K562')
    >>>
    >>> # Load data with custom sequence length
    >>> dataset = SirajMPRADataset(cell_type='HEPG2', length=300)
    >>>
    >>> # Load data filtered by genomic regions
    >>> dataset = SirajMPRADataset(
    ...     cell_type='A549',
    ...     genomic_regions='path/to/regions.bed'
    ... )

    """

    FLAG = "Siraj"
    CELL_TYPE: list[str] = ["K562", "HEPG2", "A549", "SKNSH", "HCT116"]

    PLASMID = "TCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATACGGTTATCCACAGAATCAGGGGATAACGCAGGAAAGAACATGTGAGCAAAAGGCCAGCAAAAGGCCAGGAACCGTAAAAAGGCCGCGTTGCTGGCGTTTTTCCATAGGCTCCGCCCCCCTGACGAGCATCACAAAAATCGACGCTCAAGTCAGAGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCCTGGAAGCTCCCTCGTGCGCTCTCCTGTTCCGACCCTGCCGCTTACCGGATACCTGTCCGCCTTTCTCCCTTCGGGAAGCGTGGCGCTTTCTCATAGCTCACGCTGTAGGTATCTCAGTTCGGTGTAGGTCGTTCGCTCCAAGCTGGGCTGTGTGCACGAACCCCCCGTTCAGCCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGTAAGACACGACTTATCGCCACTGGCAGCAGCCACTGGTAACAGGATTAGCAGAGCGAGGTATGTAGGCGGTGCTACAGAGTTCTTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGAACAGTATTTGGTATCTGCGCTCTGCTGAAGCCAGTTACCTTCGGAAAAAGAGTTGGTAGCTCTTGATCCGGCAAACAAACCACCGCTGGTAGCGGTGGTTTTTTTGTTTGCAAGCAGCAGATTACGCGCAGAAAAAAAGGATCTCAAGAAGATCCTTTGATCTTTTCTACGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGCGGCCGCAAATGCTAAACCACTGCAGTGGTTACCAGTGCTTGATCAGTGAGGCACCGATCTCAGCGATCTGCCTATTTCGTTCGTCCATAGTGGCCTGACTCCCCGTCGTGTAGATCACTACGATTCGTGAGGGCTTACCATCAGGCCCCAGCGCAGCAATGATGCCGCGAGAGCCGCGTTCACCGGCCCCCGATTTGTCAGCAATGAACCAGCCAGCAGGGAGGGCCGAGCGAAGAAGTGGTCCTGCTACTTTGTCCGCCTCCATCCAGTCTATGAGCTGCTGTCGTGATGCTAGAGTAAGAAGTTCGCCAGTGAGTAGTTTCCGAAGAGTTGTGGCCATTGCTACTGGCATCGTGGTATCACGCTCGTCGTTCGGTATGGCTTCGTTCAACTCTGGTTCCCAGCGGTCAAGCCGGGTCACATGATCACCCATATTATGAAGAAATGCAGTCAGCTCCTTAGGGCCTCCGATCGTTGTCAGAAGTAAGTTGGCCGCGGTGTTGTCGCTCATGGTAATGGCAGCACTACACAATTCTCTTACCGTCATGCCATCCGTAAGATGCTTTTCCGTGACCGGCGAGTACTCAACCAAGTCGTTTTGTGAGTAGTGTATACGGCGACCAAGCTGCTCTTGCCCGGCGTCTATACGGGACAACACCGCGCCACATAGCAGTACTTTGAAAGTGCTCATCATCGGGAATCGTTCTTCGGGGCGGAAAGACTCAAGGATCTTGCCGCTATTGAGATCCAGTTCGATATAGCCCACTCTTGCACCCAGTTGATCTTCAGCATCTTTTACTTTCACCAGCGTTTCGGGGTGTGCAAAAACAGGCAAGCAAAATGCCGCAAAGAAGGGAATGAGTGCGACACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG<__ENHANCER_SEQ__>CACTGCGGCTCCTGCGATCGCGTCGACGAACCTCTAGANNNNNNNNNNNNNNNNNNNNAGATCGGAAGAGCGTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTGACCTACGGCGTGCAGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAAGGTCTATATCACCGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGACCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAATGATAATAATTCTAGAGTCGGGGCGGCCGGCCGCTTCGAGCAGACATGATAAGATACATTGATGAGTTTGGACAAACCACAACTAGAATGCAGTGAAAAAAATGCTTTATTTGTGAAATTTGTGATGCTATTGCTTTATTTGTAACCATTATAAGCTGCAATAAACAAGTTAACAACAACAATTGCATTCATTTTATGTTTCAGGTTCAGGGGGAGGTGTGGGAGGTTTTTTAAAGCAAGTAAAACCTCTACAAATGTGGTAAAATCGATAAGGATCCGTCGACCGATGCCCTTGAGAGCCTTCAACCCAGTCAGCTCCTTCCGGTGGGCGCGGGGCATGACTATCGTCGCCGCACTTATGACTGTCTTCTTTATCATGCAACTCGTAGGACAGGTGCCGGCAGCGCTCTTCCGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGT"

    LEFT_FLANK = "ACTGGCCGCTTGACG"
    RIGHT_FLANK = "CACTGCGGCTCCTGC"

    def __init__(
        self,
        cell_type: str,
        split: str = "test",
        length: int = 200,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        filter_not_active: bool = False,
        filter_not_emVar: bool = False,
        filter_not_active_in_cellline: bool = True,
        filter_not_emVar_in_cellline: bool = True,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize the Siraj MPRA dataset.

        Attributes
        ----------
        cell_type : str
            Cell type to filter by.
            Must be one of: "K562", "HEPG2", "A549", "SKNSH", "HCT116".
        split : str, optional
            Specifies how to split the data. Currently only "test" is supported.
            Default is "test".
        length : int, optional
            Length of the sequence for the differential expression experiment.
            Must be positive integer. Default is 200.
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool, optional
            If True, exclude the specified regions instead of including them.
        filter_not_active : bool, optional
            If True, exclude sequences not active in all cell lines instead of including them.
        filter_not_emVar : bool, optional
            If True, exclude variants not expression-modulating in all cell lines instead of including them.
        filter_not_active_in_cellline : bool, optional
            If True, exclude sequences not active in target cell line instead of including them.
        filter_not_emVar_in_cellline : bool, optional
            If True, exclude variants not expression-modulating in target cell line instead of including them.
        transform : callable, optional, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional, optional
            Transformation applied to the target data (expression values).
        root : str, optional, optional
            Root directory where data is stored. If None, uses default data directory.
        """
        # Initialize parent class
        super().__init__(split, root)

        self.cell_type = cell_type.upper()
        if self.cell_type not in self.CELL_TYPE:
            raise ValueError(f"Wrong cell line provided: {self.cell_type}")

        self.length = int(length)
        self.transform = transform
        self.target_transform = target_transform
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions
        self.filter_not_active = filter_not_active
        self.filter_not_emVar = filter_not_emVar
        self.filter_not_active_in_cellline = filter_not_active_in_cellline
        self.filter_not_emVar_in_cellline = filter_not_emVar_in_cellline
        self.prefix = self.FLAG  # Prefix for file names

        try:
            # Load the data file
            file_name = self.prefix + "MPRA" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process data - ensure proper chromosome formatting
        if self.filter_not_active:
            df = df.loc[df["Active (any cell-type)"] == True]
        if self.filter_not_emVar:
            df = df.loc[df["emVar (any cell-type)"] == True]

        df = df[
            ["Variant"]
            + df.columns[df.columns.str.endswith(f"{self.cell_type}")].tolist()
        ]
        df = df.join(
            df["Variant"]
            .str.split(":", expand=True)
            .set_axis(["Chrom", "Pos", "Ref", "Alt"], axis=1)
            .astype({"Chrom": str, "Pos": int, "Ref": str, "Alt": str})
        )
        df["Pos"] -= 1
        half_len = self.length // 2
        df["Start"] = df.Pos - half_len
        df["End"] = df.Pos + self.length - half_len

        if self.filter_not_active_in_cellline:
            df = df.loc[df[f"active in {self.cell_type}"] == True]
        if self.filter_not_emVar_in_cellline:
            df = df.loc[df[f"emVar in {self.cell_type}"] == True]

        df = df.loc[
            (df.apply(lambda row: len(row["Ref"]), axis=1) == 1)
            & (df.apply(lambda row: len(row["Alt"]), axis=1) == 1)
        ]

        target_column = (
            f"log2Skew in {self.cell_type}"  # Column containing expression values
        )
        self.ds = df.loc[df[target_column].replace([np.inf, -np.inf], np.nan).notna()]

        self.ds["Chrom"] = self.ds["Chrom"].str.removeprefix("chr")
        self.ds[target_column] = self.ds[target_column].astype(float)

        if self.genomic_regions is not None:
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
                chromosome=row.Chrom,
                start=row.Start,
                end=row.End,
                pos=row.Pos,
                ref=row.Ref,
                alt=row.Alt,
            ),
            axis=1,
        )

        # Extract reference sequences (without variant)
        self.ds["seq_ref"] = self.ds.apply(
            lambda row: self.get_sequence(
                ref_genome=ref,
                chromosome=row.Chrom,
                start=row.Start,
                end=row.End,
                pos=row.Pos,
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

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'Chromosome': chromosome name (hg19)
            - 'Position': variant position (0-based, hg19)

        Returns
        -------
        pd.DataFrame
            Filtered dataframe containing only sequences that overlap (or don't overlap)
            with the specified genomic regions

        Notes
        -----
        - Uses bioframe library for genomic interval operations
        - All genomic coordinates use hg19 assembly with 0-based indexing
        - Sequences are defined as regions centered on variant positions
        - Input regions should be provided in hg19 coordinates with 0-based indexing
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

        # Convert to integer if possible
        for col in ["Start", "End"]:
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
        - Downloads hg19 reference genome from UCSC if not present
        - Uses 0-based coordinate system for sequence extraction
        """

        fasta_file = os.path.join(self._data_path, "hg19.fa")

        if not os.path.exists(fasta_file):
            url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
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
        self,
        ref_genome,
        chromosome: str,
        start: int,
        end: int,
        pos: int,
        ref: str,
        alt: str,
    ) -> str:
        """
        Extract sequence from a FASTA file with padding to a fixed length.

        Parameters
        ----------
        ref_genome : pyfaidx.Fasta
            FASTA file object for sequence extraction
        chromosome : str
            Chromosome name (without 'chr' prefix)
        start : int
            Start position (0-based, hg19)
        end : int
            End position (0-based, hg19)
        pos : int
            Variant position (0-based, hg19)
        ref : str
            Reference allele
        alt : str
            Alternative allele

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
        - Uses hg19 reference genome with 0-based coordinates
        - Sequences are centered on the variant position
        - Handles both substitutions and deletions
        - For deletions, the sequence length is maintained by removing the deleted base
        """
        # Verify reference nucleotide matches expected
        chromosome = "chr" + str(chromosome)

        observed_ref = str(ref_genome[chromosome][pos : pos + 1]).upper()
        if observed_ref != ref.upper():
            raise ValueError(
                f"Error processing {chromosome}:{pos}-{ref}>{alt}\nCalled ref ({ref.upper()}) differs from genomic ({observed_ref})"
            )

        half_len = (end - start) // 2

        try:
            # Handle substitution or insertion
            seq = str(ref_genome[chromosome][start : end + 1])
            modified_seq = seq[:half_len] + alt + seq[half_len + 1 :]

            return modified_seq
        except Exception as e:
            raise ValueError(
                f"Error processing {chromosome}:{pos}-{ref}>{alt}: {str(e)}"
            ) from e


class SirajSatMutDataset(MpraDataset):
    """
    Dataset class for Siraj SatMut (Saturation Mutagenesis MPRA) data.

    This class handles loading, filtering, and processing of genomic sequence data
    from the Siraj et al. study, which focuses on functional dissection of complex
    trait variants at single-nucleotide resolution across relevant cell types.

    The dataset uses human genome assembly hg19 with 0-based coordinate indexing.
    All genomic positions (start, end) follow 0-based indexing convention.

    Attributes
    ----------
    FLAG : str
        Identifier flag for Siraj datasets ("Siraj")
    CELL_TYPE : list
        Available cell types ['K562', 'HEPG2']
    MUT_NUM : list
        Available baseline mutation number [1, 2]
    PLASMID : str
        Constant plasmid backbone sequence used in the MPRA constructs
    LEFT_FLANK : str
        Left flanking sequence used for sequence extraction and alignment
    RIGHT_FLANK : str
        Right flanking sequence used for sequence extraction and alignment

    Examples
    --------
    >>> # Load data for specific cell type
    >>> dataset = SirajSatMutDataset(cell_type='K562', mut_num=1)
    >>>
    >>> # Load data with custom sequence length
    >>> dataset = SirajSatMutDataset(cell_type='HEPG2', mut_num=1, length=300)
    >>>
    >>> # Load data filtered by genomic regions
    >>> dataset = SirajSatMutDataset(
    ...     cell_type='K562',
    ...     mut_num=1,
    ...     genomic_regions='path/to/regions.bed'
    ... )

    """

    FLAG = "Siraj"
    CELL_TYPE: list[str] = ["K562", "HEPG2"]
    BASELINE_ALLELE: list[str] = ["A", "R", "AA", "AR", "RA", "RR"]

    PLASMID = "TCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATACGGTTATCCACAGAATCAGGGGATAACGCAGGAAAGAACATGTGAGCAAAAGGCCAGCAAAAGGCCAGGAACCGTAAAAAGGCCGCGTTGCTGGCGTTTTTCCATAGGCTCCGCCCCCCTGACGAGCATCACAAAAATCGACGCTCAAGTCAGAGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCCTGGAAGCTCCCTCGTGCGCTCTCCTGTTCCGACCCTGCCGCTTACCGGATACCTGTCCGCCTTTCTCCCTTCGGGAAGCGTGGCGCTTTCTCATAGCTCACGCTGTAGGTATCTCAGTTCGGTGTAGGTCGTTCGCTCCAAGCTGGGCTGTGTGCACGAACCCCCCGTTCAGCCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGTAAGACACGACTTATCGCCACTGGCAGCAGCCACTGGTAACAGGATTAGCAGAGCGAGGTATGTAGGCGGTGCTACAGAGTTCTTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGAACAGTATTTGGTATCTGCGCTCTGCTGAAGCCAGTTACCTTCGGAAAAAGAGTTGGTAGCTCTTGATCCGGCAAACAAACCACCGCTGGTAGCGGTGGTTTTTTTGTTTGCAAGCAGCAGATTACGCGCAGAAAAAAAGGATCTCAAGAAGATCCTTTGATCTTTTCTACGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGCGGCCGCAAATGCTAAACCACTGCAGTGGTTACCAGTGCTTGATCAGTGAGGCACCGATCTCAGCGATCTGCCTATTTCGTTCGTCCATAGTGGCCTGACTCCCCGTCGTGTAGATCACTACGATTCGTGAGGGCTTACCATCAGGCCCCAGCGCAGCAATGATGCCGCGAGAGCCGCGTTCACCGGCCCCCGATTTGTCAGCAATGAACCAGCCAGCAGGGAGGGCCGAGCGAAGAAGTGGTCCTGCTACTTTGTCCGCCTCCATCCAGTCTATGAGCTGCTGTCGTGATGCTAGAGTAAGAAGTTCGCCAGTGAGTAGTTTCCGAAGAGTTGTGGCCATTGCTACTGGCATCGTGGTATCACGCTCGTCGTTCGGTATGGCTTCGTTCAACTCTGGTTCCCAGCGGTCAAGCCGGGTCACATGATCACCCATATTATGAAGAAATGCAGTCAGCTCCTTAGGGCCTCCGATCGTTGTCAGAAGTAAGTTGGCCGCGGTGTTGTCGCTCATGGTAATGGCAGCACTACACAATTCTCTTACCGTCATGCCATCCGTAAGATGCTTTTCCGTGACCGGCGAGTACTCAACCAAGTCGTTTTGTGAGTAGTGTATACGGCGACCAAGCTGCTCTTGCCCGGCGTCTATACGGGACAACACCGCGCCACATAGCAGTACTTTGAAAGTGCTCATCATCGGGAATCGTTCTTCGGGGCGGAAAGACTCAAGGATCTTGCCGCTATTGAGATCCAGTTCGATATAGCCCACTCTTGCACCCAGTTGATCTTCAGCATCTTTTACTTTCACCAGCGTTTCGGGGTGTGCAAAAACAGGCAAGCAAAATGCCGCAAAGAAGGGAATGAGTGCGACACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG<__ENHANCER_SEQ__>CACTGCGGCTCCTGCGATCGCGTCGACGAACCTCTAGANNNNNNNNNNNNNNNNNNNNAGATCGGAAGAGCGTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCTTGACCTACGGCGTGCAGTGCTTCGCCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAAGGTCTATATCACCGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGACCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAATGATAATAATTCTAGAGTCGGGGCGGCCGGCCGCTTCGAGCAGACATGATAAGATACATTGATGAGTTTGGACAAACCACAACTAGAATGCAGTGAAAAAAATGCTTTATTTGTGAAATTTGTGATGCTATTGCTTTATTTGTAACCATTATAAGCTGCAATAAACAAGTTAACAACAACAATTGCATTCATTTTATGTTTCAGGTTCAGGGGGAGGTGTGGGAGGTTTTTTAAAGCAAGTAAAACCTCTACAAATGTGGTAAAATCGATAAGGATCCGTCGACCGATGCCCTTGAGAGCCTTCAACCCAGTCAGCTCCTTCCGGTGGGCGCGGGGCATGACTATCGTCGCCGCACTTATGACTGTCTTCTTTATCATGCAACTCGTAGGACAGGTGCCGGCAGCGCTCTTCCGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGT"

    LEFT_FLANK = "ACTGGCCGCTTGACG"
    RIGHT_FLANK = "CACTGCGGCTCCTGC"

    def __init__(
        self,
        cell_type: str,
        baseline_allele: str,
        split: str = "test",
        length: int = 200,
        log2Skew_pval: float | None = None,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize the Siraj SatMut dataset.

        Attributes
        ----------
        cell_type : str
            Cell type to filter by. Must be one of: "K562", "HEPG2".
        baseline_allele : str
            Number of baseline mutations. Must be 1 or 2.
        split : str, optional
            Specifies how to split the data. Currently only "test" is supported.
            Default is "test".
        length : int, optional
            Length of the sequence for the differential expression experiment.
            Must be positive integer. Default is 200.
        log2Skew_pval: float, optional
            Threshold p-value for log2Skew.
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool, optional
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

        self.length = int(length)

        self.cell_type = cell_type.upper()
        if self.cell_type not in self.CELL_TYPE:
            raise ValueError(f"Wrong cell line provided: {self.cell_type}")

        self.baseline_allele = baseline_allele
        if self.baseline_allele not in self.BASELINE_ALLELE:
            raise ValueError(f"Wrong mutation number provided: {self.baseline_allele}")

        self.log2Skew_pval = log2Skew_pval
        self.transform = transform
        self.target_transform = target_transform
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions
        self.prefix = self.FLAG  # Prefix for file names

        try:
            # Load the data file
            file_name = self.prefix + "SatMut" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process data - ensure proper chromosome formatting
        df = df.loc[df["Cell Type"] == self.cell_type]
        df = df.loc[df["allele"] == self.baseline_allele]
        df = df.loc[df["indel"].isna()]
        df = df.loc[df["oligomut"] != "m0"]
        df = df.loc[df["window"].isin(["wC", "wL", "wR"])]
        df = df.join(
            df["oligomut"]
            .str.extract(r"m([ACGT])(?:\d+)([ACGT])")
            .set_axis(["mut_ref", "mut_alt"], axis=1)
            .astype({"mut_ref": str, "mut_alt": str})
        )

        df["pos1"] = df["pos1"].astype(int) - 1
        if len(baseline_allele) == 1:
            pos = df.pos1
        else:
            df["pos2"] = df["pos2"].astype(int) - 1
            pos = np.where(df["centervar"] == "var1", df.pos1, df.pos2)
        cent_mv = df["window"].map({"wC": 0, "wL": 50, "wR": -50})

        half_len = self.length // 2
        df["start"] = pos + cent_mv - half_len + 1
        df["end"] = pos + cent_mv + self.length - half_len + 1
        df["mut_pos"] -= 1

        df = df.loc[
            (df.apply(lambda row: len(row["ref"]), axis=1) == 1)
            & (df.apply(lambda row: len(row["alt"]), axis=1) == 1)
            & (df.apply(lambda row: len(row["mut_base"]), axis=1) == 1)
        ]

        target_column = "log2Skew"  # Column containing expression values
        self.ds = df.loc[df[target_column].replace([np.inf, -np.inf], np.nan).notna()]

        if self.log2Skew_pval is not None:
            self.ds["log2Skew_pval"] = self.ds["log2Skew_pval"].astype(float)
            self.ds = self.ds.loc[self.ds["log2Skew_pval"] < self.log2Skew_pval]

        self.ds["mut_pos"] = self.ds["mut_pos"].astype(int)
        self.ds[target_column] = self.ds[target_column].astype(float)

        if self.genomic_regions is not None:
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
                chromosome=row.chr,
                start=row.start,
                end=row.end,
                baseline_allele=self.baseline_allele,
                pos=(row.pos1, row.pos2),
                ref=(row.ref, row.ref2),
                alt=(row.alt, row.alt2),
                mut_pos=row.mut_pos + half_len - 100,
                mut_ref=row.mut_ref,
                mut_alt=row.mut_alt,
            ),
            axis=1,
        )

        # Extract reference sequences (without variant)
        self.ds["seq_ref"] = self.ds.apply(
            lambda row: self.get_sequence(
                ref_genome=ref,
                chromosome=row.chr,
                start=row.start,
                end=row.end,
                baseline_allele=self.baseline_allele,
                pos=(row.pos1, row.pos2),
                ref=(row.ref, row.ref2),
                alt=(row.alt, row.alt2),
                mut_pos=row.mut_pos + half_len - 100,
                mut_ref=row.mut_ref,
                mut_alt=row.mut_ref,  # Use reference allele instead of alternative
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

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing genomic data with columns:
            - 'Chromosome': chromosome name (hg19)
            - 'Position': variant position (0-based, hg19)

        Returns
        -------
        pd.DataFrame
            Filtered dataframe containing only sequences that overlap (or don't overlap)
            with the specified genomic regions

        Notes
        -----
        - Uses bioframe library for genomic interval operations
        - All genomic coordinates use hg19 assembly with 0-based indexing
        - Sequences are defined as regions centered on variant positions
        - Input regions should be provided in hg19 coordinates with 0-based indexing
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
        - Downloads hg19 reference genome from UCSC if not present
        - Uses 0-based coordinate system for sequence extraction
        """

        fasta_file = os.path.join(self._data_path, "hg19.fa")

        if not os.path.exists(fasta_file):
            url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
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
        self,
        ref_genome,
        chromosome: str,
        start: int,
        end: int,
        baseline_allele: str,
        pos: Tuple[int, int],
        ref: Tuple[str, str],
        alt: Tuple[str, str],
        mut_pos: int,
        mut_ref: str,
        mut_alt: str,
    ) -> str:
        """
        Extract sequence from a FASTA file with padding to a fixed length.

        Parameters
        ----------
        ref_genome : pyfaidx.Fasta
            FASTA file object for sequence extraction
        chromosome : str
            Chromosome name (without 'chr' prefix)
        start : int
            Start position (0-based, hg19)
        end : int
            End position (0-based, hg19)
        baseline_allele : str
            Number of baseline mutations (1 or 2)
        pos : Tuple[int, int]
            Pair of variant positions (0-based, hg19)
        ref : Tuple[str, str]
            Pair of reference alleles (single character each)
        alt : Tuple[str, str]
            Pair of alternative alleles (single character each)
        mut_pos : int
            Target variant position (0-based, hg19)
        mut_ref : str
            Target reference allele (single character each)
        mut_alt : str
            Target alternative allele (single character each)

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
        - Uses hg19 reference genome with 0-based coordinates
        - Sequences are centered on the variant position
        - Handles both substitutions and deletions
        - For deletions, the sequence length is maintained by removing the deleted base
        """
        # Verify reference nucleotide matches expected
        chromosome = "chr" + str(chromosome)

        observed_ref1 = str(ref_genome[chromosome][pos[0] : pos[0] + 1]).upper()
        if observed_ref1 != ref[0].upper():
            raise ValueError(
                f"Error processing {chromosome}:\n{pos}-{ref}>{alt}\n{mut_pos}-{mut_ref}>{mut_alt}\nCalled ref1 ({ref[0].upper()}) differs from genomic ({observed_ref1})"
            )
        if len(baseline_allele) == 2:
            observed_ref2 = str(ref_genome[chromosome][pos[1] : pos[1] + 1]).upper()
            if observed_ref2 != ref[1].upper():
                raise ValueError(
                    f"Error processing {chromosome}:\n{pos}-{ref}>{alt}\n{mut_pos}-{mut_ref}>{mut_alt}\nCalled ref2 ({ref[1].upper()}) differs genomic ({observed_ref2})"
                )

        try:
            # Handle substitution or insertion
            seq = str(ref_genome[chromosome][start : end + 1]).upper()
            if baseline_allele[0] == "A":
                seq = seq[: pos[0] - start] + alt[0] + seq[pos[0] - start + 1 :]
            if baseline_allele[1:] == "A":
                seq = seq[: pos[1] - start] + alt[1] + seq[pos[1] - start + 1 :]

            observed_ref = seq[mut_pos : mut_pos + 1]
            if observed_ref != mut_ref.upper():
                raise ValueError(
                    f"Error processing {chromosome}:\n{pos}-{ref}>{alt}\n{mut_pos}-{mut_ref}>{mut_alt}\nCalled target ref ({mut_ref.upper()}) differs from genomic ({observed_ref})"
                )

            mut_seq = seq[:mut_pos] + mut_alt + seq[mut_pos + 1 :]

            return mut_seq
        except Exception as e:
            raise ValueError(
                f"Error processing {chromosome}:\n{pos}-{ref}>{alt}\n{mut_pos}-{mut_ref}>{mut_alt}: {str(e)}"
            ) from e
