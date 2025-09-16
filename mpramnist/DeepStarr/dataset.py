import pandas as pd
from typing import List, Union, Optional, Dict
import os
import bioframe as bf
import warnings

from mpramnist.mpradataset import MpraDataset


class DeepStarrDataset(MpraDataset):
    FLAG = "DeepStarr"

    CELL_TYPES = ["Developmental", "HouseKeeping"]
    LIST_OF_CHR = [
        "chr2L",
        "chr2LHet",
        "chr2RHet",
        "chr3L",
        "chr3LHet",
        "chr3R",
        "chr3RHet",
        "chr4",
        "chrX",
        "chrXHet",
        "chrYHet",
        "chr2R",
    ]
    ACTIVITY_COLUMNS = ["Dev_log2", "Hk_log2"]

    def __init__(
        self,
        split: str | List[str],
        activity_column: str | List[str] = ["Dev_log2", "Hk_log2"],
        use_original_reverse_complement: bool | None = None,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Attributes
        ----------
        split : str | List[str]
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        activity_column : str | List[str]
            Specifies the cell type for filtering the data.
        use_original_reverse_complement : bool
            Determines whether to generate the reverse complement of sequences using the same approach as the original study
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool
            If True, exclude the specified regions instead of including them
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)

        self.activity_column = activity_column
        if use_original_reverse_complement is None:
            if isinstance(split, list) or split != "train":
                use_original_reverse_complement = False
            else:
                use_original_reverse_complement = True

        self.transform = transform
        self.target_transform = target_transform
        self.split, column = self.split_parse(split)
        self.prefix = self.FLAG + "_"
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions

        try:
            file_name = self.prefix + "all_chr" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Apply genomic region filtering
        df = self.filter_by_genomic_regions(df)

        if self.genomic_regions is None:
            df = df[df[column].isin(self.split)].reset_index(drop=True)
        else:
            self.split = "genomic region"

        if use_original_reverse_complement:
            """
            WARNING: 
            This function uses original paper's parameters so:
            > The training dataset (`split=train`) is pre-augmented with reverse complements
            > - 2× sequences (original + RC)  
            > - Identical labels for RC pairs  
            > Manual reverse complementing will cause data leakage!
            """

            if self.split == ["train"]:
                warnings.warn(
                    "WARNING! "
                    "\nNote: The training set contains reverse-complement augmentation as implemented in the original study.  "
                    "\n• Dataset size: 2N (N original + N reverse-complemented sequences)  "
                    "\n• Label consistency: y_rc ≡ y_original  "
                    "\n• Do not reapply this transformation during preprocessing. ",
                    stacklevel=1,
                )

            # reverse_complement
            rev_aug = df.copy()
            rev_aug.sequence = rev_aug.sequence.apply(self.reverse_complement)
            df = pd.concat([df, rev_aug], ignore_index=True)

        targets = df[self.activity_column].to_numpy()
        seq = df.sequence.to_numpy()
        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

    def reverse_complement(self, seq: str, mapping=None) -> str:
        if mapping is None:
            mapping = {"A": "T", "G": "C", "T": "A", "C": "G", "N": "N"}

        try:
            return "".join(mapping[base] for base in reversed(seq.upper()))
        except KeyError as e:
            raise ValueError(f"Invalid character in sequence: {e}")

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
        # Rename columns to match bioframe schema
        data_df = df.copy()
        data_df = data_df.rename(
            columns={"chr": "chrom", "start": "start", "end": "end"}
        )

        # Convert to integer if possible
        for col in ["start", "end"]:
            if col in data_df.columns:
                data_df[col] = pd.to_numeric(data_df[col], errors="coerce").astype(
                    "Int64"
                )

        # Find intersections
        intersections = bf.overlap(data_df, regions_df, how="inner", return_index=True)

        if self.exclude_regions:
            # Exclude sequences that overlap with specified regions
            filtered_df = df[~df.index.isin(intersections["index"])]
        else:
            # Include only sequences that overlap with specified regions
            filtered_df = df[df.index.isin(intersections["index"])]

        return filtered_df

    def split_parse(self, split: str) -> str:
        """
        Parses the input split and returns a list of splits.

        Parameters
        ----------
        split : str
            Defines the data split, expected values: 'train', 'val', 'test'.

        Returns
        -------
        str
            A string containing the parsed split.
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}
        column = ""  # which column to consider

        # Process string input
        if isinstance(split, str):
            if split in valid_splits:
                column = "split"
                return [split], column
            elif split in self.LIST_OF_CHR:
                column = "chr"
                return [split], column
            else:
                raise ValueError(
                    f"Invalid fold value: {split}. Must be one of {self.LIST_OF_CHR}."
                )
        # Validate list of folds
        elif isinstance(split, list):
            result = []
            column = "chr"
            for item in split:
                if item in self.LIST_OF_CHR:
                    result.append(item)
                else:
                    raise ValueError(
                        f"Invalid fold value: {item}. Must be one of {self.LIST_OF_CHR}."
                    )
            return result, column
        else:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', 'test' or {self.LIST_OF_CHR}."
            )
