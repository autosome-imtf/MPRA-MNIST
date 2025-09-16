import pandas as pd
from typing import List, Union, Optional, Callable, Dict
from functools import partial
import os
import bioframe as bf
from mpramnist.mpradataset import MpraDataset


class MalinoisDataset(MpraDataset):
    # from boda dataset
    LEFT_FLANK = "ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG"
    RIGHT_FLANK = "CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT"

    CELL_TYPES = ["K562", "HepG2", "SKNSH"]

    FLAG = "Malinois"

    def __init__(
        self,
        split: str | List[Union[int, str]] | int,
        genomic_regions: Optional[Union[str, List[Dict]]] = None,
        exclude_regions: bool = False,
        genomic_regions_column: str = ["start.hg19", "stop.hg19"],
        filtration: str = "original",  # 'original', 'own' or 'none'
        activity_columns: List[str] = ["K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"],
        stderr_columns: List[str] = ["K562_lfcSE", "HepG2_lfcSE", "SKNSH_lfcSE"],
        data_project: List[str] = ["UKBB", "GTEX", "CRE"],
        sequence_column="sequence",
        duplication_cutoff: Optional[float] = None,
        stderr_threshold: float = 1.0,
        std_multiple_cut: float = 6.0,
        up_cutoff_move: float = 3.0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_original_reverse_complement: bool = False,
        root=None,
    ):
        """
        Initializes the dataset loader with specified filtering, duplication, and transformation settings.

        Parameters
        ----------
        split : Union[str, List[Union[int, str]], int]
            Specifies the data split to use (e.g., 'train', 'val', 'test', or list of indices).
        genomic_regions : str | List[Dict], optional
            Genomic regions to include/exclude. Can be:
            - Path to BED file
            - List of dictionaries with 'chrom', 'start', 'end' keys
        exclude_regions : bool
            If True, exclude the specified regions instead of including them.
        filtration : str
            Specifies the filtering method. Options are 'original', 'own', or 'none'.
        activity_columns : List[str]
            List of column names with activity data to be used for filtering and duplication.
        stderr_columns : List[str]
            List of column names with standard error values used for quality filtering.
        data_project : List[str]
            Specifies the data projects to include in filtering.
        duplication_cutoff : Optional[float]
            If specified, sequences with a maximum activity value above this threshold will be duplicated.
        stderr_threshold : float
            Maximum allowed standard error threshold for filtering rows.
        std_multiple_cut : float
            The multiplier applied to standard deviations to calculate the upper cut-off threshold.
        up_cutoff_move : float
            Shift value for adjusting the upper cut-off threshold.
        transform : Optional[Callable]
            Transformation function applied to each sequence object.
        target_transform : Optional[Callable]
            Transformation function applied to the target data.
        use_original_reverse_complement : bool
            Determines whether to generate the reverse complement of sequences using the same approach as the original study
        """
        # Validate filtration parameter
        if filtration not in {"original", "own", "none"}:
            raise ValueError("filtration must be one of {'original', 'own', 'none'}")

        super().__init__(split, root)

        # Assign attributes
        self.stderr_columns = stderr_columns
        self.sequence_column = sequence_column
        self.genomic_regions_column = genomic_regions_column
        self.data_project = data_project
        self.project_column = "data_project"
        self.chr_column = "chr"
        self.duplication_cutoff = duplication_cutoff
        self.stderr_threshold = stderr_threshold
        self.std_multiple_cut = std_multiple_cut
        self.up_cutoff_move = up_cutoff_move
        self.filtration = filtration
        self.use_original_reverse_complement = use_original_reverse_complement
        self.transform = transform
        self.target_transform = target_transform
        self.prefix = self.FLAG + "_"
        self.genomic_regions = genomic_regions
        self.exclude_regions = exclude_regions

        # Parse columns and split parameters
        activity_columns = self.activity_columns_parse(activity_columns)
        self.split = self.split_parse(split)

        # Load and process the dataset
        self.ds = self._load_and_filter_data(activity_columns)
        self._cell_type = activity_columns
        self.target = activity_columns  # initialization for MpraDataset.__getitem__()

        self.name_for_split_info = self.prefix

    def _load_and_filter_data(self, activity_columns):
        """
        Loads and preprocesses the dataset by selecting specific columns, handling missing values,
        renaming columns, filtering data based on project and filtration type, splitting by chromosome,
        and handling duplication.

        Parameters:
        -----------
        activity_columns : list of str
            List of columns containing activity data in the dataset.

        Returns:
        --------
        pd.DataFrame
            Filtered and processed DataFrame.

        Raises:
        -------
        FileNotFoundError
            If the specified data file is not found.
        ValueError
            If `filtration` is not one of ['original', 'own', 'none'].
        """

        try:
            file_name = self.prefix + "Table_S2" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t", low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Select columns and drop rows with NaN in critical columns
        columns = [
            self.sequence_column,
            *activity_columns,
            self.chr_column,
            self.project_column,
            *self.stderr_columns,
            *self.genomic_regions_column,
        ]
        df = df[columns].dropna()

        # Rename sequence column
        df.rename(columns={self.sequence_column: "seq"}, inplace=True)
        self.sequence_column = "seq"

        # Filter by project column
        df = df[df[self.project_column].isin(self.data_project)].reset_index(drop=True)

        # Apply filtration
        filters = {
            "original": lambda df: self._filter_data(
                df,
                activity_columns,
                self.stderr_columns,
                duplication_cutoff=self.duplication_cutoff,
                is_padding=True,
            ),
            "own": lambda df: self._filter_data(
                df,
                activity_columns,
                self.stderr_columns,
                self.data_project,
                self.duplication_cutoff,
                self.stderr_threshold,
                self.std_multiple_cut,
                self.up_cutoff_move,
            ),
            "none": lambda df: self._apply_none_filtration(df),
        }
        if self.filtration in filters:
            df = filters[self.filtration](df)
        else:
            raise ValueError("filtration value must be 'original', 'own' or 'none'")

        targets = df[activity_columns].to_numpy()
        seq = df.seq.to_numpy()
        df = {"targets": targets, "seq": seq}

        return df

    def _apply_none_filtration(self, df):
        """
        Apply filtration for the 'none' case with chromosome/genomic region splitting logic.
        """
        # Apply genomic region filtering if specified
        if self.genomic_regions is None:
            # Split data by chromosome if no genomic regions specified
            df = df[df[self.chr_column].isin(self.split)].reset_index(drop=True)
        else:
            df = self.filter_by_genomic_regions(df)
            self.split = "genomic region"

        return df

    def original_pad_seq(
        self,
        df,
        in_column_name,
        padded_seq_len=600,
        upStreamSeq=LEFT_FLANK,
        downStreamSeq=RIGHT_FLANK,
    ):
        sequence = df[in_column_name]
        origSeqLen = len(sequence)
        paddingLen = padded_seq_len - origSeqLen
        assert paddingLen <= (len(upStreamSeq) + len(downStreamSeq)), (
            "Not enough padding available"
        )
        if paddingLen > 0:
            if -paddingLen // 2 + paddingLen % 2 < 0:
                upPad = upStreamSeq[-paddingLen // 2 + paddingLen % 2 :]
            else:
                upPad = ""
            downPad = downStreamSeq[: paddingLen // 2 + paddingLen % 2]
            paddedSequence = upPad + sequence + downPad
            assert len(paddedSequence) == padded_seq_len, "Kiubo?"
            return paddedSequence
        else:
            return sequence

    def reverse_complement(self, seq: str, mapping=None) -> str:
        if mapping is None:
            mapping = {"A": "T", "G": "C", "T": "A", "C": "G", "N": "N"}

        try:
            return "".join(mapping[base] for base in reversed(seq.upper()))
        except KeyError as e:
            raise ValueError(f"Invalid character in sequence: {e}")

    def _filter_data(
        self,
        df: pd.DataFrame,
        activity_columns: List[str],
        stderr_columns=None,
        data_project=None,
        duplication_cutoff=None,
        stderr_threshold=1.0,
        std_multiple_cut=6.0,
        up_cutoff_move=3.0,
        is_padding=False,
    ) -> pd.DataFrame:
        """
        Filters the DataFrame based on specified thresholds for standard error and activity metrics.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing data to be filtered.
        activity_columns : List[str]
            List of columns containing activity data for threshold-based filtering.
        stderr_columns : List[str], optional
            List of columns to assess standard error. Defaults to specific columns if not provided.
        data_project : List[str], optional
            List of projects to filter by, though not directly used here; can be customized further if needed.
        duplication_cutoff : float, optional
            The cutoff threshold for handling duplicates.
        stderr_threshold : float, optional
            Threshold for standard error filtering; rows with max `stderr_columns` values above this are removed.
        std_multiple_cut : float, optional
            Multiplier for the standard deviation used to set upper and lower activity thresholds.
        up_cutoff_move : float, optional
            Value added to the upper cutoff threshold to allow further adjustment.
        is_padding : bool
            Determines whether filtration is original and need to padding sequences using the same approach as the original study

        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame based on specified thresholds.
        """
        # Set default values for optional parameters
        if stderr_columns is None:
            stderr_columns = ["K562_lfcSE", "HepG2_lfcSE", "SKNSH_lfcSE"]
        if data_project is None:
            data_project = ["UKBB", "GTEX", "CRE"]

        # Standard error threshold filtering
        quality_filter = df[stderr_columns].max(axis=1) < stderr_threshold
        df = df[quality_filter].reset_index(drop=True)

        # Calculate means and standard deviations for activity columns
        means = df[activity_columns].mean().to_numpy()
        stds = df[activity_columns].std().to_numpy()

        # Calculate upper and lower cutoffs
        up_cut = means + stds * std_multiple_cut + up_cutoff_move
        down_cut = means - stds * std_multiple_cut

        # Apply combined filtering for non-extreme values
        non_extremes_filter = (
            (df[activity_columns] < up_cut) & (df[activity_columns] > down_cut)
        ).all(axis=1)
        df = df[non_extremes_filter].reset_index(drop=True)

        # Apply genomic region filtering
        df = self.filter_by_genomic_regions(df)

        # Split data by chromosome
        if self.genomic_regions is None:
            df = df[df[self.chr_column].isin(self.split)].reset_index(drop=True)
        else:
            self.split = "genomic region"

        # Handle duplication if specified
        if duplication_cutoff is not None:
            df = self.duplicate_high_activity_rows(
                df, activity_columns, duplication_cutoff
            )

        if is_padding:
            # padding
            fn_padding = partial(
                self.original_pad_seq,
                in_column_name=self.sequence_column,
                padded_seq_len=600,
            )
            df[self.sequence_column] = df.apply(fn_padding, axis=1)

        if self.use_original_reverse_complement:
            # reverse_complement
            rev_aug = df.copy()
            rev_aug.seq = rev_aug.seq.apply(self.reverse_complement)
            df = pd.concat([df, rev_aug], ignore_index=True)

        return df

    def duplicate_high_activity_rows(
        self, df: pd.DataFrame, activity_columns: List[str], duplication_cutoff: float
    ) -> pd.DataFrame:
        """
        Duplicates rows in the DataFrame where the maximum value across specified activity columns
        exceeds a defined threshold. The original DataFrame is first sorted in descending order
        based on the maximum activity values.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame containing data to be filtered and duplicated.
        activity_columns : List[str]
            List of columns containing activity data used to determine rows for duplication.

        Returns:
        --------
        pd.DataFrame
            DataFrame with specified rows duplicated based on the maximum activity value.
        """

        max_values = df[activity_columns].max(axis=1)

        # Filter rows where the maximum activity value exceeds the duplication cutoff
        duplication_filter = max_values > duplication_cutoff
        duplicated_rows = df[duplication_filter].reset_index(drop=True)

        # Concatenate the duplicated rows with the original DataFrame
        df_with_duplicates = pd.concat([df, duplicated_rows], ignore_index=True)

        return df_with_duplicates

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
            columns={"chr": "chrom", "start.hg19": "start", "stop.hg19": "end"}
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

    def split_parse(self, split: list[Union[int, str]] | int | str) -> list[str]:
        """
        Parses the input split and returns a list of folds.
        """

        split_default = {
            "train": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "8",
                "9",
                "10",
                "11",
                "12",
                "14",
                "15",
                "16",
                "17",
                "18",
                "20",
                "22",
                "Y",
            ],
            "val": ["19", "21", "X"],
            "test": ["7", "13"],
        }  # default split of data

        list_of_chr = [str(i) for i in range(1, 23)] + ["X", "Y"]

        # Process string input for specific keys or fold names ("X", "Y")
        if isinstance(split, str):
            if split in list_of_chr:
                split = [split]
            elif split in split_default:
                split = split_default[split]
            else:
                raise ValueError(
                    f"Invalid split value: {split}. Expected 'train', 'val', or 'test', range 1-22 or 'X', 'Y'."
                )

        # int to list for unified processing
        elif isinstance(split, int):
            split = [str(split)]

        # Validate list of folds
        elif isinstance(split, list):
            result = []
            for item in split:
                if isinstance(item, int) and 1 <= item <= 22:
                    result.append(str(item))
                elif isinstance(item, str) and item in list_of_chr:
                    result.append(item)
                else:
                    raise ValueError(
                        f"Invalid fold value: {item}. Must be in range 1-22 or 'X'/'Y'."
                    )

            split = result  # Ensure final result is clean and validated

        return split

    def activity_columns_parse(
        self,
        activity_columns: str | List[str],
        default_activity_columns=["K562", "HepG2", "SKNSH"],
    ) -> List[str]:
        """
        Parses the input activity columns and returns a list of parsed activity_columns.
        """
        if isinstance(activity_columns, str):
            activity_columns = [activity_columns]
        if isinstance(activity_columns, List):
            for i in range(len(activity_columns)):
                act = activity_columns[i]
                act = act.split("_")[0]
                if act not in default_activity_columns:
                    raise ValueError(
                        f"Invalid activity column: {act}. Must be one of {default_activity_columns}."
                    )
                activity_columns[i] = act + "_log2FC"
                # activity_columns[i] = act + "_mean"
        return activity_columns
