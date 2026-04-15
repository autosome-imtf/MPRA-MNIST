import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import os

from mpramnist.mpradataset import MpraDataset


class EvfratovDataset(MpraDataset):
    """
    Dataset class for Evfratov MPRA data.

    This class extends MpraDataset to handle ribosomal profiling data from
    E. coli experiments. The dataset contains sequence-activity relationships
    for translation efficiency measurements with classification tasks.

    Attributes
    ----------
    FLAG : str
        Dataset identifier flag, set to "Evfratov".

    Examples
    --------
    >>> # Basic usage with 23-length sequences
    >>> dataset = EvfratovDataset(split='train', length_of_seq=23)
    >>> len(dataset)
    5000
    
    >>> # With merged classes for balanced classification
    >>> dataset = EvfratovDataset(
    ...     split='val',
    ...     length_of_seq=33,
    ...     merge_last_classes=True
    ... )
    >>> sequence, target = dataset[0]
    >>> print(sequence.shape, target.shape)
    (33,) (1,)

    Notes
    -----
    - Original data contains 8 classes, can be reduced to 7 by merging last two classes
    """

    FLAG = "Evfratov"

    def __init__(
        self,
        split: str,
        length_of_seq: Union[str, int] = "23",  # 23 or 33
        merge_last_classes: bool = False,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize EvfratovDataset instance.

        Attributes
        ----------
        split : str
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        length_of_seq: str | int
            Length of sequences to use. Must be either "23" or "33" (or corresponding integers).
            Determines which dataset variant to load (23bp or 33bp sequences).
        merge_last_classes : bool
            A flag indicating whether to merge the last two classes in the dataset.
            If True, the last two classes (typically those with the fewest examples)
            will be merged into one. This can be useful for addressing class imbalance
            or simplifying the classification task. If False, the classes remain
            unchanged. By default, it is recommended to set this to False unless the
            user is certain that merging is necessary.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        root : str, optional, default=None
            Root directory where dataset files are stored or should be downloaded.
            If None, uses the default dataset directory from parent class.

        Raises
        ------
        ValueError
            If `split` parameter is not 'train', 'val', or 'test'.
            If `length_of_seq` is not "23" or "33" (or equivalent integers).
        FileNotFoundError
            If the dataset file for the specified split and sequence length cannot be found.

        Notes
        -----
        - Raw count data is converted to probability distributions before label assignment
        - Labels are assigned based on the bin with maximum expression
        """
        super().__init__(split, root)

        self.activity_columns = "label"
        self.cell_type = "The JM109 E. coli strain"
        self.length_of_seq = str(length_of_seq)
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"

        try:
            file_name = self.prefix + self.length_of_seq + "_" + self.split + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        sequences = df["sequence"]
        df_counts = df.drop(columns=["sequence"])

        # Convert counts to distributions
        row_sums = df_counts.sum(axis=1)
        df = df_counts.div(row_sums, axis=0).fillna(
            0
        )  #  change NaN to 0 if raw sum is 0

        # Assign labels based on the column with the maximum value
        df[self.activity_columns] = df.idxmax(axis=1)
        df[self.activity_columns] = df[self.activity_columns].apply(
            lambda x: df.columns.get_loc(x)
        )

        df["sequence"] = sequences

        # Merge last two classes if required
        if merge_last_classes:
            df[self.activity_columns] = df[self.activity_columns].replace(7, 6)
            self.n_classes = 7
        else:
            self.n_classes = 8

        # Replace 'U' with 'T' in sequences
        df["sequence"] = df["sequence"].str.upper().str.replace("U", "T")

        self.df = df
        targets = self.df[self.activity_columns].to_numpy()
        seq = self.df.sequence.to_numpy()

        self.ds = {"targets": targets, "seq": seq}
        
        self.name_for_split_info = self.prefix + self.length_of_seq + "_"

    def hist_plot(self):
        """
        Plot a histogram of class label distribution.

        Creates a bar plot showing the count of samples in each class.
        Useful for visualizing class imbalance and dataset composition.

        Examples
        --------
        >>> dataset = EvfratovDataset(split='train', length_of_seq=23)
        >>> dataset.hist_plot()
        # Displays histogram with class counts

        Returns
        -------
        None
            Displays matplotlib plot directly.

        Notes
        -----
        - Plot includes exact count numbers above each bar
        - Uses skyblue bars with black edges
        - Includes grid lines for better readability
        - Title indicates it's a histogram of label counts
        """
        data = self.df.label
        n_classes = self.n_classes

        counts, bins = np.histogram(data, bins=n_classes, range=(0, n_classes))

        x = np.arange(n_classes)
        plt.bar(x, counts, color="skyblue", edgecolor="black")

        for i, count in enumerate(counts):
            plt.text(
                x[i],
                count,
                int(count),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.xlabel("Labels")
        plt.ylabel("Quantity")
        plt.title("Histogram count of label")

        plt.xticks(x)

        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.show()

    def split_parse(self, split: str) -> str:
        """
        Parse and validate the split parameter.

        Validates that the provided split string is one of the allowed values
        and returns the validated split identifier.

        Parameters
        ----------
        split : str
            Data split identifier. Must be one of: 'train', 'val', 'test'.

        Returns
        -------
        str
            Validated split string.

        Raises
        ------
        ValueError
            If split is not one of the allowed values ('train', 'val', 'test').

        Examples
        --------
        >>> dataset = EvfratovDataset(split='train', length_of_seq=23)
        >>> dataset.split_parse('val')
        'val'
        
        >>> dataset.split_parse('invalid')
        ValueError: Invalid split value: invalid. Expected 'train', 'val', or 'test'.

        Notes
        -----
        - This method is called automatically during initialization
        - Split validation ensures the correct data file is loaded
        - Valid splits are hardcoded as {'train', 'val', 'test'}
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}

        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )

        return split
