import pandas as pd
from typing import List
import os

from mpramnist.mpradataset import MpraDataset


class FluorescenceDataset(MpraDataset):
    CELL_TYPES = ["JURKAT", "K562", "THP1"]
    FLAG = "Fluorescence"

    def __init__(
        self,
        split: str,
        activity_columns: str | List[str] = [
            "JURKAT",
            "K562",
            "THP1",
        ],  # all three cell types is default
        task: str = "regression",
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Attributes
        ----------
        split : str
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        activity_columns : str | List[str]
            Specifies the cell type for filtering the data.
        task : str
            Defines regression or classification task
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)
        self._cell_type = activity_columns

        if isinstance(activity_columns, str):
            if activity_columns not in self.CELL_TYPES:
                raise ValueError(
                    f"Invalid activity_columns: {activity_columns}. Must be one of {self.CELL_TYPES}."
                )
            activity_columns = [activity_columns]
        if isinstance(activity_columns, List):
            for i in range(len(activity_columns)):
                act = activity_columns[i]
                if act not in self.CELL_TYPES:
                    raise ValueError(
                        f"Invalid activity_columns: {act}. Must be one of {self.CELL_TYPES}."
                    )

        self.activity_columns = activity_columns
        self.task = task
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"

        try:
            file_name = self.prefix + self.split + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        if task == "regression":
            self.activity_columns = ["numerical_" + i for i in self.activity_columns]
        targets = df[self.activity_columns].to_numpy()
        seq = df.sequence.to_numpy()
        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

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

        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )

        return split
