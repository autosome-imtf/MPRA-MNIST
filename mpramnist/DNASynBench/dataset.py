import pandas as pd
import os

from mpramnist.DNASynBench.custom_datasets import Bench_Dataset
from mpramnist.DNASynBench.tasks import task_2
from mpramnist.mpradataset import MpraDataset


class DNASynDataset(MpraDataset):
    FLAG = "DNASyn"

    def __init__(self, split: str, args: dict, transform=None, target_transform=None):
        super().__init__(split)

        self.activity_columns = "label"
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"

        df = task_2(**args).reset_index(drop=True)

        df = df[df["split"].isin(self.split)].reset_index(drop=True)

        self.df = df
        targets = self.df[self.activity_columns].to_numpy()
        seq = self.df.sequence.to_numpy()

        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

    def split_parse(self, split: str) -> str:
        # Default valid splits
        valid_splits = {"train", "val", "test"}

        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )

        return [split]
