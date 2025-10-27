import pandas as pd
from typing import List
import os

from mpramnist.mpradataset import MpraDataset


class SharprDataset(MpraDataset):
    """
    Dataset class for Sharpr MPRA (Massively Parallel Reporter Assay) data.
    
    This class extends MpraDataset to handle Sharpr MPRA data with
    multiple cell types and promoters. It provides access to sequence data
    and corresponding activity measurements from K562 and HepG2 cell lines
    with minP and SV40 promoters.

    This implementation and preprocessed data are adapted from:
    https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/Sharpr_MPRA.py
    
    Attributes
    ----------
    CELL_TYPES : list of str
        Predefined list of all available activity measurement columns in the dataset.
        Includes replicate measurements and averages for:
        - K562 cell line with minP promoter (rep1, rep2, avg)
        - K562 cell line with SV40 promoter (rep1, rep2, avg) 
        - HepG2 cell line with minP promoter (rep1, rep2, avg)
        - HepG2 cell line with SV40 promoter (rep1, rep2, avg)
    FLAG : str
        Dataset identifier flag, set to "Sharpr".
    
    Parameters
    ----------
    split : str
        Defines which data split to use. Expected values: 'train', 'val', 'test'.
    cell_type : List[str]
        List of column names with activity data to be used as targets.
        Must be a subset of CELL_TYPES.
    transform : callable, optional
        Transformation function applied to each sequence sample.
        If provided, will be applied to sequences during data loading.
    target_transform : callable, optional
        Transformation function applied to target activity values.
        If provided, will be applied to targets during data loading.
    root : str, optional
        Root directory where dataset is stored. If None, uses default location.
    
    Raises
    ------
    ValueError
        If provided split value is not 'train', 'val', or 'test'.
    FileNotFoundError
        If the requested data file cannot be found in the dataset directory.
    
    Examples
    --------
    >>> dataset = SharprDataset(
    ...     split='train',
    ...     cell_type=['k562_minp_avg', 'hepg2_minp_avg']
    ... )
    >>> len(dataset)
    15000
    >>> sequence, target = dataset[0]
    >>> print(sequence.shape, target.shape)
    (1000,) (2,)
    
    Notes
    -----
    - The dataset automatically downloads data files if not present
    - Activity measurements are typically log-transformed values
    - Multiple activity columns can be used for multi-task learning
    """

    CELL_TYPES = [
        "k562_minp_rep1", "k562_minp_rep2", "k562_minp_avg",
        "k562_sv40p_rep1", "k562_sv40p_rep2", "k562_sv40p_avg",
        "hepg2_minp_rep1", "hepg2_minp_rep2", "hepg2_minp_avg",
        "hepg2_sv40p_rep1", "hepg2_sv40p_rep2", "hepg2_sv40p_avg",
    ]
    FLAG = "Sharpr"

    def __init__(
        self,
        split: str,
        cell_type: List[str] = None,
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize SharprDataset instance.

        Attributes
        ----------
        split : str
            Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).
        cell_type : List[str], default None
            Cell types or List of column names with activity data to be used as targets.
            Must be a subset of CELL_TYPES.
        transform : callable, optional
            Transformation applied to each sequence object.
        target_transform : callable, optional
            Transformation applied to the target data.
        """
        super().__init__(split, root)

        self.transform = transform
        self.target_transform = target_transform
        self.cell_type = cell_type
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"

        if self.cell_type is None:
            self.cell_type = self.CELL_TYPES
            
        try:
            file_name = self.prefix + self.split + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        targets = df[self.cell_type].to_numpy()
        seq = df.seq.to_numpy()
        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

    def split_parse(self, split: str) -> str:
        """
        Parse and validate the split parameter.
        
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
            If split is not one of the allowed values.
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}

        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )

        return split
