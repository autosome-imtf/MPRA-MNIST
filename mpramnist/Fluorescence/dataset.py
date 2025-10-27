import pandas as pd
from typing import List
import os

from mpramnist.mpradataset import MpraDataset


class FluorescenceDataset(MpraDataset):
    """
    Dataset class for Fluorescence MPRA data.

    This class extends MpraDataset to handle fluorescence-based MPRA data from
    different cell types. It supports regression tasks with fluorescence intensity
    measurements across multiple cell lines.

    This implementation and preprocessed data are adapted from:
    https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/FluorescenceData_classification.py

    Attributes
    ----------
    CELL_TYPES : list of str
        Available cell types in the dataset: ["JURKAT", "K562", "THP1"].
    FLAG : str
        Dataset identifier flag, set to "Fluorescence".

    Examples
    --------
    >>> # Basic usage with default settings (all cell types)
    >>> dataset = FluorescenceDataset(split='train')
    >>> len(dataset)
    10000
    
    >>> # Specific cell type for regression
    >>> dataset = FluorescenceDataset(
    ...     split='val',
    ...     cell_type='K562',
    ...     task='regression'
    ... )
    >>> sequence, target = dataset[0]
    >>> print(sequence.shape, target.shape)
    (1000,) (1,)

    Notes
    -----
    - Currently only supports regression tasks
    - Fluorescence measurements represent protein expression levels
    - Data is collected from three different human cell lines
    - Sequences are typically promoter or regulatory elements
    """

    CELL_TYPES = ["JURKAT", "K562", "THP1"]
    FLAG = "Fluorescence"

    def __init__(
        self,
        split: str,
        cell_type: str | List[str] = [
            "JURKAT",
            "K562",
            "THP1",
        ],  # all three cell types is default
        task: str = "regression", # only regression is now available
        transform=None,
        target_transform=None,
        root=None,
    ):
        """
        Initialize FluorescenceDataset instance.

        Attributes
        ----------
        split : str
            Defines which data split to use. Must be one of: 'train', 'val', 'test'.
            Determines which dataset file to load (e.g., 'Fluorescence_train.tsv').
        cell_type : str | List[str], optional, default=["JURKAT", "K562", "THP1"]
            Cell type(s) to include in the dataset. Can be a single cell type string
            or list of multiple cell types. All three cell types are included by default.
        task : str, optional, default="regression"
            Specifies the machine learning task. Currently only "regression" is supported.
            Classification may be added in future versions.
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
            If `cell_type` contains invalid cell type names not in CELL_TYPES.
            If `task` is not 'regression' (other tasks not yet implemented).
        FileNotFoundError
            If the dataset file for the specified split cannot be found or downloaded.
        KeyError
            If specified cell type columns are not found in the loaded dataframe.

        Notes
        -----
        - For regression tasks, column names are prefixed with 'numerical_'
        - Fluorescence values represent relative expression levels
        - Multiple cell types result in multi-output regression
        - Dataset files are automatically downloaded if not present
        """
        super().__init__(split, root)

        if isinstance(cell_type, str):
            if cell_type not in self.CELL_TYPES:
                raise ValueError(
                    f"Invalid cell_type: {cell_type}. Must be one of {self.CELL_TYPES}."
                )
            cell_type = [cell_type]
        if isinstance(cell_type, List):
            for i in range(len(cell_type)):
                act = cell_type[i]
                if act not in self.CELL_TYPES:
                    raise ValueError(
                        f"Invalid cell_type: {act}. Must be one of {self.CELL_TYPES}."
                    )

        self.cell_type = cell_type
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
            self.cell_type = ["numerical_" + i for i in self.cell_type]
        targets = df[self.cell_type].to_numpy()
        seq = df.sequence.to_numpy()
        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

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
        >>> dataset = FluorescenceDataset(split='train')
        >>> dataset.split_parse('val')
        'val'
        
        >>> dataset.split_parse('invalid')
        ValueError: Invalid split value: invalid. Expected 'train', 'val', or 'test'.

        Notes
        -----
        - This method is called automatically during initialization
        - Split validation ensures the correct data file is loaded
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}

        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )

        return split
