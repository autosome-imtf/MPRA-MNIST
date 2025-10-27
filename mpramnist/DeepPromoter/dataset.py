import pandas as pd
import os

from mpramnist.mpradataset import MpraDataset


class DeepPromoterDataset(MpraDataset):
    """
    Dataset class for DeepPromoter MPRA (Massively Parallel Reporter Assay) data.

    This class extends MpraDataset to handle E. coli promoter classification data.
    The dataset contains sequence-activity relationships
    for translation efficiency measurements with regression tasks.

    Attributes
    ----------
    FLAG : str
        Dataset identifier flag, set to "DeepPromoter".

    Examples
    --------
    >>> # Basic usage for training
    >>> dataset = DeepPromoterDataset(split='train')
    >>> len(dataset)
    10000
    
    >>> # Access individual samples
    >>> sequence, target = dataset[0]
    >>> print(sequence.shape, target.shape)
    (300,) (1,)
    >>> print(f"Sequence: {sequence}, Target: {target}")
    Sequence: ATGCGTA..., Target: 1

    Notes
    -----
    - Sequences are from E. coli DH5α strain
    - All data is stored in a single file with split information in a column
    - Sequence length is typically 50 bp
    """

    FLAG = "DeepPromoter"

    def __init__(self, split: str, transform=None, target_transform=None, root=None):
        """
        Initialize DeepPromoterDataset instance.
        
        Attributes
        ----------
        split : str
            Defines which data split to use. Must be one of: 'train', 'val', 'test'.
            The dataset filters sequences based on the 'split' column in the data file.
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
        FileNotFoundError
            If the main dataset file cannot be found or downloaded.
        KeyError
            If required columns ('split', 'target', 'sequence') are missing from the data file.

        Notes
        -----
        - All data is loaded from a single file: 'DeepPromoter_all_seqs.tsv'
        - The 'split' column in the data file determines train/val/test assignment
        """
        super().__init__(split, root)

        self.activity_columns = "target"
        self.cell_type = "The DH5α E. coli strain"
        self.transform = transform
        self.target_transform = target_transform
        self.split = self.split_parse(split)
        self.prefix = self.FLAG + "_"

        try:
            file_name = self.prefix + "all_seqs" + ".tsv"
            self.download(self._data_path, file_name)
            file_path = os.path.join(self._data_path, file_name)
            df = pd.read_csv(file_path, sep="\t")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        df = df[df["split"].isin(self.split)].reset_index(drop=True)

        self.df = df
        targets = self.df[self.activity_columns].to_numpy()
        seq = self.df.sequence.to_numpy()

        self.ds = {"targets": targets, "seq": seq}

        self.name_for_split_info = self.prefix

    def split_parse(self, split: str) -> str:
        """
        Parse and validate the split parameter.

        Validates that the provided split string is one of the allowed values
        and returns it as a list for compatibility with the filtering mechanism.

        Parameters
        ----------
        split : str
            Data split identifier. Must be one of: 'train', 'val', 'test'.

        Returns
        -------
        List[str]
            List containing the validated split string.

        Raises
        ------
        ValueError
            If split is not one of the allowed values ('train', 'val', 'test').

        Examples
        --------
        >>> dataset = DeepPromoterDataset(split='train')
        >>> dataset.split_parse('val')
        ['val']
        
        >>> dataset.split_parse('train')
        ['train']
        
        >>> dataset.split_parse('invalid')
        ValueError: Invalid split value: invalid. Expected 'train', 'val', or 'test'.

        Notes
        -----
        - Returns a list to maintain compatibility with pandas isin() filtering
        - The original data file contains a 'split' column with these values
        - This method ensures only valid splits are requested
        """

        # Default valid splits
        valid_splits = {"train", "val", "test"}

        # Process string input
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split value: {split}. Expected 'train', 'val', or 'test'."
            )

        return [split]
