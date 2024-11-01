
import os
from os.path import expanduser
import warnings


def get_default_root():
    home = expanduser("~")
    dirpath = os.path.join(home, ".mpradataset")

    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    except:
        warnings.warn("Failed to setup default root.")
        dirpath = None

    return dirpath


DEFAULT_ROOT = get_default_root()

HOMEPAGE = "https://github.com/autosome-imtf/mpraMnist"

INFO = {
    "...": {},
    "VikramDataset": {
        "python_class": "VikramDataset",
        "description": "The VikramDataset is based on lentiMPRA assay, which determines the regulatory activity of over 680,000 sequences, representing a nearly comprehensive set of all annotated CREs among three cell types (HepG2, K562, and WTC11). HepG2 is a human liver cancer cell line, K562 is myelogenous leukemia cell line, WTC11 is pluripotent stem cell line derived from adult skin ",
        "url": "-",
        "target_columns": {"expression", "averaged_expression"},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {"train": 98336, "val": 12292, "test": 12292},
        "folds": {"train": "1, 2, 3, 4, 5, 6, 7, 8", "val": 9, "test": 10}
    },
    "MalinoisDataset": {
        "python_class": "MalinoisDataset",
        "description": "MalinoisDataset is based on ",
        "url": "-",
        "target_columns": {"K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {"train": 1, "val": 1, "test": 1},
        "folds": {"train": "1, 2, 3, 4, 5, 6, 7, 8", "val": 9, "test": 10}
    },
}