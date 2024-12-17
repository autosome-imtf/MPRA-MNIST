__version__ = "0.1.2"

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
    "VikramDataset": {
        "python_class": "VikramDataset",
        "description": "The VikramDataset is based on lentiMPRA assay, which determines the regulatory activity of over 680,000 sequences, representing a nearly comprehensive set of all annotated CREs among three cell types (HepG2, K562, and WTC11). HepG2 is a human liver cancer cell line, K562 is myelogenous leukemia cell line, WTC11 is pluripotent stem cell line derived from adult skin ",
        "url_HepG2": "https://zenodo.org/api/records/14021416/draft/files/VikramDataset_HepG2.tsv?download=1",
        "MD5_HepG2": "e34ead50382f11dbe3537bd66399548b",
        "url_K562": "https://zenodo.org/api/records/14021416/draft/files/VikramDataset_WTC11.tsv?download=1",
        "MD5_K562": "4792c240248fd7cd69ed9e1575610fe4",
        "url_WTC11": "https://zenodo.org/api/records/14021416/draft/files/VikramDataset_K562.tsv?download=1",
        "MD5_WTC11": "4b4235ef795a41f95adbb267d651b43e",
        "target_columns": {"expression", "averaged_expression"},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {"train": 98336, "val": 12292, "test": 12292},
        "folds": {"train": "1, 2, 3, 4, 5, 6, 7, 8", "val": 9, "test": 10}
    },
    "VikramJointDataset": {
        "python_class": "VikramJointDataset",
        "description": "",
        "url_HepG2": "",
        "MD5_HepG2": "",
        "url_K562": "",
        "MD5_K562": "",
        "url_WTC11": "",
        "MD5_WTC11": "",
        "target_columns": {"HepG2","K562","WTC11"},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {"train": 1, "val": 1, "test": 1},
        "folds": {"train": "1, 2, 3, 4, 5, 6, 7, 8", "val": 9, "test": 10}
    },
    "MalinoisDataset": {
        "python_class": "MalinoisDataset",
        "description": "MalinoisDataset is based on ",
        "url": "https://zenodo.org/api/records/14021416/draft/files/MalinoisDataset.tsv?download=1",
        "MD5": "f45e8658736b545f02d331cb61fc0978",
        "target_columns": {"K562_log2FC", "HepG2_log2FC", "SKNSH_log2FC"},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {"train": 668946, "val": 62406, "test": 66712},
        "folds": {"train": "1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 22, Y",
                  "val": "19, 21, X", 
                  "test": "7, 13"
                 }
    },
}