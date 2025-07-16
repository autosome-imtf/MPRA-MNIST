__version__ = "0.1.2"

import os
from os.path import expanduser
import warnings


def get_default_root():
    home = expanduser("~")
    dirpath = os.path.join(home, ".mpramnist", "data", "")

    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    except:
        warnings.warn("Failed to setup default root.")
        dirpath = None

    return dirpath


DEFAULT_ROOT = get_default_root()

HOMEPAGE = "https://github.com/autosome-imtf/MPRA-MNIST"

INFO = {
    "Agarwal": {
        "python_class": "Agarwal",
        "url_Agarwal_HepG2.tsv": "https://zenodo.org/records/15228546/files/Agarwal_HepG2.tsv?download=1",
        "MD5_Agarwal_HepG2.tsv": "e34ead50382f11dbe3537bd66399548b",
        "url_Agarwal_K562.tsv": "https://zenodo.org/records/15228546/files/Agarwal_K562.tsv?download=1",
        "MD5_Agarwal_K562.tsv": "4b4235ef795a41f95adbb267d651b43e",
        "url_Agarwal_WTC11.tsv": "https://zenodo.org/records/15228546/files/Agarwal_WTC11.tsv?download=1",
        "MD5_Agarwal_WTC11.tsv": "4792c240248fd7cd69ed9e1575610fe4",
        "target_columns": {"expression", "averaged_expression"},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {"train": 98336, "val": 12292, "test": 12292},
        "folds": {"train": "1, 2, 3, 4, 5, 6, 7, 8", "val": 9, "test": 10}
    },
    "AgarwalJoint": {
        "python_class": "AgarwalJoint",
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
    "Malinois": {
        "python_class": "Malinois",
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
    "StarrSeq": {
        "python_class": "StarrSeq",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Sure": {
        "python_class": "Sure",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Sharpr": {
        "python_class": "Sharpr",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Fluorescence": {
        "python_class": "Fluorescence",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Evfratov": {
        "python_class": "Evfratov",
        "description": "Evfratov is based on ",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "DeepPromoter": {
        "python_class": "DeepPromoter",
        "description": "DeepPromoter is based on ",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "DeepStarr": {
        "python_class": "DeepStarr",
        "description": "DeepStarr is based on ",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Kircher": {
        "python_class": "Kircher",
        "description": "Kircher is based on ",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Dream": {
        "python_class": "Dream",
        "description": "Dream is based on ",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Vaishnav": {
        "python_class": "Vaishnav",
        "description": "Vaishnav is based on ",
        "url": "",
        "MD5": "",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
}