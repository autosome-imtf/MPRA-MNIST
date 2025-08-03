__version__ = "0.1.0"

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
        "url_Agarwal_HepG2.tsv": "https://zenodo.org/records/15195607/files/Agarwal_HepG2.tsv?download=1",
        "MD5_Agarwal_HepG2.tsv": "1a580a4aa2914db345ac313b5b757872",
        "url_Agarwal_K562.tsv": "https://zenodo.org/records/15195607/files/Agarwal_K562.tsv?download=1",
        "MD5_Agarwal_K562.tsv": "cff66778255d1eec8f6b0ee4ac6ecda4",
        "url_Agarwal_WTC11.tsv": "https://zenodo.org/records/15195607/files/Agarwal_WTC11.tsv?download=1",
        "MD5_Agarwal_WTC11.tsv": "85702456de2f26fc9e5f062580e34340",
        "target_columns": {"expression", "averaged_expression"},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {"train": 98336, "val": 12292, "test": 12292},
        "folds": {"train": "1, 2, 3, 4, 5, 6, 7, 8", "val": 9, "test": 10}
    },
    "AgarwalJoint": {
        "python_class": "AgarwalJoint",
        "url_AgarwalJoint_joint_data.tsv": "https://zenodo.org/records/15195607/files/AgarwalJoint_joint_data.tsv?download=1",
        "MD5_AgarwalJoint_joint_data.tsv": "1cef844477c3d51bf9b7a3389a5f3c1a",
        "target_columns": {"HepG2","K562","WTC11"},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {"train": 1, "val": 1, "test": 1},
        "folds": {"train": "1, 2, 3, 4, 5, 6, 7, 8", "val": 9, "test": 10}
    },
    "Malinois": {
        "python_class": "Malinois",
        "url_Malinois_Table_S2.tsv": "https://zenodo.org/records/15195607/files/Malinois_Table_S2.tsv?download=1",
        "MD5_Malinois_Table_S2.tsv": "f45e8658736b545f02d331cb61fc0978",
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
        "url_StarrSeq_ATACSeq_all_chr_file.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_ATACSeq_all_chr_file.fasta.gz?download=1",
        "MD5_StarrSeq_ATACSeq_all_chr_file.fasta.gz": "739c90c58c93194a5df818d98d00691e",
        "url_StarrSeq_binary_test_enhancer.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_test_enhancer.fasta.gz?download=1",
        "MD5_StarrSeq_binary_test_enhancer.fasta.gz": "ed01b85d47506b64eb464bb4b09e389b",
        "url_StarrSeq_binary_test_promoter.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_test_promoter.fasta.gz?download=1",
        "MD5_StarrSeq_binary_test_promoter.fasta.gz": "c7561f6bc977693079d38e00f22378ce",
        "url_StarrSeq_binary_train_enhancer.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_train_enhancer.fasta.gz?download=1",
        "MD5_StarrSeq_binary_train_enhancer.fasta.gz": "3e474ac6c6d87a05859ada8e5b7dd1c1",
        "url_StarrSeq_binary_train_enhancer_from_input.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_train_enhancer_from_input.fasta.gz?download=1",
        "MD5_StarrSeq_binary_train_enhancer_from_input.fasta.gz": "8d72efc718a314d63b77b46a6da8ec79",
        "url_StarrSeq_binary_train_enhancer_permutated.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_train_enhancer_permutated.fasta.gz?download=1",
        "MD5_StarrSeq_binary_train_enhancer_permutated.fasta.gz": "c347a66f16211559e049de9c4e80c33d",
        "url_StarrSeq_binary_train_promoter.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_train_promoter.fasta.gz?download=1",
        "MD5_StarrSeq_binary_train_promoter.fasta.gz": "90173f2f042416e2428105b2920b98b8",
        "url_StarrSeq_binary_train_promoter_from_input.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_train_promoter_from_input.fasta.gz?download=1",
        "MD5_StarrSeq_binary_train_promoter_from_input.fasta.gz": "8494ca5292a6d9445052eb99cc856b00",
        "url_StarrSeq_binary_val_enhancer.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_val_enhancer.fasta.gz?download=1",
        "MD5_StarrSeq_binary_val_enhancer.fasta.gz": "ede1ad8bec2f633e99b7e6f9d6866981",
        "url_StarrSeq_binary_val_promoter.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_binary_val_promoter.fasta.gz?download=1",
        "MD5_StarrSeq_binary_val_promoter.fasta.gz": "1b7a6177120a4f875f3374ba82c8e6f0",
        "url_StarrSeq_CaptProm_test.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_CaptProm_test.fasta.gz?download=1",
        "MD5_StarrSeq_CaptProm_test.fasta.gz": "9596d4755fe6ddf0039d81603106218d",
        "url_StarrSeq_CaptProm_train.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_CaptProm_train.fasta.gz?download=1",
        "MD5_StarrSeq_CaptProm_train.fasta.gz": "e0235ba235742d362e8c4b365a036628",
        "url_StarrSeq_CaptProm_val.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_CaptProm_val.fasta.gz?download=1",
        "MD5_StarrSeq_CaptProm_val.fasta.gz": "40aa4578124ceeb09c84c76f51c7d0bd",
        "url_StarrSeq_genEnh_all_chr_file.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_genEnh_all_chr_file.fasta.gz?download=1",
        "MD5_StarrSeq_genEnh_all_chr_file.fasta.gz": "6332a6dd899f24d6612de4963af6c52e",
        "url_StarrSeq_genProm_test.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_genProm_test.fasta.gz?download=1",
        "MD5_StarrSeq_genProm_test.fasta.gz": "b06811e1fff61dfaaee0cac17a453b35",
        "url_StarrSeq_genProm_train.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_genProm_train.fasta.gz?download=1",
        "MD5_StarrSeq_genProm_train.fasta.gz": "b52afbe57a66e4c4f4381a4e8ffb291e",
        "url_StarrSeq_genProm_val.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_genProm_val.fasta.gz?download=1",
        "MD5_StarrSeq_genProm_val.fasta.gz": "5f783a547346be2f0ba41530a42125da",
        "url_StarrSeq_ranEnh_test.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_ranEnh_test.fasta.gz?download=1",
        "MD5_StarrSeq_ranEnh_test.fasta.gz": "0bffb1fa925f676f5b0b20d7f3aa12fe",
        "url_StarrSeq_ranEnh_train.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_ranEnh_train.fasta.gz?download=1",
        "MD5_StarrSeq_ranEnh_train.fasta.gz": "1d90ec94f6d69ae74ad36da4ff046e1e",
        "url_StarrSeq_ranEnh_val.fasta.gz": "https://zenodo.org/records/15195607/files/StarrSeq_ranEnh_val.fasta.gz?download=1",
        "MD5_StarrSeq_ranEnh_val.fasta.gz": "ab00ecd6dbeb1e494a40d64cb9719735",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Sure": {
        "python_class": "Sure",
        "url_Sure_SuRE42_HG02601_test.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE42_HG02601_test.tsv?download=1",
        "MD5_Sure_SuRE42_HG02601_test.tsv": "ef284e63d82e655d437a02e2651f07e4",
        "url_Sure_SuRE42_HG02601_train.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE42_HG02601_train.tsv?download=1",
        "MD5_Sure_SuRE42_HG02601_train.tsv": "828b98606ccff434e5bc2dd03e538b7b",
        "url_Sure_SuRE42_HG02601_val.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE42_HG02601_val.tsv?download=1",
        "MD5_Sure_SuRE42_HG02601_val.tsv": "f0cb2000c9de76e7bd6ba6d5d0c166a7",
        "url_Sure_SuRE43_GM18983_test.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE43_GM18983_test.tsv?download=1",
        "MD5_Sure_SuRE43_GM18983_test.tsv": "dd738f265679d07c9f2154b5ee78cb6f",
        "url_Sure_SuRE43_GM18983_train.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE43_GM18983_train.tsv?download=1",
        "MD5_Sure_SuRE43_GM18983_train.tsv": "016e868c44cd92337202849e0189cc89",
        "url_Sure_SuRE43_GM18983_val.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE43_GM18983_val.tsv?download=1",
        "MD5_Sure_SuRE43_GM18983_val.tsv": "dabf840978cd813e3c008d949782d4f0",
        "url_Sure_SuRE44_HG01241_test.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE44_HG01241_test.tsv?download=1",
        "MD5_Sure_SuRE44_HG01241_test.tsv": "c209f06f593cb68136a00d33c5115d94",
        "url_Sure_SuRE44_HG01241_train.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE44_HG01241_train.tsv?download=1",
        "MD5_Sure_SuRE44_HG01241_train.tsv": "e5cfd02413046ae710a1415369f8d0d9",
        "url_Sure_SuRE44_HG01241_val.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE44_HG01241_val.tsv?download=1",
        "MD5_Sure_SuRE44_HG01241_val.tsv": "60d62ab39a0a3f88169637b54afec321",
        "url_Sure_SuRE45_HG03464_test.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE45_HG03464_test.tsv?download=1",
        "MD5_Sure_SuRE45_HG03464_test.tsv": "8370581d54fcd3eef4af7e5fe58fdf74",
        "url_Sure_SuRE45_HG03464_train.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE45_HG03464_train.tsv?download=1",
        "MD5_Sure_SuRE45_HG03464_train.tsv": "324fa27a481d62a8138262166f9e87ea",
        "url_Sure_SuRE45_HG03464_val.tsv": "https://zenodo.org/records/15195607/files/Sure_SuRE45_HG03464_val.tsv?download=1",
        "MD5_Sure_SuRE45_HG03464_val.tsv": "eaec888e4e13c2c98ec4eb28e041bcfa",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Sharpr": {
        "python_class": "Sharpr",
        "url_Sharpr_test.tsv": "https://zenodo.org/records/15195607/files/Sharpr_test.tsv?download=1",
        "MD5_Sharpr_test.tsv": "c93d3bd2b5a9e8e8f71699af50a81614",
        "url_Sharpr_train.tsv": "https://zenodo.org/records/15195607/files/Sharpr_train.tsv?download=1",
        "MD5_Sharpr_train.tsv": "b862d888630bda8de8f968fa930e2203",
        "url_Sharpr_val.tsv": "Sharpr_val.tsv",
        "MD5_Sharpr_val.tsv": "c93d3bd2b5a9e8e8f71699af50a81614",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
    "Fluorescence": {
        "python_class": "Fluorescence",
        "url_Fluorescence_test.tsv": "https://zenodo.org/records/15195607/files/Fluorescence_test.tsv?download=1",
        "MD5_Fluorescence_test.tsv": "23dc89727366d031f7d55506585ed97f",
        "url_Fluorescence_train.tsv": "https://zenodo.org/records/15195607/files/Fluorescence_train.tsv?download=1",
        "MD5_Fluorescence_train.tsv": "703e7831688549aeab4b391988dade84",
        "url_Fluorescence_val.tsv": "https://zenodo.org/records/15195607/files/Fluorescence_val.tsv?download=1",
        "MD5_Fluorescence_val.tsv": "9d7f564bb9cc4f9100edf784fe6d1001",
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
        "url_Evfratov_23_test.tsv": "https://zenodo.org/records/15195607/files/Evfratov_23_test.tsv?download=1",
        "MD5_Evfratov_23_test.tsv": "a1bfaced916f2cd1fd312eca4aa36787",
        "url_Evfratov_23_train.tsv": "https://zenodo.org/records/15195607/files/Evfratov_23_train.tsv?download=1",
        "MD5_Evfratov_23_train.tsv": "3bccff4b46d3a859f1bcaa3c4204bebd",
        "url_Evfratov_23_val.tsv": "https://zenodo.org/records/15195607/files/Evfratov_23_val.tsv?download=1",
        "MD5_Evfratov_23_val.tsv": "174fe1f42aab6888e1a1e90d70b89bd6",
        "url_Evfratov_33_test.tsv": "https://zenodo.org/records/15195607/files/Evfratov_33_test.tsv?download=1",
        "MD5_Evfratov_33_test.tsv": "b3af0b99fbe1e0a1400935294cff9520",
        "url_Evfratov_33_train.tsv": "https://zenodo.org/records/15195607/files/Evfratov_33_train.tsv?download=1",
        "MD5_Evfratov_33_train.tsv": "622674c09d2e55c6b783e1fc8eedfab2",
        "url_Evfratov_33_val.tsv": "https://zenodo.org/records/15195607/files/Evfratov_33_val.tsv?download=1",
        "MD5_Evfratov_33_val.tsv": "cb23a2c03f0d690f5d0abe233d653de2",
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
        "url_DeepPromoter_all_seqs.tsv": "https://zenodo.org/records/15195607/files/DeepPromoter_all_seqs.tsv?download=1",
        "MD5_DeepPromoter_all_seqs.tsv": "242bac7cbe193700b680bb6e839a3f06",
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
        "url_DeepStarr_all_chr.tsv": "https://zenodo.org/records/15195607/files/DeepStarr_all_chr.tsv?download=1",
        "MD5_DeepStarr_all_chr.tsv": "9f453ab098d59c8122b87e5be90480cb",
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
        "url_Kircher_GRCh38_ALL.tsv": "https://zenodo.org/records/15195607/files/Kircher_GRCh38_ALL.tsv?download=1",
        "MD5_Kircher_GRCh38_ALL.tsv": "d979b8e749c7e2c468c97682fe4e2854",
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
        "url_Dream_paired.tsv": "https://zenodo.org/records/15195607/files/Dream_paired.tsv?download=1",
        "MD5_Dream_paired.tsv": "7512ae3f91711af4331210548cb3cb52",
        "url_Dream_single.tsv": "https://zenodo.org/records/15195607/files/Dream_single.tsv?download=1",
        "MD5_Dream_single.tsv": "1bcbc80702d74151c57e2da8a82b2c7a",
        "url_Dream_train.tsv": "https://zenodo.org/records/15195607/files/Dream_train.tsv?download=1",
        "MD5_Dream_train.tsv": "a7f71419d75b048a1f1b84c352b3ab78",
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
        "url_Vaishnav_complex_drift.tsv": "https://zenodo.org/records/15195607/files/Vaishnav_complex_drift.tsv?download=1",
        "MD5_Vaishnav_complex_drift.tsv": "26eb589f80e6f7922d4fdcc561ff5e52",
        "url_Vaishnav_complex_native.tsv": "https://zenodo.org/records/15195607/files/Vaishnav_complex_native.tsv?download=1",
        "MD5_Vaishnav_complex_native.tsv": "d18cec6b7fa58da1c5f761f6551804d8",
        "url_Vaishnav_complex_paired.tsv": "https://zenodo.org/records/15195607/files/Vaishnav_complex_paired.tsv?download=1",
        "MD5_Vaishnav_complex_paired.tsv": "52f1f353ecdcbee7ad7a6f930ec9150d",
        "url_Vaishnav_complex_train_val.tsv": "https://zenodo.org/records/15195607/files/Vaishnav_complex_train_val.tsv?download=1",
        "MD5_Vaishnav_complex_train_val.tsv": "c7b0964cd9132595961dbe28698e70c4",
        "url_Vaishnav_defined_drift.tsv": "https://zenodo.org/records/15195607/files/Vaishnav_defined_drift.tsv?download=1",
        "MD5_Vaishnav_defined_drift.tsv": "ff0cee1369303c0cc3469680027f432c",
        "url_Vaishnav_defined_native.tsv": "https://zenodo.org/records/15195607/files/Vaishnav_defined_native.tsv?download=1",
        "MD5_Vaishnav_defined_native.tsv": "95c6809a57a564d39e4a12be41682dd5",
        "url_Vaishnav_defined_paired.tsv": "https://zenodo.org/records/15195607/files/Vaishnav_defined_paired.tsv?download=1",
        "MD5_Vaishnav_defined_paired.tsv": "e156cf61cd413e31e73a60f41c8adc94",
        "url_Vaishnav_defined_train_val.tsv": "https://zenodo.org/records/15195607/files/Vaishnav_defined_train_val.tsv?download=1",
        "MD5_Vaishnav_defined_train_val.tsv": "1877098e135103dcc96e79c27bfa8122",
        "target_columns": {},
        "scalar_features": {},
        "vector_features": {},
        "n_samples": {},
        "folds": {
                 }
    },
}