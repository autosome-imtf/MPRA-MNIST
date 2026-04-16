# STARRseq dataset

## Main Information

The datasets are based on ultra-complex massively parallel reporter assays (MPRAs) performed using the self-transcribing active regulatory region sequencing (STARR-seq) method ([Sahu et al. 2022](https://www.nature.com/articles/s41588-021-01009-4)). This study systematically characterized the sequence determinants of human gene regulatory element activity using four distinct STARR-seq libraries, complemented by additional genomic assays for validation and benchmarking.


### Random Enhancer

This library consisted of **synthetic random 170-bp sequences**. To identify sequence features influencing human enhancer activity without prior assumptions and to determine enhancer tissue specificity, an extremely complex library of random enhancers was used in GP5d cells. We use the original split: 7,942,528 training and 1,985,634 validation sequences (after removing reverse-complement augmentations), and 3,309,387 test sequences. Since reverse-complement augmentations were used in the original study for training and validation sets, we recommend applying the `ReverseComplement(0.5)` transform.


### Genomic Enhancer

This library consisted of **~500-bp fragments of genomic DNA**. Active enhancers were defined as those showing significant enrichment in STARR-seq RNA compared to the plasmid input (peaks called using MACS2) in GP5d and HepG2 cells. **For machine learning consistency, these sequences were standardized to 170 bp.** We use the original chromosome-based split: 89,062 validation sequences from chromosomes 4, 6, and 8; 99,362 test sequences from chromosomes 2, 10, and 11; and 323,756 training sequences from the remaining chromosomes. Reverse-complement augmentation was applied to the training set in the original study, so we recommend using the `ReverseComplement(0.5)` transform.


### Capture Promoter
This library uses **synthetic random 150-bp sequences**. **The task is to predict if position 100 in a 120-bp sequence is a transcription start site (TSS) of a functional promoter**. Active promoters were defined as Eukaryotic Promoter Database (EPD) entries overlapping a CAGE peak in GP5d cells. The binary labels (1/0) correspond to this fixed-position classification. We use the original split: 79,732 training, 13,290 validation, and 13,290 test sequences.

### Genomic Promoter

This library consists of **genomic 120-bp sequences** extracted as **100 bp upstream and 20 bp downstream of known TSSs** from the EPD. **The task is binary classification of the entire sequence as an active promoter region.** Negative sequences (class 0) are random genomic 120-bp sequences not overlapping EPD promoters. Due to an issue with the published test data, we use 7,285 validation sequences as the test set and split the 40,996 training sequences 9:1 for training/validation.


### ATAC-seq

This is a **separate assay dataset** (not a STARR-seq library) from the same study, measuring chromatin accessibility via ATAC-seq in GP5d and HepG2 cells. It serves as an independent benchmark to test whether sequence features learned from STARR-seq data can predict open chromatin. **The task is binary classification: predict whether a sequence overlaps an ATAC-seq peak** (indicative of open chromatin). We use a chromosome-based split: 910,186 validation sequences from chromosomes 4, 6, and 8; 912,794 test sequences from chromosomes 2, 10, and 11; and 2,712,578 training sequences from the remaining chromosomes.


### Binary 

This library tests promoter-enhancer interactions using pairs of random 150-bp sequences. It includes: active promoter + matched active enhancer pairs; active promoter + shuffled active enhancer pairs; active promoter + inactive enhancer pairs; and inactive promoter + active enhancer pairs. Activity was measured via STARR-seq in GP5d and HepG2 cells. **The task is binary classification: predict whether a given promoter-enhancer pair is active.** We use the original split: 2,991,302 training, 748,828 validation, and 1,252,044 test sequences.

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/Malinois_Dataset_example_model_legnet.ipynb) for detailed usage example and training

## Tasks

### Binary Classification

*   **Random Enhancer, Genomic Enhancer:** Predict whether a 170-bp sequence functions as an active enhancer (binary classification: active/inactive).
*   **Capture Promoter, Genomic Promoter:** Predict whether position 100 in a 120-bp sequence is a transcription start site (TSS) of a functional promoter (binary classification: TSS present/absent at that specific position).
*   **Genomic Promoter:** Predict if a 120-bp sequence (aligned -100..+20 from a known TSS) is an active promoter region.
*   **ATAC-seq:** Predict whether a sequence resides in open chromatin (binary classification: overlaps ATAC-seq peak or not).
*   **Binary:** Predict whether a given promoter-enhancer pair is active (binary classification: active/inactive pair).


### Data Representation

Sequences are stored in `.fasta.gz` format, and the target label is specified in the comment line of each sequence.

```
>NS500683:269:H23L5AFXY:3:21402:23696:10856:1:N:0:GTCCGC 1
ccgaaaatgtttcttcacataaatagatgcttttctttttttttttttgtctcagggctactcggtactgtctcatcagcagcattggtattataagggtgtaggggtacccatgaatagatcaggcttcgggggcggcagcgcggatcaggccaggcgcggcgggtgaa
>NS500683:269:H23L5AFXY:1:11106:3770:14258:1:N:0:GTCCGC 1
ctggttttcgtatgagcaataggcttaacctatttagggttggttctcacagttagtagtgaggacagcttcaactggtcccgagttagattggtcacttctgttttgccgtggccgtatcttggtagaaagcgtatttgaaacggattatctcggcatcgtctattcct
>NS500683:269:H23L5AFXY:1:11301:23879:10582:1:N:0:GTCCGC 1
gcctaggcttgtttttcattcagcgttatcgttttagttgatactgatctacgccccgccctgaactaacttgatgaggtattactatttggaattcaagaggtaatataataatgtttttagttgtccgtgcatgttcgtgtcctggatcgatgaccgtagcgggtccg
>NS500683:269:H23L5AFXY:3:11610:23779:17896:1:N:0:GCCAAT 0
cttctctactcaatgctgttcgtacgtcttctggttttgtgcgaactctaatacgtcccgaggctttttaggtccgtatgtgactctgttatatgtggctccgttacaactaatacctctgttttgtctcagtctggaggctcgacgtacctgaccaatgtgcagacgta
>NS500683:269:H23L5AFXY:4:11605:24483:19334:1:N:0:CTTGTA 0
aatattttttaagagctttatactgccactgtatgtacccacacttagtctaaaggagatgatcttccatgctgcgcattattaaaaatcggttaatttccctttctaaagcgtttaaatttgcccggtatgtccagtgtgtatgagcagggatctcattgtagggaacc
```

## Parameters

### **`task : str`**

The name of the task to load. Must be one of:
- `"randomenhancer"`: Random enhancer data
- `"genomicpromoter"`: Genomic promoter data  
- `"capturepromoter"`: Capture promoter data
- `"genomicenhancer"`: Genomic enhancer data (chromosome-based splits)
- `"atacseq"`: ATAC-seq data (chromosome-based splits)
- `"binary"`: Binary classification task

### **`split : str | List[str] | List[int] | int`**

Specifies how to split the data:
- For default split tasks: `"train"`, `"val"`, or `"test"`
- For chromosome-based tasks (`"genomicenhancer"`, `"atacseq"`): chromosome names/numbers (e.g., `"chr2"`, `2`) or predefined splits.

### **`binary_class : Literal["enhancer_from_input", "promoter_from_input", "enhancer_permutated"]`**, optional

Specifies the subtype for the binary classification task. This parameter is primarily intended for creating specialized **training** sets.

`None` (default): Loads the standard dataset with active promoter/enhancer pairs (`"promoter"` / `"enhancer"`) as class `1` and various negative pairs as class `0`. Suitable for `"train"`, `"val"`, and `"test"` splits.

`"enhancer_from_input"`, etc.: Loads a dataset where pairs are constructed using the specified negative component (e.g., inactive enhancer) for more controlled training. Primarily used with `split="train"`.

### **`root : str`**, optional

Root directory where data is stored. If `None`, uses default data path.

### **`transform : callable`**, optional

Function to apply transformations to the input sequences.

### **`target_transform : callable`**, optional

Function to apply transformations to the target labels.

## Data Handling Considerations

1) **Task Selection**: The `task` parameter determines which of the six available STARR-seq datasets to load.

2) **Predefined Splits**: The `"randomenhancer"`, `"capturepromoter"`, `"genomicpromoter"`, and `"binary"` tasks use fixed training, validation, and test splits. Use `split="train"`, `"val"`, or `"test"`.

3) **Chromosome-based Splits**: The `"genomicenhancer"` and `"atacseq"` tasks use chromosome-based splitting to ensure independent test sets. You can use the predefined `split="train"`, `"val"`, or `"test"`, or specify specific chromosomes (e.g., `split=["chr2", "chr10"]`).

4) **Binary Task Variants**: For the `"binary"` task, the `binary_class` parameter allows loading specialized datasets for training:

- `"promoter_from_input"`: Pairs of inactive promoters with active enhancers.

- `"enhancer_permutated"`: Pairs of active promoters with shuffled (non-corresponding) active enhancers.

- `"enhancer_from_input"`: Pairs of active promoters with inactive enhancers.

- `None` (default): The standard binary dataset containing mixed negative pairs.

5) For the `"binary"` task, each data point returns a tuple: `(seq, seq_enh, label)`, where the label indicates whether the pair is active (`1`) or inactive (`0`).

6) **Example Usage**: See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/Malinois_Dataset_example_model_legnet.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    from mpramnist.StarrSeq.dataset import StarrSeqDataset
    from mpramnist.trainers import LitModel_StarrSeq
    from mpramnist import transforms as t
    import torch.utils.data as data
```

### 2) Initialize transforms

```python
    train_transform = t.Compose(
        [
            t.ReverseComplement(0.5),
            t.Seq2Tensor(),
        ]
    )

    val_test_transform = t.Compose([t.Seq2Tensor()])
```

### 3) Dataset Creation

```python

    # Random Enhancer task
    ranenh_dataset = StarrSeqDataset(
        task="randomenhancer", split="train", transform=train_transform, root="../data"
    )

    # Genomic Promoter task
    genp_dataset = StarrSeqDataset(
        task="genomicpromoter", split="train", transform=train_transform, root="../data/"
    )

    # Capture Promoter task
    capt_dataset = StarrSeqDataset(
        task="capturepromoter", split="train", transform=train_transform, root="../data/"
    )

    # Genomic Enhancer task
    genenh_dataset = StarrSeqDataset(
        task="genomicenhancer", split="train", transform=train_transform, root="../data/"
    )

    # ATAC-seq task
    atacseq_dataset = StarrSeqDataset(
        task="atacseq", split="train", transform=train_transform, root="../data/"
    )

    # Binary task - Standard dataset
    classic_binary_dataset = StarrSeqDataset(
        task="binary",
        binary_class=None,
        split="train",
        transform=train_transform,
        root="../data/",
    )

    # Binary task - Specialized training variant
    binary_enh_permut_dataset = StarrSeqDataset(
        task="binary",
        binary_class="enhancer_permutated",
        split="train",
        transform=train_transform,
        root="../data/",
    )

    # Validation split
    val_dataset = StarrSeqDataset(
        task="binary", split="val", transform=val_test_transform, root="../data/"
    )
  
```

### 4) Dataloader Creation

```python 
    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8
    )

    val_loader = data.DataLoader(
        dataset=val_dataset, batch_size=128, shuffle=False, num_workers=8
    )

```

## Original Benchmark Quality

**Metric Area Under Precision-Recall Curve (AUPRC)**

| Dataset | Model Performance (from Sahu et al. 2022) | 
|-----------|:---------------:|
| Random Enhancer | 0.65 | 
| Genomic Enhancer | 0.8 | 
| Capture Promoter | 0.95 | 
| Genomic Promoter | 0.98 | 
| ATACseq | 0.9 | 
| Binary | 0.873 | 

    
## Achieved Quality Using LegNet Model in MPRA-MNIST

**Metric Area Under Precision-Recall Curve (AUPRC)**

| Dataset | LegNet Performance | 
|-----------|:---------------:|
| Random Enhancer | 0.625 | 
| Genomic Enhancer | 0.77 | 
| Capture Promoter | 0.926 | 
| Genomic Promoter | 0.965 | 
| ATACseq | 0.907 | 
| Binary | 0.87 | 
 
## Citation

When using this dataset, please cite the original publication:

[Sahu et al. 2022](https://www.nature.com/articles/s41588-021-01009-4)

Sahu, B., Hartonen, T., Pihlajamaa, P. et al. Sequence determinants of human gene regulatory elements. Nat Genet 54, 283–294 (2022). https://doi.org/10.1038/s41588-021-01009-4

```bibtex
    @article{Sahu2022sequence,
        title = {Sequence determinants of human gene regulatory elements},
        volume = {54},
        ISSN = {1546-1718},
        url = {http://dx.doi.org/10.1038/s41588-021-01009-4},
        DOI = {10.1038/s41588-021-01009-4},
        number = {3},
        journal = {Nature Genetics},
        publisher = {Springer Science and Business Media LLC},
        author = {Sahu, Biswajyoti and Hartonen, Tuomo and Pihlajamaa, Päivi and Wei, Bei and Dave, Kashyap and Zhu, Fangjie and Kaasinen, Eevi and Lidschreiber, Katja and Lidschreiber, Michael and Daub, Carsten O. and Cramer, Patrick and Kivioja, Teemu and Taipale, Jussi},
        year = {2022},
        month = feb,
        pages = {283–294}
    }
```