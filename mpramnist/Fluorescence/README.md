# Fluorescence dataset

## Main Information

The Fluorescence dataset (Reddy AJ et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/)) is based on a lentiviral MPRA system designed to measure promoter-driven expression in three immune cell lines: Jurkat (T cells), K562 (lymphoblasts), and THP-1 (monocytes). The dataset includes 17,104 DNA sequences of 250 bp length, split into training (70%), validation (10%), and test (20%) sets.

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/FluorescenceDataset_example.ipynb) for detailed usage example and training

## Tasks

### Regression

The regression task involves predicting 3 normalized regulatory activity values (one for each cell line), which reflect the sequence's ability to activate expression in the respective cellular context.

### Classification

The classification task is not yet implemented and will be added in a future release.

### Data Representation

```
sequence	JURKAT	K562	THP1	numerical_JURKAT	numerical_K562	numerical_THP1
GGGGGCGCT...	False	False	True	-0.2697505930801609	-1.1044306691338297	0.1111939002042786
CCATGCGGC...	True	True	True	0.4672150010860452	0.4805318315873264	0.5567267144276968
CGCGTTCCA...	False	True	False	-0.693749046357636	1.1414958790310077	-0.8999900063099485
```

**sequence**: The 250 bp DNA sequence.

**JURKAT, K562, THP1 (boolean)**: Binary labels indicating activity above a threshold in the respective cell line.

**numerical_JURKAT, numerical_K562, numerical_THP1 (float)**: Normalized continuous activity values for regression.

## Parameters

### **`split : str`**
    Defines which data split to use. Must be one of: 'train', 'val', 'test'.
    Determines which dataset file to load (e.g., 'Fluorescence_train.tsv').

### **`cell_type : str` | List[str]**, optional, default=`["JURKAT", "K562", "THP1"]`
    Cell type(s) to include in the dataset. Can be a single cell type string
    or list of multiple cell types. All three cell types are included by default.

### **`task : str`, optional**, default="regression"
    Specifies the machine learning task. Currently only "regression" is supported.
    Classification may be added in future versions.

### **`transform : callable`, optional**
    Transformation applied to each sequence object.

### **`target_transform : callable`, optional**
    Transformation applied to the target data.

### **`root : str`, optional, default=`None`**
    Root directory where dataset files are stored or should be downloaded.
    If None, uses the default dataset directory from parent class.


## Data Handling Considerations

1) **Cell Type Selection**: Use the `cell_type` parameter to select one or multiple cell lines. All three cell lines `["JURKAT", "K562", "THP1"]` used by default

2) **Available Tasks**: Currently, only the regression task is supported. Classification will be added in future versions..

3) **Multi-task Learning**: This dataset is designed for multi-task learning, where a single model predicts activities across multiple cell lines simultaneously.

4) **Example Usage**: See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/FluorescenceDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.Fluorescence.dataset import FluorescenceDataset
    import torch.utils.data as data
    import mpramnist.transforms as t
```

### 2) Initialize transforms

```python
    transform = t.Compose(
        [
            t.ReverseComplement(0.5),       # Reverse-complement augmentation (training only)
            t.Seq2Tensor(),                 # Convert the sequence string to a numerical tensor
        ]
    )
    val_trainsform = t.Compose(
        [
            t.ReverseComplement(0.0),       # for validation and test
            t.Seq2Tensor(),                 # Convert the sequence string to a numerical tensor
        ]
    )
```

### 3) Dataset Creation

```python
    # Basic usage with default settings (all cell types)
    train_dataset = FluorescenceDataset(split='train', transform = transform)

    
    # Specific cell type for regression
    dataset = FluorescenceDataset(
        split='val',
        cell_type='K562',    # Use only K562 cell line
        task='regression',
        transform = val_transform,
        root = "../data/"
    )

```

### 4) Dataloader Creation

```python 
    train_loader = data.DataLoader(
        dataset=train_dataset, 
        batch_size=1024, 
        shuffle=True, # Shuffle is recommended for training
        num_workers=16 
    )

    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1024,
        shuffle=False, # No need to shuffle for validation/testing
        num_workers=16 
    )
```

## Original Benchmark Quality

Pearson correlation, r

    r = 0,615 for K562

    r = 0,599 for JURKAT

    r = 0,555 for THP-1


## Achieved Quality Using LegNet Model in MPRA-MNIST

Pearson correlation, r

    r = 0,63 for K562

    r = 0,62 for JURKAT

    r = 0,52 for THP-1


## Citation

When using this dataset, please cite the original publication:

[Reddy AJ et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/)

Reddy AJ, Herschl MH, Geng X, Kolli S, Lu AX, Kumar A, Hsu PD, Levine S, Ioannidis NM. Strategies for effectively modelling promoter-driven gene expression using transfer learning. bioRxiv [Preprint]. 2024 May 19:2023.02.24.529941. doi: 10.1101/2023.02.24.529941. PMID: 36909524; PMCID: PMC10002662.

```bibtex
    
    @article{reddy2024strategies,
        title = {Strategies for effectively modelling promoter-driven gene expression using transfer learning},
        author = {Reddy, Aniketh Janardhan and Herschl, Michael H. and Geng, Xinyang and Kolli, Sathvik and Lu, Amy X. and Kumar, Aviral and Hsu, Patrick D. and Levine, Sergey and Ioannidis, Nilah M.},
        journal = {bioRxiv},
        year = {2023},
        month = feb,
        doi = {10.1101/2023.02.24.529941},
        url = {https://doi.org/10.1101/2023.02.24.529941}
    }   
```