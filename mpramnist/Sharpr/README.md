# Sharpr dataset

## Main information

The Sharpr (Systematic High-resolution Activation and Repression Profiling using Reporters) dataset is based on the Sharpr-MPRA dataset, which contains expression activity measurements for approximately 487 thousand synthetic promoter sequences of 145 bp length, tested in K562 and HepG2 cell lines (Ernst et al. 2016). For testing and validation, we use approximately 10 thousand sequences from chromosome 18 and 19 thousand sequences from chromosome 8, respectively. We recommend using all other sequences for training, as was done in the original study (Reddy et al. 2024). The output data (labels) consist of 12 scalar values: 8 measurements (2 cell lines × 2 underlying promoters × 2 replicates) and 4 values representing replicate-averaged measurements.

Our implementation and preprocessed data are adapted from:
    https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/Sharpr_MPRA.py

## Tasks

### Regression Task

The raw counts from the experiments were processed by:
 1. Computing log2(RNA+1 / DNA+1) for each 145bp sequence in each of the 12 tasks (described below) in the two cell lines - K562 and HepG2
 2. Applying column-wise z-score normalization to the log fold-changes (i.e., each task's output values had mean 0 and variance 1)

The regression task involves predicting the expression level (calculated from normalized log fold-changes) for each replic of the two cell lines - K562 and HepG2.

```    
    name	chromosome	start	end	strand	seq	k562_minp_rep1	k562_minp_rep2	k562_minp_avg	... split
    -----------------------------------------------------------------------------------------------------
    H1hesc_1_0_0_chr20_30310735	20	30310588	30310733	+	GGGAGCCCA...	-1.8394496	-1.3714164	-1.860508   ...	train
    H1hesc_1_6_0_chr8_128830575	8	128830428	128830573	+	CCCATTTTA...	-0.8249799	-1.5034798	-1.3725991  ...	val
    H1hesc_1_94_0_chr18_11851975	18	11851828	11851973	+	AATCTGGAG...	-1.1480043	-1.559959	-1.5876448  ...	test

```

See [Sharpr Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/SharprDataset_example.ipynb) for detailed information.

## Parameters

### **split : str**

Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).

### **cell_type : List[str]**

List of column names with activity data to be used as targets.
Must be a subset of 
    
    CELL_TYPES = [
        "k562_minp_rep1", "k562_minp_rep2", "k562_minp_avg",
        "k562_sv40p_rep1", "k562_sv40p_rep2", "k562_sv40p_avg",
        "hepg2_minp_rep1", "hepg2_minp_rep2", "hepg2_minp_avg",
        "hepg2_sv40p_rep1", "hepg2_sv40p_rep2", "hepg2_sv40p_avg",
    ]

### **genomic_regions : str | List[Dict], optional**

Genomic regions to include/exclude. Can be:
- Uses hg19 reference genome
- Path to BED file
- List of dictionaries with 'chrom', 'start', 'end' keys
- Uses 0-based indexing for genomic coordinates

### **exclude_regions : bool**

If True, exclude the specified regions instead of including them

### **transform : callable, optional**

Transformation applied to each sequence object.

### **target_transform : callable, optional**

Transformation applied to the target data.

### **root : str, optional**

Root directory where data is stored. If None, uses default data path.

## Data Handling Considerations

1) **Cell Types**: The `cell_type` parameter determines which columns to use for prediction. This allows the task to be formulated as either multi-label regression or single-label regression.

2) **Genomic Coordinates**: Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

3) **Example Usage**: See [Sharpr Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/SharprDataset_example.ipynb) for detailed usage example and training

## Examples

### 1)  Import Important Packages

```python
    import torch
    import mpramnist
    from mpramnist.Sharpr.dataset import SharprDataset
    import torch.utils.data as data

    print(SharprDataset.CELL_TYPES)

>>> ['k562_minp_rep1',
 ... 'k562_minp_rep2',
 ... 'k562_minp_avg',
 ... 'k562_sv40p_rep1',
 ... 'k562_sv40p_rep2',
 ... 'k562_sv40p_avg',
 ... 'hepg2_minp_rep1',
 ... 'hepg2_minp_rep2',
 ... 'hepg2_minp_avg',
 ... 'hepg2_sv40p_rep1',
 ... 'hepg2_sv40p_rep2',
 ... 'hepg2_sv40p_avg']
```

### 2) Dataset Creation

```python
    train_dataset = SharprDataset(
        split="train",
        cell_type=SharprDataset.CELL_TYPES,
    )   

    # Load regression data with genomic region filtering
    test_dataset = SharprDataset(
        split="test",
        cell_type = SharprDataset.CELL_TYPES,
        genomic_regions="promoters.bed"
    )

    # Load data excluding specific genomic regions
    regions = [{"chrom": "chr1", "start": 1000000, "end": 2000000}]
    dataset = SharprDataset(
        split="val",
        cell_type = SharprDataset.CELL_TYPES,
        genomic_regions=regions,
        exclude_regions=True
    )
```

### 3) Dataloader Creation

```python
    # Create DataLoader for training
    loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,  # True for training, False for val and test
        num_workers=16,
        pin_memory=True,
    )
```

## Original Benchmark Quality

No other study has used this data for pretraining, so we don't have information about the quality metrics achieved by the original authors.

## Achieved Quality Using LegNet Model

| K562 average Regression | HepG2 average Regression |
|:---------------:|:----------------:|
| 0.408 | 0.354 |

## Citation

When using this dataset, please cite the original publication:

[Ernst J et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/) 

Ernst J, Melnikov A, Zhang X, Wang L, Rogov P, Mikkelsen TS, Kellis M. Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions. Nat Biotechnol. 2016 Nov;34(11):1180-1190. doi: 10.1038/nbt.3678. Epub 2016 Oct 3. PMID: 27701403; PMCID: PMC5125825.

```bibtex
    @article{ernst2016Genome-scale,
        title={Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions},
        author={Ernst J, Melnikov A, Zhang X, Wang L, Rogov P, Mikkelsen TS, Kellis M.},
        journal={Nat. Biotechnol.},
        volume={34(11)},
        pages={1180-1190},
        year={2016},
        doi={10.1038/nbt.3678}
    }
```