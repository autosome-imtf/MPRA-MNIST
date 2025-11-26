# Sure dataset

## Main information

The Sure dataset (Survey of Regulatory Elements) is based on the analysis of genomes from 4 individuals from 4 different populations ([van Arensbergen et al. 2017](https://pubmed.ncbi.nlm.nih.gov/28024146/)) and it was scaled up by [van Arensbergen et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6609452/). The genomes of these individuals are broken into 150–500bp fragments and each fragment is cloned into a reporter plasmid. These sequence fragments can drive expression and function as promoters in transfected cells if the fragment contains a valid TSS. ~ 2.4B and ~ 1.2B fragments were found to be expressed in K-562 and HepG2 cells, respectively.

Preprocessed data and code were integrated from the work of [Reddy et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/) ([GitHub](https://github.com/anikethjr/promoter_models/blob/main/promoter_modelling/dataloaders/SuRE.py)). Separate datasets are created for each individual, supporting both classification and regression task formats. The final datasets contain approximately 400-600K training sequences and 50-70K test and validation sequences per individual.

## Tasks

### Classification Task

The classification task involves predicting two independent class labels (ranging from 0 to 4) for each sequence, representing expression levels in K562 and HepG2 cell lines respectively.

```
    chr     start       end     strand      split       seq                      K562_bin	HepG2_bin
    ---------------------------------------------------------------------------------------------------------
    X	    130598969	130599266	-       train    ATAAGCTTTTTGA...            1           4
    6	    41888832	41889084	-	    train    GTTAGCTTCTCTCA...           4	         2
    16	    47945205	47945676	-	    train    CTTATGAAGCTTAG...           0 	         1
```

### Regression Task

The regression task involves predicting the average expression level (calculated from normalized read counts) for each of the two cell lines - K562 and HepG2.

```
    chr     start       end     strand      split       seq                  avg_K562_exp  avg_HepG2_exp
    ---------------------------------------------------------------------------------------------------------
    3       80783739    80783991    -       train    TGGTTGCCCATTTT...           22.333      0.5
    9       73103533    73103851    +       train    GTAAAGTACTCAGT...           28.0	     3.5
    19      19483672	19484047    +       train    ACAAAAGACTCTGA...           22.666 	 6.5
```

See [Sure Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/SureDataset_example.ipynb]) for detailed information.

## Parameters

### **split : str**

Defines which split to use (e.g., 'train', 'val', 'test', or list of fold indices).

### **genome_id : str**

Identifier of the genome to use. Must be one of:
- "SuRE42_HG02601"
- "SuRE43_GM18983" 
- "SuRE44_HG01241"
- "SuRE45_HG03464"

Specifies which genomic dataset to load.

###  **task : str**

Type of machine learning task. Must be one of:
- "classification": for multi-class classification tasks
- "regression": for continuous value prediction tasks

Determines how target values are processed and interpreted.

### **permute : bool, optional, default=True**

Whether to transpose one-hot encoded sequence matrices from 
(4, sequence_length) to (sequence_length, 4) format.
This is done for compatibility with padding functions.

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

1) **Variable Sequence Lengths**: The main characteristic of this data is that sequence lengths vary. To handle this, we use an approach where sequences in each batch are padded with "N" nucleotides to match the length of the longest sequence in the batch. The `pad_collate` function is used for implementation. However, to enable this function to work properly, the shape of sequence tensors needs to be changed, which is achieved by setting the `permute=True` parameter.

2) **Permute Parameter**: When `permute=True`, the function transforms tensors from shape (4, sequence_length) to (sequence_length, 4).

3) **Genomic Coordinates**: Use the `genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes in the dataset. *Uses 0-based indexing for genomic coordinates.*

4) **Example Usage**: See [Sure Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/SureDataset_example.ipynb]) for detailed usage example and training

## Examples

### 1)  Import Important Packages and Create Padding Function

```python
    import torch
    import mpramnist
    from torch.nn.utils.rnn import pad_sequence
    from mpramnist.Sure.dataset import SureDataset
    import torch.utils.data as data

    def pad_collate(batch):  # required because sequence lengths vary
        (seq, targets) = zip(*batch)
        seq = pad_sequence(seq, 
                        batch_first=True, 
                        padding_value=0.25  # padding with "N" nucleotides
                        )
        return seq, torch.vstack(targets)
```

### 2) Dataset Creation

```python
    # Load training data for classification from one genome
    train_dataset = SureDataset(
        split="train",
        genome_id="SuRE42_HG02601", 
        task="classification"
    )

    # Load regression data with genomic region filtering
    dataset = SureDataset(
        split="test",
        genome_id="SuRE43_GM18983",
        task="regression",
        genomic_regions="promoters.bed"
    )

    # Load data excluding specific genomic regions
    regions = [{"chrom": "chr1", "start": 1000000, "end": 2000000}]
    dataset = SureDataset(
        split="val",
        genome_id="SuRE44_HG01241",
        task="classification", 
        genomic_regions=regions,
        exclude_regions=True
    )
```
### 3) Dataloader Creation

```python
    # Create DataLoader for training
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,  # shuffle for training
        num_workers=16,
        pin_memory=True,
        collate_fn=pad_collate,  # pad sequences to max length in each batch
    )

    # Create DataLoader for validation
    val_loader = data.DataLoader(
        dataset=dataset,
        batch_size=1024,
        shuffle=False,  # no shuffle for validation or test
        num_workers=16,
        pin_memory=True,
        collate_fn=pad_collate,
    )
```
## Original Benchmark Quality

No other study has used this data for pretraining, so we don't have information about the quality metrics achieved by the original authors.

## Achieved Quality Using LegNet Model

| Genome ID | K562 Regression | HepG2 Regression | K562 Classification | HepG2 Classification |
|-----------|:---------------:|:----------------:|:-------------------:|:--------------------:|
| SuRE42_HG02601 | <span style="color:green">0.511</span> | <span style="color:orange">0.357</span> | --- | --- |
| SuRE43_GM18983 | <span style="color:green">0.497</span> | <span style="color:orange">0.343</span> | --- | --- |
| SuRE44_HG01241 | <span style="color:blue">0.573</span> | <span style="color:red">0.317</span> | --- | --- |
| SuRE45_HG03464 | <span style="color:purple">**0.624**</span> | <span style="color:red">0.320</span> | --- | --- |

## Citation

When using this dataset, please cite the original publication:

[van Arensbergen J et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5498152/) 

van Arensbergen J, FitzPatrick VD, de Haas M, Pagie L, Sluimer J, Bussemaker HJ, van Steensel B. Genome-wide mapping of autonomous promoter activity in human cells. Nat Biotechnol. 2017 Feb;35(2):145-153. doi: 10.1038/nbt.3754. Epub 2016 Dec 26. PMID: 28024146; PMCID: PMC5498152.

```bibtex
    @article{arensbergen2017Genome-wide,
        title={Genome-wide mapping of autonomous promoter activity in human cells},
        author={van Arensbergen J, FitzPatrick VD, de Haas M, Pagie L, Sluimer J, Bussemaker HJ, van Steensel B},
        journal={Nat. Biotechnol.},
        volume={35(2)},
        pages={145--153},
        year={2017},
        doi={10.1038/nbt.3754}
    }
```