# Agarwal dataset

## Main Information

The Agarwal dataset ([Agarwal et al. 2025](https://www.nature.com/articles/s41586-024-08430-9)) is based on a lentiviral MPRA system and includes:

- Potential enhancers (identified from open chromatin regions in corresponding cell types)

- Canonical promoters centered on transcription start sites (TSS)

- Shuffled enhancer sequences with preserved dinucleotide composition

- Control elements with known activity in hepatocyte (HepG2), lymphoblast (K562), and induced pluripotent stem cell (WTC11) lines

The dataset comprises 122,926 (HepG2), 196,664 (K562), and 46,185 (WTC11) sequences, each 200 nucleotides long. Sequences are divided into training, validation, and test sets using an 8:1:1 ratio, following the original study.

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/AgarwalDataset_example.ipynb) for detailed usage example and training

## Tasks

### Regression

The regression task involves predicting scalar values representing regulatory activity for the corresponding cell line. Activity is measured as the logarithm of the ratio of transcript sequence reads (RNA) to reporter sequence reads (DNA): *log₂(RNA/DNA).*

### Data Representation

```

seq_id	    chromosome  start	    end	        strand	    seq	                expression	    averaged_expression	        fold
seq10002_R	    10	    88965538	88965738	-	        AGCAATCCCTGGGAAAA	-1.306	            -1.306	                10
seq10004_F	    10	    89029900	89030100	+	        TAGCTCAACACAAATCC	 0.43	            -0.017	                10
seq10004_R	    10	    89029900	89030100	-	        CATTGTTTCCATAGGGA	-0.464	            -0.017	                10
seq10005_F	    10	    89032143	89032343	+	        GACCCTAAATCAGTATG	-1.231	            -1.6350000000000002	    7
seq10005_R	    10	    89032143	89032343	-	        AAAGGGACTTTCCGCAT	-2.039	            -1.6350000000000002	    7

```

## Parameters

### **split : Union[str, List[int], int]**

Defines which data split to use. Opions:
- String: 'train', 'val', 'test' (uses predefined fold sets)
- List[int]: List of specific fold numbers (1-10)
- int: Single fold number (1-10)

### **cell_type : str**

Cell type for filtering the data. Must be one of: 'HepG2', 'K562', 'WTC11'

### **genomic_regions : Optional[Union[str, List[Dict]]], optional**

Genomic regions to include or exclude. Options:
- str: Path to BED file containing genomic regions
- List[Dict]: List of dictionaries with 'chrom', 'start', 'end' keys
- None: No genomic region filtering
- Uses **0-based** indexing for genomic coordinates in **hg38**

### **exclude_regions : bool, default=False**

If True, exclude the specified genomic regions instead of including them

### **averaged_target : bool, default=False**

If True, use 'averaged_expression' (mean activity between forward and reverse-complement sequences) as target;
otherwise use individual 'expression' values

### **root : optional**

Root directory for data storage

### **transform : callable, optional**

Transformation function applied to each sequence

### **target_transform : callable, optional**

Transformation function applied to target values


## Data Handling Considerations

1) **Cell Type Selection**: Use the `cell_type` parameter to select specific cell lines. The data is not multi-label, as sequences measured in HepG2 were not measured in K562.

2) **Genomic Region Filtering**: Use `the genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes. Uses **0-based** indexing for genomic coordinates in **hg38**.

3) **Constant Flanks**: Original sequences in the study had constant 15-nucleotide flanks on each side. These flanks have been removed from the provided sequences, but for optimal or comparable results, we recommend adding them back before training, validation, and testing. Use `AgarwalDataset.CONSTANT_LEFT_FLANK` and `AgarwalDataset.CONSTANT_RIGHT_FLANK` as shown in the examples below.

4) **LegNet Shift Augmentation**: The LegNet model uses embedded flanks for shift augmentation. These flanks are stored in `AgarwalDataset.LEFT_FLANK` and `AgarwalDataset.RIGHT_FLANK` attributes. Use them as shown in the examples below. Shift augmentation involves shifting sequences by a certain number of nucleotides left or right.

5) **Target Selection**: You can use either individual sequence activity measurements (`averaged_target = False`, using the *expression* column) or averaged activities between forward and reverse-complement sequences (`averaged_target = True`, using the *averaged_expression* column).

6) **Example Usage**: See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/AgarwalDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.Agarwal.dataset import AgarwalDataset
    import torch.utils.data as data
    import mpramnist.transforms as t
```

### 2) Initialize trannsforms

```python

    # Constant flank required for each sequence
    constant_left_flank = AgarwalDataset.CONSTANT_LEFT_FLANK 
    constant_right_flank = AgarwalDatase.CONSTANT_RIGHT_FLANK  

    # These flanks are used in LegNet for shifting augmentation
    left_flank = AgarwalDataset.LEFT_FLANK  
    right_flank = AgarwalDataset.RIGHT_FLANK

    # Training transform with augmentations
    transform = t.Compose(
        [
            t.AddFlanks(constant_left_flank, constant_rigtht_flank), # Add constant flanks

            # Transforms for shift augmentation (use only for training)
            t.AddFlanks("", right_flank),   # these transforms are used to the shift augmentation.
            t.RightCrop(230, 260),          # Shift parameters are (0, len(right_flank))
            t.LeftCrop(230, 230),           # Do not use shift augmentation for validation and test

            t.ReverseComplement(0.5),       # Reverse-complement augmentation (training only)
            t.Seq2Tensor(),
        ]
```

### 3) Dataset Creation

```python
    # Load training data for HepG2 cell type
    dataset = AgarwalDataset(split='train', cell_type='HepG2')
    
    # Load data filtered by genomic regions from BED file
    dataset = AgarwalDataset(
        split='train',
        cell_type='K562',
        transform = transform,
        genomic_regions='path/to/regions.bed'
    )
    
    # Load data excluding specific genomic regions
    regions = [{'chrom': '1', 'start': 1000, 'end': 2000}]
    dataset = AgarwalDataset(
        split=[1, 2, 3],
        cell_type='WTC11',
        genomic_regions=regions,
        transform = transform,
        exclude_regions=True
    )

    val_dataset = AgarwalDataset(
        cell_type="HepG2",
        split=[9], # or 'val'
        transform = validation_transform, # validation transforms should not use shift and reverse-complement
        root="../data/",
    )
```

### 4) Dataloader Creation

```python 
    val_loader = data.DataLoader(
        dataset=val_dataset, batch_size=1024, shuffle=False, num_workers=16
    )
```

## Original Benchmark Quality

Pearson correlation, r

    r = 0,83 for HepG2

    r = 0,87 for K562

    r = 0,79 for WTC11

## Achieved Quality Using LegNet Model in MPRA-MNIST

Pearson correlation, r

    r = 0,804 for HepG2

    r = 0,829 for K562

    r = 0,727 for WTC11

## Citation

When using this dataset, please cite the original publication:

[Agarwal et al. 2025](https://www.nature.com/articles/s41586-024-08430-9) 

Agarwal, V., Inoue, F., Schubach, M. et al. Massively parallel characterization of transcriptional regulatory elements. Nature 639, 411–420 (2025). https://doi.org/10.1038/s41586-024-08430-9

```bibtex
    @article{agarwal2025massively,
        title={Massively parallel characterization of transcriptional regulatory elements},
        author={Agarwal, V. and Inoue, F. and Schubach, M. and others},
        journal={Nature},
        volume={639},
        pages={411--420},
        year={2025},
        doi={10.1038/s41586-024-08430-9}
    }
```

    

