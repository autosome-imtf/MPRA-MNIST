# AgarwalJoint dataset

## Main Information

The AgarwalJoint dataset ([Agarwal et al. 2025](https://www.nature.com/articles/s41586-024-08430-9)) is based on the original Agarwal dataset and contains 55,338 sequences: approximately 19,000 potential enhancers from each cell line (HepG2, K562, WTC11).

The dataset consists of 200-nucleotide sequences divided into training, validation, and test sets using an 8:1:1 ratio. The primary task is regression, predicting 3 scalar activity values representing expression activation levels in each cell line.

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/AgarwalJointDataset_example.ipynb) for detailed usage example and training

## Tasks

### Regression

The regression task involves predicting 3 scalar activity values representing expression activation levels in each cell line (HepG2, K562, WTC11). Activity is measured as the logarithm of the ratio of transcript sequence reads (RNA) to reporter sequence reads (DNA): *log₂(RNA/DNA).*

### Data Representation

```

seq_id	            	chromosome	start	    end	        strand	    seq	                    HepG2	                K562	            WTC11	    fold
ENSG00000000971	    	1	        196651781	196651981	+	  GATATCACCAGCTGCTGATTTG...	 0.0705767810952613	-0.386981332585921	 0.26108557591418	2
ENSG00000001630	    	7	        92134284	92134484	-	  TGGGTTTAGTAGGAGACCTGGG...	-0.250686030152044	-0.167357423155188	 0.382690550999934	6
ENSG00000002726	    	7	        150824694	150824894	+	  CAAGGTGGCTGGGGAGAAGGCC...	 0.547128759230438	-0.49542235153705	-0.103962309065462	10
ENSG00000003056	    	12	        8949506     8949706	    -	  GGGGTCTGGTGGGAGGAGCGGT...	-0.638726825596637	-0.133358216202116	 0.726073181797631	1
ENSG00000003096	    	X	        118116627	118116827	-	  CCCTCAGCAGCCCCCCCACACC...	 0.881140094733589	-0.766456352126663	-0.194117721629037	5

```

## Parameters

### **split : Union[str, List[int], int]**

Defines which data split to use. Opions:
- String: 'train', 'val', 'test' (uses predefined fold sets)
- List[int]: List of specific fold numbers (1-10)
- int: Single fold number (1-10)

### **cell_type : str**

Cell type(s) for filtering the data. Can be:
- str: Single cell type ('HepG2', 'K562', or 'WTC11')
- List[str]: Multiple cell types

### **genomic_regions : Optional[Union[str, List[Dict]]], optional**

Genomic regions to include or exclude. Options:
- str: Path to BED file containing genomic regions (hg38, 0-based)
- List[Dict]: List of dictionaries with 'chrom', 'start', 'end' keys (hg38, 0-based)
- None: No genomic region filtering
- Uses **0-based** indexing for genomic coordinates in **hg38**

### **exclude_regions : bool, default=False**

If True, exclude the specified genomic regions instead of including them

### **root : optional**

Root directory for data storage

### **transform : callable, optional**

Transformation function applied to each sequence

### **target_transform : callable, optional**

Transformation function applied to target values


## Data Handling Considerations

1) **Cell Type Selection**: Use the `cell_type` parameter to select one or multiple cell lines. 

2) **Genomic Region Filtering**: Use `the genomic_regions` and `exclude_regions` parameters to select or exclude specific genomic regions across chromosomes. Uses **0-based** indexing for genomic coordinates in **hg38**.

3) **Constant Flanks**: Original sequences in the study had constant 15-nucleotide flanks on each side. These flanks have been removed from the provided sequences, but for optimal or comparable results, we recommend adding them back before training, validation, and testing. Use `AgarwalJointDataset.CONSTANT_LEFT_FLANK` and `AgarwalJointDataset.CONSTANT_RIGHT_FLANK` as shown in the examples below.

4) **LegNet Shift Augmentation**: The LegNet model uses embedded flanks for shift augmentation. These flanks are stored in `AgarwalJointDataset.LEFT_FLANK` and `AgarwalJointDataset.RIGHT_FLANK` attributes. Use them as shown in the examples below. Shift augmentation involves shifting sequences by a certain number of nucleotides left or right.

5) **Multi-task Learning**: This dataset is designed for multi-task learning, where a single model predicts activities across multiple cell lines simultaneously.

6) **Example Usage**: See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/AgarwalJointDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.AgarwalJoint.dataset import AgarwalJointDataset
    import torch.utils.data as data
    import mpramnist.transforms as t
```

### 2) Initialize trannsforms

```python

    # Constant flank required for each sequence
    constant_left_flank = AgarwalJointDataset.CONSTANT_LEFT_FLANK 
    constant_right_flank = AgarwalJointDatase.CONSTANT_RIGHT_FLANK  

    # These flanks are used in LegNet for shifting augmentation
    left_flank = AgarwalJointDataset.LEFT_FLANK  
    right_flank = AgarwalJointDataset.RIGHT_FLANK

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
    # Load training data for HepG2 cell type only
    dataset = AgarwalJointDataset(split='train', cell_type='HepG2')

    # Load data for multiple cell types
    dataset = AgarwalJointDataset(
        split='train',
        cell_type=['HepG2', 'K562']
    )

    # Load data filtered by genomic regions from BED file
    dataset = AgarwalJointDataset(
        split='train',
        cell_type=['HepG2', 'K562', 'WTC11'],
        genomic_regions='path/to/regions.bed'
    )

    # Load data excluding specific genomic regions
    regions = [{'chrom': 'chr1', 'start': 1000, 'end': 2000}]
    dataset = AgarwalJointDataset(
        split=[1, 2, 3],
        cell_type='WTC11',
        genomic_regions=regions,
        exclude_regions=True
    )

    val_dataset = AgarwalJointDataset(
        cell_type=['HepG2', 'K562', 'WTC11'],
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

    r = 0,78 for HepG2

    r = 0,75 for K562

    r = 0,77 for WTC11

## Achieved Quality Using LegNet Model in MPRA-MNIST

Pearson correlation, r

    r = 0,79 for HepG2

    r = 0,76 for K562

    r = 0,77 for WTC11

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
    

