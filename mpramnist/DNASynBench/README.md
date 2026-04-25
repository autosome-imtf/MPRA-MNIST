# DNASynBench dataset

## Main Information

DNASynBench представляет собой инструмент для создания бенчмарковых датасетов на основе нескольких задач регуляторной геномики и предназначен для тестирования моделей на способность улавливать биологические закономерности в данных.

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/DNASynBench/DNASynBench/DNASynDataset_example.ipynb) for detailed usage example and training

## Tasks

| Number  | Name                     | Motifs per task | Type of task              |
|---------|--------------------------|-----------------|---------------------------|
| task_1  | Motif presency           | 1               | Classification            |
| task_2  | Linear cooperativity     | variable        | Regression                |
| task_3  | Nonlinear cooperativity  | variable        | Regression                |
| task_4  | Alien motif              | 2               | Classification            |
| task_5  | Motif combination        | 2               | Classification            |
| task_6  | Range dependent activity | 2 the same      | Classification            |

By default, dataset are split into: 0.75 sequences for training, 0.125 for validation, and 0.125 for testing. This ratio is a parameter and could be tuned if necessary.

### Regression

The regression task is to predict a single scalar value representing normalized promoter activity. The activity values in the dataset are calculated using the following experimental measurements from the original study:

1) **Experimental measurement**: Each promoter variant ("clone") was inserted upstream of the sfGFP reporter gene, and its expression was measured as fluorescence per cell (`F/OD600`).

2) **Normalization**: Promoter strength (S) was calculated relative to positive and negative controls using the formula:

S = ( (F/OD600)_clone - (F/OD600)_blank ) / ( (F/OD600)_BBA_J23_119 - (F/OD600)_blank )

Or in methematical notation:

$$S = \frac{(F/OD600)_{\text{clone}} - (F/OD600)_{\text{blank}}}{(F/OD600)_{\text{BBA\_J23\_119}} - (F/OD600)_{\text{blank}}}$$

Where:

 - **(F/OD600)**= Fluorescence normalized by cell density

 - **blank** = Control with a 10-nt random sequence (GGGCTCTGTA)

 - **BBA_J23_119** = Reference wild-type promoter (positive control)

3) **Interpretation**: 
   - S = 1: Activity equal to reference promoter
   - S < 1: Weaker than reference
   - S > 1: Stronger than reference
   - Many AI-generated promoters showed S values >> 1, indicating significantly higher activity.


4) **Final values**: Reported activities are averages of three independent biological experiments.

The target values in this dataset represent these normalized promoter activity scores, which quantify transcriptional strength relative to experimental controls.

### Data Representation

```
                sequence	                        target	split
taatttttatctgtctgtgcgctatgcctatattggttaaagtatttagt	36.84	train
cgcgggaatcgcgcaggcagcactaccccagagcgtggtagcctgggagt	94.32	val
gaatcactttttcgttgccgccttctttgaatttatcaacgatattaccc	70.45	test

```

## Parameters

### **`split : str`**
Defines which data split to use. Must be one of: `'train'`, `'val'`, `'test'`.
The dataset filters sequences based on the 'split' column in the data file.
### **`transform : callable`**, optional
Transformation applied to each sequence object.
### **`target_transform : callable`**, optional
Transformation applied to the target data.
### **`root : str`**, optional, `default=None`
Root directory where dataset files are stored or should be downloaded.
If `None`, uses the default dataset directory from parent class.

## Data Handling Considerations

1) **Example Usage**: See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/DeepPromoterDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.trainers import LitModel_DeepPromoter
    import torch.utils.data as data
    import mpramnist.transforms as t
```

### 2) Initialize trannsforms

```python
    # Define set of transforms
    train_transform = t.Compose(
        [
            t.ReverseComplement(0.5),  # Apply reverse complement with 50% probability
            t.Seq2Tensor(),
        ]
    )
    val_test_transform = t.Compose(
        [
            t.Seq2Tensor(),  # No reverse complement for validation/test
        ]
    )
```

### 3) Dataset Creation

```python
    # Basic usage for training
    dataset = DeepPromoterDataset(
        split='train', 
        transform = train_transform,
        root="../data/"
    )

    val_dataset = DeepPromoterDataset(
        split="val", 
        transform=val_test_transform, 
        root="../data/"
    )
    
```

### 4) Dataloader Creation

```python 
    train_loader = data.DataLoader(
        dataset=dataset, 
        batch_size=1024, 
        shuffle=True,   # Shuffle for the training set
        num_workers=8
    )

    val_loader = data.DataLoader(
        dataset=val_dataset, 
        batch_size=1024, 
        shuffle=False,   # No need to shuffle for validation/testing
        num_workers=8
    )
```
