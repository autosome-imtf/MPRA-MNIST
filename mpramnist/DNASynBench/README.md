# DNASynBench dataset

## Main Information

DNASynBench is a tool for creating benchmark datasets based on several regulatory genomics tasks and is designed to test models for their ability to capture biological patterns in data.

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/DNASynBench/mpramnist/DNASynBench/DNASynDataset_example.ipynb) for detailed usage example and training

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

### 1) Motif presency
Binary classification task to find the motif in the sequence.

**Settings:**
* length
* n_seqs
* motif
* ratio
* gc_content
* split_ratio
* random_state
<img width="456" height="66" alt="1_eng" src="https://github.com/user-attachments/assets/51533320-a829-4f8f-ae2e-2a90b88d937b"/>

### 2, 3) Linear and Nonlinear cooperativity
Regression tasks that assume that the activity of a sequence depends linearly or non-linearly on the number of motifs.

**Settings:**
|  motif   | length | n_seqs | min_num | max_num | gc_content | split_ratio | random_state |
| required | length | n_seqs | min_num | max_num | gc_content | split_ratio | random_state |
<img width="461" height="114" alt="2_eng" src="https://github.com/user-attachments/assets/d4e7f039-3467-4ae1-92e1-536bb50c3a10" />


## Data Representation

```
                   sequence	                      target	split
taatttttatctgtctgtgcgctatgcctatattggttaaagtatttagt	36.84	train
cgcgggaatcgcgcaggcagcactaccccagagcgtggtagcctgggagt	94.32	val
gaatcactttttcgttgccgccttctttgaatttatcaacgatattaccc	70.45	test

```
**Interpretation**: 
   - S = 1: Activity equal to reference promoter
   - S < 1: Weaker than reference
   - S > 1: Stronger than reference
   - Many AI-generated promoters showed S values >> 1, indicating significantly higher activity.

## Common parameters

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

### 2) Initialize transforms

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
