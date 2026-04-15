# Evfratov dataset

## Main Information

The Evfratov dataset is based on a large-scale MPRA (Massively Parallel Reporter Assay) study that systematically investigated the impact of randomized 5′-UTR sequences on translation initiation efficiency in E. coli ([Evfratov et al., 2017](https://academic.oup.com/nar/article/45/6/3487/2605795?login=false)). The goal was to discover novel regulatory elements beyond the known features like the Shine-Dalgarno sequence.

All sequences include the start codon AUG followed by either 20 or 30 randomized nucleotides (resulting in total lengths of 23 or 33 nt). The processed dataset contains 11,692 (for 23 nt) and 11,889 (for 33 nt) unique sequences with their assigned efficiency class. The data is pre-split into training, validation, and test sets in an 8:1:1 ratio.

See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/EvfratovDataset_example.ipynb) for detailed usage example and training

## Tasks

### Classification

The primary task for this dataset is multi-class classification. Each sequence (after processing) is labeled with one of 8 classes, which correspond to discretized bins (ranges) of translation efficiency. This makes the dataset valuable for studying regulatory elements within the 5′-UTR.


### Data Representation

```
sequence	                F1	F2	F3	F4	F5	F6	F7	F8
UUCAUCACAUGGCCGUAAUUAUG	    0	0	14	0	0	0	0	0
UUAACGAAUUAUGAACAGGCAUG	    0	0	11	0	0	0	0	0
ACUGUAAGAAGGUGAGAUGUAUG	    0	0	0	0	0	100	3	0
```

## Parameters

### **`split : str`**

Defines which split to use (e.g., `'train'`, `'val'`, `'test'`, or list of fold indices).

### **`length_of_seq: str | int`**

Length of sequences to use. Must be either `"23"` or `"33"` (or corresponding integers).
Determines which dataset variant to load (23bp or 33bp sequences).

### **`merge_last_classes : bool`**

A flag indicating whether to merge the last two classes in the dataset.
If `True`, the last two classes (typically those with the fewest examples)
will be merged into one. This can be useful for addressing class imbalance
or simplifying the classification task. If `False`, the classes remain
unchanged. By default, it is recommended to set this to `False` unless the
user is certain that merging is necessary.

### **`transform : callable, optional`**

Transformation applied to each sequence object.

### **`target_transform : callable, optional`**

Transformation applied to the target data.

### **`root : str,`** optional, `default=None`

Root directory where dataset files are stored or should be downloaded.
If `None`, uses the default dataset directory from parent class.

## Data Handling Considerations

1) **Sequence Length**: The `length_of_seq` parameter allows you to choose which dataset to use, containing sequences of either `23` nt (20 nt after AUG) or `33` nt (30 nt after AUG).

2) **Merging classes**: Setting `merge_last_classes=True` can help mitigate class imbalance by merging the two least populated classes.

3) **Example Usage**: See [Usage Example](https://github.com/autosome-imtf/MPRA-MNIST/blob/main/examples/EvfratovDataset_example.ipynb) for detailed usage example and training

## Examples

### 1) Import Important Packages

```python
    import mpramnist
    from mpramnist.Evfratov.dataset import EvfratovDataset
    import torch.utils.data as data
    import mpramnist.transforms as t
```

### 2) Initialize transforms

```python
    train_transform = t.Compose(
        [
            t.ReverseComplement(0.5),
            t.Seq2Tensor(),
        ]
    )

    test_transform = t.Compose(
        [  # test transform has probability of reverse-complement = 0.
            t.Seq2Tensor(),
            t.ReverseComplement(0),  # Reverse complementary transformation for all sequences with a probability of 0
        ]
    )
   
```

### 3) Dataset Creation

```python
    # Basic usage with 23-length sequences
    dataset = EvfratovDataset(split='train', length_of_seq=23)

    # With merged classes for balanced classification
    val_dataset = EvfratovDataset(
        split='val',
        length_of_seq=33,
        merge_last_classes=True
    )
    # Plot histogram with 7 classes if merge_last_classes=True
    val_dataset.hist_plot()
```

### 4) Dataloader Creation

```python 
    train_loader = data.DataLoader(
        dataset=train_dataset, 
        batch_size=1024, 
        shuffle=True, # Shuffle for the training set
        num_workers=8
    )
```

## Original Benchmark Quality

**F1 Score**

- **23 nt** sequences (20 nt after AUG): F1 = **0.22**

- **33 nt** sequences (30 nt after AUG): F1 = **0.22**


## Achieved Quality Using LegNet Model in MPRA-MNIST

**F1 Score**

- **23 nt** sequences (20 nt after AUG): F1 = **0.50**

- **33 nt** sequences (30 nt after AUG): F1 = **0.41**


## Citation

When using this dataset, please cite the original publication:

[Evfratov et al. 2016](https://academic.oup.com/nar/article/45/6/3487/2605795?login=false)

```bibtex
    @article{Evfratov2017application,
        title = {Application of sorting and next generation sequencing to study 5΄-UTR influence on translation efficiency in Escherichia coli},
        volume = {45},
        ISSN = {1362-4962},
        url = {https://doi.org/10.1093/nar/gkw1141},
        DOI = {10.1093/nar/gkw1141},
        number = {6},
        journal = {Nucleic Acids Research},
        publisher = {Oxford University Press (OUP)},
        author = {Evfratov, Sergey A. and Osterman, Ilya A. and Komarova, Ekaterina S. and Pogorelskaya, Alexandra M. and Rubtsova, Maria P. and Zatsepin, Timofei S. and Semashko, Tatiana A. and Kostryukova, Elena S. and Mironov, Andrey A. and Burnaev, Evgeny and Krymova, Ekaterina and Gelfand, Mikhail S. and Govorun, Vadim M. and Bogdanov, Alexey A. and Sergiev, Petr V. and Dontsova, Olga A.},
        year = {2017},
        month = jan,
        pages = {3487–3502}
    }
```
