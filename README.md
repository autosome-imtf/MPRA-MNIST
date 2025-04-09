## To launch .ipynb в examples 

    clone https://github.com/autosome-imtf/MpraDataset.git
    cd MpraDataset
    pip install setuptools wheel
    python setup.py sdist bdist_wheel
    pip install -e .

## TODO

* Tuning LegNet on Malinois data

## We have such datasets:

| Name | Artcile and link | DOI | Cell types |
| ----------- | ----------- | ----------- | -----------|
| HUMAN | ----------- | ----------- | ----------- |
| `VikramDataset` | [Massively parallel characterization of transcriptional regulatory elements](https://www.nature.com/articles/s41586-024-08430-9) | 10.1038/s41586-024-08430-9 | HepG2, K562, WTC11 |
| `VikramJointDataset` | [Massively parallel characterization of transcriptional regulatory elements](https://www.nature.com/articles/s41586-024-08430-9) | 10.1038/s41586-024-08430-9 | HepG2, K562, WTC11 |
| `KircherDataset` | [Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution](https://www.nature.com/articles/s41467-019-11526-w) | 10.1038/s41467-019-11526-w | HepG2, K562, etc |
| `MalinoisDataset` | [Machine-guided design of synthetic cell type-specific cis-regulatory elements](https://pmc.ncbi.nlm.nih.gov/articles/PMC10441439/) | 10.1101/2023.08.08.552077 | HepG2, K562, SK-N-SH |
| `MassiveStarrSeqDataset` | [Sequence determinants of human gene regulatory elements](https://www.nature.com/articles/s41588-021-01009-4#citeas) | 10.1038/s41588-021-01009-4 | HepG2, GP5D, RPE1 |
| `SureDataset` | [Genome-wide mapping of autonomous promoter activity in human cells](https://pubmed.ncbi.nlm.nih.gov/28024146/) | 0.1038/nbt.3754 | HepG2, K562 |
| `SharpDataset` | [Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/) | 10.1038/nbt.3678 | HepG2, K562 |
| `FluorescenceDataset` | [Strategies for effectively modelling promoter-driven gene expression using transfer learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/) | 10.1101/2023.02.24.529941| JURKAT, K562, THP1 |
| BACTERIA | ----------- | ----------- | ----------- |
| `EvfratovDataset` | [Application of sorting and next generation sequencing to study 5΄-UTR influence on translation efficiency in Escherichia coli](https://academic.oup.com/nar/article/45/6/3487/2605795) | 10.1093/nar/gkw1141 | ----------- |
| `DeepPromoter` | [Synthetic promoter design in Escherichia coli based on a deep generative network](https://academic.oup.com/nar/article/48/12/6403/5837049) | 10.1093/nar/gkaa325 | ----------- |
| YEAST | ----------- | ----------- | ----------- |
| `DreamDataset` | [Random Promoter DREAM Challenge Consortium. Evaluation and optimization of sequence-based gene regulatory deep learning models.](https://pubmed.ncbi.nlm.nih.gov/38405704/) | 10.1101/2023.04.26.538471 | ----------- |
| `VaishnavDataset` | [The evolution, evolvability and engineering of gene regulatory DNA](https://www.nature.com/articles/s41586-022-04506-6) | 10.1038/s41586-022-04506-6 | ----------- |
| DROSOPHILA | ----------- | ----------- | ----------- |
| `DeepStarrDataset` | [DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers](https://www.nature.com/articles/s41588-022-01048-5) | 10.1038/s41588-022-01048-5 | ----------- |

## Planned datasets

| Prioritet | Artcile and link | DOI |
| ----------- | ----------- | ----------- |
| HUMAN | HUMAN | HUMAN |
| 1 | [Deciphering the functional impact of Alzheimer’s Disease-associated variants in resting and proinflammatory immune cells](https://www.medrxiv.org/content/10.1101/2024.09.13.24313654v1.full-text) | 10.1101/2024.09.13.24313654 |
| 1 | [Uncovering the whole genome silencers of human cells via Ss-STARR-seq](https://www.nature.com/articles/s41467-025-55852-8) | 10.1038/s41467-025-55852-8 |
| BACTERIA | BACTERIA | BACTERIA |
| 2 | [Predictive Modeling of Gene Expression and Localization of DNA Binding Site Using Deep Convolutional Neural Networks](https://www.biorxiv.org/content/10.1101/2024.12.17.629042v1.abstract), this is pre-processed data from [Deciphering the regulatory genome of Escherichia coli, one hundred promoters at a time](https://elifesciences.org/articles/55308) | 10.1101/2024.12.17.629042 |
| 3 [Structure and Evolution of Constitutive Bacterial Promoters](https://www.biorxiv.org/content/10.1101/2020.05.19.104232v1) | 10.1101/2020.05.19.104232 |
| YEAST | YEAST | YEAST |
| 3 | [Deep learning of the regulatory grammar of yeast 5′ untranslated regions from 500,000 random sequences](https://genome.cshlp.org/content/27/12/2015) | 10.1101/gr.224964.117 |