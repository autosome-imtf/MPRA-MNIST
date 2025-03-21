## Для запуска .ipynb в examples помогает прописать такие строки

    clone https://github.com/autosome-imtf/MpraDataset.git
    cd MpraDataset
    pip install setuptools wheel
    python setup.py sdist bdist_wheel
    pip install -e .

We have such datasets:

| Name | Artcile and link | DOI | Cell context |
| ----------- | ----------- | ----------- | ----------- |
| `VikramDataset` | [Massively parallel characterization of transcriptional regulatory elements](https://www.nature.com/articles/s41586-024-08430-9) | 10.1038/s41586-024-08430-9 | Human |
| `VikramJointDataset` | [Massively parallel characterization of transcriptional regulatory elements](https://www.nature.com/articles/s41586-024-08430-9) | 10.1038/s41586-024-08430-9 | Human |
| `MalinoisDataset` | [Machine-guided design of synthetic cell type-specific cis-regulatory elements](https://pmc.ncbi.nlm.nih.gov/articles/PMC10441439/) | 10.1101/2023.08.08.552077 | Human |
| `MassiveStarrSeqDataset` | [Sequence determinants of human gene regulatory elements](https://www.nature.com/articles/s41588-021-01009-4#citeas) | 10.1038/s41588-021-01009-4 | Human |
| `SureDataset` | [Genome-wide mapping of autonomous promoter activity in human cells](https://pubmed.ncbi.nlm.nih.gov/28024146/) | 0.1038/nbt.3754 | Human | 
| `SharpDataset` | [Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/) | 10.1038/nbt.3678 | Human |
| `FluorescenceDataset` | [Strategies for effectively modelling promoter-driven gene expression using transfer learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/) | 10.1101/2023.02.24.529941| Human |
| ----------- | ----------- | ----------- | ----------- |
| `EvfratovDataset` | [Application of sorting and next generation sequencing to study 5΄-UTR influence on translation efficiency in Escherichia coli](https://academic.oup.com/nar/article/45/6/3487/2605795) | 10.1093/nar/gkw1141 | Bacteria |
| `TODO` | [Synthetic promoter design in Escherichia coli based on a deep generative network](https://academic.oup.com/nar/article/48/12/6403/5837049) | 10.1093/nar/gkaa325 | Bacteria |
|$${\color{yellow}`TODO` | [Predictive Modeling of Gene Expression and Localization of DNA Binding Site Using Deep Convolutional Neural Networks](https://www.biorxiv.org/content/10.1101/2024.12.17.629042v1.abstract), this is pre-processed data from [Deciphering the regulatory genome of Escherichia coli, one hundred promoters at a time](https://elifesciences.org/articles/55308) | 10.1101/2024.12.17.629042 | Bacteria}$$ |
| $${\color{red}`TODO`}$$ | $${\color{red}[Structure and Evolution of Constitutive Bacterial Promoters](https://www.biorxiv.org/content/10.1101/2020.05.19.104232v1)}$$ | $${\color{red}10.1101/2020.05.19.104232}$$ | $${\color{red}Bacteria}$$ |
| ----------- | ----------- | ----------- | ----------- |
| `DeepStarrDataset` | [DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers](https://www.nature.com/articles/s41588-022-01048-5) | 10.1038/s41588-022-01048-5 | Drosophila |
| ----------- | ----------- | ----------- | ----------- |
| `TODO` | ----------- | ----------- | Yeast |
| `TODO` | ----------- | ----------- | Yeast |
| $${\color{red}`TODO` | [Deep learning of the regulatory grammar of yeast 5′ untranslated regions from 500,000 random sequences](https://genome.cshlp.org/content/27/12/2015) | 10.1101/gr.224964.117 | Yeast}$$ |
| ----------- | ----------- | ----------- | ----------- |

