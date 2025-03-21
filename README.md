## To launch .ipynb в examples 

    clone https://github.com/autosome-imtf/MpraDataset.git
    cd MpraDataset
    pip install setuptools wheel
    python setup.py sdist bdist_wheel
    pip install -e .

## We have such datasets:

| Name | Artcile and link | DOI | Cell context | Cell types | Case example link |
| ----------- | ----------- | ----------- | ----------- | -----------| -----------|
| `VikramDataset` | [Massively parallel characterization of transcriptional regulatory elements](https://www.nature.com/articles/s41586-024-08430-9) | 10.1038/s41586-024-08430-9 | Human | HepG2, K562, WTC11 | [example](https://github.com/autosome-imtf/MpraDataset/blob/main/examples/VikramDataset_example.ipynb) |
| `VikramJointDataset` | [Massively parallel characterization of transcriptional regulatory elements](https://www.nature.com/articles/s41586-024-08430-9) | 10.1038/s41586-024-08430-9 | Human | HepG2, K562, WTC11 | [example](https://github.com/autosome-imtf/MpraDataset/blob/main/examples/VikramJointDataset_example.ipynb) |
| `MalinoisDataset` | [Machine-guided design of synthetic cell type-specific cis-regulatory elements](https://pmc.ncbi.nlm.nih.gov/articles/PMC10441439/) | 10.1101/2023.08.08.552077 | Human | HepG2, K562, SK-N-SH | [example](https://github.com/autosome-imtf/MpraDataset/blob/main/examples/Malinois_example.ipynb) |
| `MassiveStarrSeqDataset` | [Sequence determinants of human gene regulatory elements](https://www.nature.com/articles/s41588-021-01009-4#citeas) | 10.1038/s41588-021-01009-4 | Human | HepG2, GP5D, RPE1 | [example](https://github.com/autosome-imtf/MpraDataset/blob/main/examples/MassiveStarrSeq_example.ipynb) |
| `SureDataset` | [Genome-wide mapping of autonomous promoter activity in human cells](https://pubmed.ncbi.nlm.nih.gov/28024146/) | 0.1038/nbt.3754 | Human | HepG2, K562 | [example](https://github.com/autosome-imtf/MpraDataset/blob/main/examples/SureDataset_example.ipynb) | 
| `SharpDataset` | [Genome-scale high-resolution mapping of activating and repressive nucleotides in regulatory regions](https://pmc.ncbi.nlm.nih.gov/articles/PMC5125825/) | 10.1038/nbt.3678 | Human | HepG2, K562 | [example](https://github.com/autosome-imtf/MpraDataset/blob/main/examples/SharprDataset_example.ipynb) |
| `FluorescenceDataset` | [Strategies for effectively modelling promoter-driven gene expression using transfer learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002662/) | 10.1101/2023.02.24.529941| Human | JURKAT, K562, THP1 | [example](https://github.com/autosome-imtf/MpraDataset/blob/main/examples/FluorescenceDataset_example.ipynb) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| `EvfratovDataset` | [Application of sorting and next generation sequencing to study 5΄-UTR influence on translation efficiency in Escherichia coli](https://academic.oup.com/nar/article/45/6/3487/2605795) | 10.1093/nar/gkw1141 | Bacteria | ----------- | [example](https://github.com/autosome-imtf/MpraDataset/blob/main/examples/EvfratovDataset_example.ipynb) |
| `TODO` | [Synthetic promoter design in Escherichia coli based on a deep generative network](https://academic.oup.com/nar/article/48/12/6403/5837049) | 10.1093/nar/gkaa325 | Bacteria | ----------- | ----------- |
|$${\color{yellow}TODO}$$ | [Predictive Modeling of Gene Expression and Localization of DNA Binding Site Using Deep Convolutional Neural Networks](https://www.biorxiv.org/content/10.1101/2024.12.17.629042v1.abstract), this is pre-processed data from [Deciphering the regulatory genome of Escherichia coli, one hundred promoters at a time](https://elifesciences.org/articles/55308) | $${\color{yellow}10.1101/2024.12.17.629042}$$ | $${\color{yellow}Bacteria}$$ | ----------- | ----------- |
| $${\color{red}TODO}$$ | [Structure and Evolution of Constitutive Bacterial Promoters](https://www.biorxiv.org/content/10.1101/2020.05.19.104232v1) | $${\color{red}10.1101/2020.05.19.104232}$$ | $${\color{red}Bacteria}$$ | ----------- | ----------- |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| `TODO` | ----------- | ----------- | Yeast | ----------- | ----------- |
| `TODO` | ----------- | ----------- | Yeast | ----------- | ----------- |
| $${\color{red}TODO}$$ | [Deep learning of the regulatory grammar of yeast 5′ untranslated regions from 500,000 random sequences](https://genome.cshlp.org/content/27/12/2015) | $${\color{red}10.1101/gr.224964.117}$$ | $${\color{red}Yeast}$$ | ----------- | ----------- |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| `DeepStarrDataset` | [DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers](https://www.nature.com/articles/s41588-022-01048-5) | 10.1038/s41588-022-01048-5 | Drosophila | ----------- | ----------- |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |

## Is it better? 

<table>
  <tr>
    <th>Name</th>
    <th>Artcile and link</th>
    <th>DOI</th>
    <th>Cell context</th>
    <th>Cell types</th>
    <th>Case example link</th>  
  </tr>
  <tr>
    <td>VikramDataset</td>
    <td><a href="https://www.nature.com/articles/s41586-024-08430-9">Massively parallel characterization of transcriptional regulatory elements</a></td>
    <td>10.1038/s41586-024-08430-9</td>
    <td>Human</td>
    <td>HepG2, K562, WTC11</td>
    <td><a href="https://github.com/autosome-imtf/MpraDataset/blob/main/examples/VikramDataset_example.ipynb">example</a></td> 
  </tr>
  <tr>
    <td>VikramJointDataset</td>
    <td><a href="https://www.nature.com/articles/s41586-024-08430-9">Massively parallel characterization of transcriptional regulatory elements</a></td>
    <td>10.1038/s41586-024-08430-9</td>
    <td>Human</td>
    <td>HepG2, K562, WTC11</td>
    <td><a href="https://github.com/autosome-imtf/MpraDataset/blob/main/examples/VikramDataset_example.ipynb">example</a></td> 
  </tr>
  <tr>
    <td>Данные 1</td>
    <td>Данные 2</td>
    <td>Данные 3</td>
    <td>-----------</td>
    <td>-----------</td>
    <td>-----------</td>
    <td></td>
  </tr>
  <tr style="background-color: red;">
    <td>TODO</td>
    <td><a href="https://www.biorxiv.org/content/10.1101/2024.12.17.629042v1.abstract">Predictive Modeling of Gene Expression and Localization of DNA Binding Site Using Deep Convolutional Neural Networks</a>, this is pre-processed data from <a href="https://elifesciences.org/articles/55308">Deciphering the regulatory genome of Escherichia coli, one hundred promoters at a time</a></td>
    <td>10.1101/2024.12.17.629042</td>
    <td>Bacteria</td>
    <td>-----------</td>
    <td>-----------</td>
  </tr>
  <tr>
    <td>-----------</td>
    <td>-----------</td>
    <td>-----------</td>
    <td>-----------</td>
    <td>-----------</td>
    <td>-----------</td>
  </tr>
    <td>`DeepStarrDataset`</td>
    <td><a href="https://www.nature.com/articles/s41588-022-01048-5">DeepSTARR predicts enhancer activity from DNA sequence and enables the de novo design of synthetic enhancers</a></td>
    <td>10.1038/s41588-022-01048-5</td>
    <td>Drosophila</td>
    <td>-----------</td>
    <td>-----------</td>
</table>
