# CancerInformatics

This repository is an implementation for testing genetics factors related to the seed and soil hypothesis of metastatic cancer.

The repo is organized as follows:
The analysis_pipeline.ipynb notebook contains all of the code necessary to (1) read in and process the data, (2) perform differential gene expression analysis to identify upregulated genes, (3) perform k-means clustering to identify functional groups, (4) perform gene set enrichment analysis to extract biological pathways from the identified clusters, and (5) use these pathways to identify potential genetic determinants of the seed and soil hypothesis.

The clustering.py file contain all the necessary code used by the analysis_pipeline.ipynb notebook to perform either k-means or k-clique clustering. The data_util.py file contains code for reading in and processing the data used by the analysis_pipeline.ipynb notebook.













