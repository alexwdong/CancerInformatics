import numpy as np
import pandas as pd
import re

from matplotlib import pyplot as plt
import gseapy as gp
import math
from scipy.stats import ttest_ind
import time
import pickle as pkl
from sklearn import preprocessing

# Breast         - 'BCRA'
# Head-neck      - 'HNSC'
# Melanoma(skin) - 'SKCM'
# Stomach        - 'STAD'
# Bladder        - 'BLCA'
# Sarcoma        - 'SARC'
# Pancreatic     - 'PAAD'

def create_dataset(cancer_type, new_tumor_event_site, data_new, filepath_ge, filepath_ph):
    # Get clinical data for cancer type using follow-up file
    data_clinical = data_new[data_new['type']==cancer_type]

    # Read in gene expression data
    data_ge = pd.read_csv(filepath_ge,delimiter = '\t')
    data_ge.rename(columns = {'Ensembl_ID':'Sample'},inplace=True)
    data_ge = data_ge.set_index('Sample').T
    data_ge['sample_id'] = data_ge.index
    
    # Read in phenotype data to get tumor / healthy cell distinction
    data_ph = pd.read_csv(filepath_ph, delimiter = '\t')
    
    # Get sample ids
    idx_ge = [idx[:-4] for idx in data_ge.index]
    idx_clinical = data_clinical['bcr_patient_barcode']
    idx_inter = set(idx_ge).intersection(idx_clinical)
    data_ge['bcr_patient_barcode'] = idx_ge
    
    # Make subset of clinical data for join
    data_clinical_subset = data_clinical[['bcr_patient_barcode','new_tumor_event_site']]
    
    # Make subset of phenotype data for tumor type
    data_ph_subset = data_ph[['submitter_id.samples','sample_type.samples']]
    data_ph_subset = data_ph_subset.rename(columns={'submitter_id.samples': 'sample_id'});
    
    # Combine data using inner join
    data_combined = pd.merge(data_ge,data_clinical_subset, on = 'bcr_patient_barcode',how = 'inner')
    data_combined = pd.merge(data_combined, data_ph_subset, on = 'sample_id',how = 'inner')
    data_combined.index = data_combined['sample_id']
    
    data_final = data_combined[data_combined['new_tumor_event_site'] == new_tumor_event_site]
    
    return data_final

def gene_selection(df, names_df):
    '''
    This function truncates each ensembl ID and selects a subset of the genes
    based on the ensembl IDs listed in names_df
    
    df: DataFrame
        DataFrame of gene expression values returned by the "create_dataset" function
    
    names_df: DataFrame
        DataFrame containing list of ensembl IDs to be retained
    
    results_df: DataFrame
        DataFrame containing the genes listed in names_df
    '''
    col_trunc_list = [x[:15] for x in df.columns[:-4]]
    col_trunc_df = pd.DataFrame(col_trunc_list)
    col_trunc_df = pd.concat([col_trunc_df,pd.DataFrame(df.columns[:-4])],axis=1)
    col_trunc_df.columns = ['ensembl_gene_id','og_index']
    
    final_names = pd.merge(names_df,col_trunc_df)
    gene_list = final_names.og_index.tolist()
    feature_list = gene_list + ['sample_id', 'bcr_patient_barcode', 'new_tumor_event_site','sample_type.samples']
    
    results_df = df[feature_list]
    results_df = results_df.loc[:,~results_df.columns.duplicated()]
    
    return results_df

def remove_mean_cutoffs_tumor(df, cutoff = 1, plotting = True):
    '''
    df: DataFrame
        DataFrame of gene expression values returned by the "gene_selection" function
    
    cutoff: Int 
        Cutoff value for the average gene expression value of each gene
    
    plotting: Boolean
        Plots a histogram of the average values of each column as well as a chart of the
        cummulative counts of each bin
        
    results_df: DataFrame
        Updated version of input DataFrame containing only columns having a mean value
        greater than the cutoff 
    '''
    df = df.drop(columns=['sample_id','bcr_patient_barcode','new_tumor_event_site',])
    tumor_df = df[df['sample_type.samples']=="Primary Tumor"]
    tumor_df = tumor_df.drop(columns = 'sample_type.samples')
        
    mean_vec = tumor_df.mean()
    
    if plotting == True:
    
        plt.hist(mean_vec)

        plt.title('Histogram of mean of features')
        plt.ylabel('Counts')
        plt.xlabel('Mean')
    
        plt.show()
    
        hist, bin_edges = np.histogram(mean_vec,bins=100)
        plt.plot(bin_edges[1:],np.cumsum(hist))
        plt.title('Cumulative count of features vs mean of features')
        plt.ylabel('Cumulative Count of Features')
        plt.xlabel('Mean')
    
    filtered_df = tumor_df[tumor_df.columns[mean_vec>=cutoff]]
    genes_c = [gene for gene in filtered_df.columns]
    genes_c_trunc = [x[:15] for x in genes_c]
    results_df = filtered_df
    results_df.columns = genes_c_trunc
    
    return results_df

def rescale_log(x):
    '''
    The GTEX data uses log_2(fpkm+.001), while the TCGA data uses log_2(fpkm+1).
    We are going to rescale the GTEX data to log_2(fpkm+.1)
    '''
    new_x = math.log(pow(2,x)-.001+1,2)

    return new_x

def rescale_log(x):
    '''
    The GTEX data uses log_2(fpkm+.001), while the TCGA data uses log_2(fpkm+1).
    We are going to rescale the GTEX data to log_2(fpkm+.1)
    '''
    new_x = math.log(pow(2,x)-.001+1,2)

    return new_x

def get_healthy_genes(filepath):
    '''
    This function takes in a filepath for gene expression data of healthy tissue and returns
    the data in DataFrame with all ensembl IDs truncated for version number
    
    filepath: String
        Filepath for a csv file containing the gene expression values of healthy tissue
        corresponding to the tumor site of the cancerous data
        
    healthy_trunc_df: DataFrame
        DataFrame containing gene expression values of the corresponding healthy tissue,
        with each gene name truncated to remove version number
    '''
    healthy_df = pd.read_csv(filepath, index_col=0)
    healthy_df = healthy_df.set_index('sample').T
    
    healthy_df = healthy_df.applymap(rescale_log)
    genes_h = [genes for genes in healthy_df.columns]
    
    # Truncate ensembl ID to remove version number so that IDs line up
    genes_h_trunc = [x[:15] for x in genes_h]
    
    healthy_trunc_df = healthy_df
    healthy_trunc_df.columns = genes_h_trunc
    
    return healthy_trunc_df

def differential_gene_expression_analysis(df, healthy_trunc_df, p_cutoff = 0.01):
    '''
    This function takes in two DataFrames - one containing gene expression values of cancerous
    tissue and the other containing values of healthy tissue and performs differential gene
    expression analysis on the two datasets. This function uses SciPy's ttest_ind function to
    perform the required t-tests.
    
    df: DataFrame
        DataFrame containing gene expression values of cancerous tissue, with each gene
        name truncated to remove version number
    
    healthy_trunc_df: DataFrame
        DataFrame containing gene expression values of healthy tissue, with each gene
        name truncated to remove version number    
    '''
    genes_c_trunc = df.columns
    genes_h_trunc = healthy_trunc_df.columns
    gene_int = set(genes_c_trunc).intersection(set(genes_h_trunc))

    # Loop through all genes for two-sample t-test
    gene_list = []
    mean_ge_list = []

    for gene in genes_c_trunc:
        t, p = ttest_ind(healthy_trunc_df[gene], df[gene], equal_var=False)
        h_mean = healthy_trunc_df[gene].mean()
        c_mean = df[gene].mean()

        if p < p_cutoff and c_mean > h_mean:
            gene_list.append(gene)
            mean_ge_list.append((c_mean, h_mean,c_mean-h_mean))
    
    return gene_list, mean_ge_list

def setup_gene_name_dict(filepath):
    '''
    filepath: String
        Filepath to csv containing mapping from ensembl id to gene name
        
    ensembl2gene: Dict
        Dictionary that takes in an ensembl ID and returns its gene name
        
    gene2ensembl: Dict
        Dictionary that takes in a gene name and returns its ensembl ID 
    '''
    names_full_df = pd.read_csv(filepath)
    ensembl_id_series = names_full_df['ensembl_gene_id']
    gene_id_series = names_full_df['external_gene_name']
    ensembl2gene = {}
    gene2ensembl = {}

    for i in range(names_full_df.shape[0]):
        ensembl2gene[ensembl_id_series[i]] = gene_id_series[i]
        gene2ensembl[gene_id_series[i]] = ensembl_id_series[i]    
    
    return ensembl2gene, gene2ensembl

def run_enrichr(gene_list, gene_sets):
    '''
    gene_list: List
        List containing genes names used for the analysis
    
    gene_sets: List
        List of enrichr gene libaries to use
    
    enr: Enrichr object
        Analysis output, use "enr.results" to print table of results
    '''
    enr = gp.enrichr(gene_list = gene_list,
                     description='pathway',
                     gene_sets= gene_sets,
                     organism='Human',
                     cutoff=0.5
                    )
    
    return enr

def remove_mean_cutoffs_healthy(df, cutoff = 1, plotting = True):
    '''
    df: DataFrame
        DataFrame of gene expression values of healthy tissues
    
    cutoff: Int 
        Cutoff value for the average gene expression value of each gene
    
    plotting: Boolean
        Plots a histogram of the average values of each column as well as a chart of the
        cummulative counts of each bin
        
    results_df: DataFrame
        Updated version of input DataFrame containing only columns having a mean value
        greater than the cutoff 
    '''
        
    mean_vec = df.mean()
    
    if plotting == True:
    
        plt.hist(mean_vec)

        plt.title('Histogram of mean of features')
        plt.ylabel('Counts')
        plt.xlabel('Mean')
    
        plt.show()
    
        hist, bin_edges = np.histogram(mean_vec,bins=100)
        plt.plot(bin_edges[1:],np.cumsum(hist))
        plt.title('Cumulative count of features vs mean of features')
        plt.ylabel('Cumulative Count of Features')
        plt.xlabel('Mean')
    
    results_df = df[df.columns[mean_vec>=cutoff]]
    
    return results_df

def get_healthy2_gene_list(healthy2_results_df, ensembl2gene):
    '''
    health2_results_df: DataFrame
        DataFrame containing gene expression values of healthy tissues, tissue site should
        correspond with the metastatic site of the analyzed cancer type
    '''
    healthy2_gene_list = []
    missing_id_list = []

    # There are some genes missing in the ensembl2gene dictionary
    for ensembl_id in healthy2_results_df.columns:
        try:
            healthy2_gene_list.append(ensembl2gene[ensembl_id])
        except:
            missing_id_list.append(ensembl_id)    
    
    return healthy2_gene_list, missing_id_list

def get_healthy_tissue_gene_exp_df(sample_attr_DS_path,gtex_tcga_path,tissue_str):
    '''
    Inputs:
        sample_attr_DS_path: path to a file that looks like 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
        gtex_tcga_path: path to a file that looks like 'D:\Downloads\TcgaTargetGtex_RSEM_isoform_fpkm\TcgaTargetGtex_RSEM_isoform_fpkm'
    Output:
        gtex_tcga_df: data frame with different genes as rows, and different samples as columns
    
    '''
    
    #Read in attributes file. This file contains a map from Sample ID to Tissue type of sample. 
    #We read in two columns only, SAMPID (Sample ID) and SMTS
    attributesDS_df = pd.read_csv(sample_attr_DS_path,delimiter='\t',usecols=['SAMPID','SMTS'])

    ids_and_tissue_type_df = attributesDS_df[attributesDS_df['SMTS']==tissue_str]
    sample_ids_to_use = ids_and_tissue_type_df['SAMPID'].values # contains all the sample ids that we are looking for
    
    #We'll append 'sample' to the columns to look for, so that cols_to_read contains the name of all the columns to read
    #from the GTEX+TCGA combined file.
    cols_to_read = np.append(np.array(['sample']),sample_ids_to_use)
    #Not all of the GTEX samples are present in the GTEX+TCGA combined file, so we need to find a set intersection between 
    #the SampleIds present in the sample_attr file and the gtex_tcga file
    gtex_tcga_header = pd.read_csv(gtex_tcga_path,delimiter='\t',nrows=1)
    cols_to_read = np.intersect1d(cols_to_read,gtex_tcga_header.columns)
    
    #Now, we'll read the the TCGA+GTEX file for the gene expression of only the specific samples we need.
    print("starting, this takes a while")
    start_time = time.time()
    gtex_tcga_df = pd.read_csv(gtex_tcga_path,delimiter='\t',usecols=cols_to_read)
    print('time elapsed:', time.time() - start_time)
    
    return gtex_tcga_df