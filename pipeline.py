import numpy as np
import pandas as pd
import re

from data_util import create_dataset, gene_selection, remove_mean_cutoffs_tumor
from data_util import get_healthy_genes, differential_gene_expression_analysis, setup_gene_name_dict
from data_util import run_enrichr, remove_mean_cutoffs_healthy, get_healthy2_gene_list

from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from scipy.stats import norm

import gseapy as gp
import math
from matplotlib import pyplot as plt

def analysis_pipeline(cancer_type, tumor_site1, tumor_site2, cutoff_list = [1, 1, 0.01, 0.01, 4, 0.01]):

    #####################################################################################################
    ########################################### Read in Data ############################################
    #####################################################################################################

    filepath_new = '../mmc1-tony.xlsx'
    data_new = pd.read_excel(filepath_new,index_col=0)

    data_tumor_site1 = create_dataset(cancer_type = 'BLCA',
                                    new_tumor_event_site = 'Lung',
                                    data_new = data_new,
                                    filepath_ge = '../BLCA/TCGA-BLCA.htseq_fpkm.tsv',
                                    filepath_ph = '../BLCA/TCGA-BLCA.GDC_phenotype.tsv')

    data_tumor_site2 = create_dataset(cancer_type = 'BLCA',
                                    new_tumor_event_site = 'Liver',
                                    data_new = data_new,
                                    filepath_ge = '../BLCA/TCGA-BLCA.htseq_fpkm.tsv',
                                    filepath_ph = '../BLCA/TCGA-BLCA.GDC_phenotype.tsv')

    print('==========Raw Data==========')
    print('{}-{} sample count: {}'.format(cancer_type, tumor_site1, data_tumor_site1.shape[0]))
    print('{}-{} gene count: {}'.format(cancer_type, tumor_site1, data_tumor_site1.shape[1]))

    print('\n{}-{} sample count: {}'.format(cancer_type, tumor_site2, data_tumor_site2.shape[0]))
    print('{}-{} gene count: {}'.format(cancer_type, tumor_site2, data_tumor_site2.shape[1]))

    sampleAttributesDS_path = '../GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
    gtexTCGA_path = '../TcgaTargetGtex_rsem_gene_fpkm'

    # Healthy tissue data to compare with tumor data from the cancer origin site (e.g. BRCA_lung to healthy breast)    
    healthy1_gene_exp_df = get_healthy_tissue_gene_exp_df(sample_attr_DS_path=sampleAttributesDS_path,
                                                      gtex_tcga_path=gtexTCGA_path,
                                                      tissue_str=cancer_origin)

    # Healthy tissue data to compare with the first metastatic site (e.g. BRCA_lung to healthy lung)
    healthy2_gene_exp_df = get_healthy_tissue_gene_exp_df(sample_attr_DS_path=sampleAttributesDS_path,
                                                      gtex_tcga_path=gtexTCGA_path,
                                                      tissue_str=tumor_site1)

    # Write healthy tissue data to csv
    healthy1_gene_exp_df.to_csv('../{}_gene_exp_healthy.csv'.format(cancer_origin))
    healthy2_gene_exp_df.to_csv('../{}_gene_exp_healthy.csv'.format(tumor_site1))
    
    #####################################################################################################
    ###################### Selection of Genes Related to Transporters and Enzymes #######################
    #####################################################################################################

    # Get names df - List of genes related to transporter and enzyme production
    names_df = pd.read_csv('../ensembl_index.csv',index_col = 0)

    # Get bladder-lung data
    data_tumor_site1 = gene_selection(data_tumor_site1, names_df)

    # Manually inspect to make sure all samples end in "01A" or "01B"
    data_tumor_site1.drop('TCGA-GD-A2C5-11A', inplace=True) # Remove as sample ends in "11A"

    # Get bladder-liver data
    data_tumor_site2 = gene_selection(data_tumor_site2, names_df)

    print('==========Post Gene Selection==========')
    print('{}-{} sample count: {}'.format(cancer_type, tumor_site1, data_tumor_site1.shape[0]))
    print('{}-{} gene count: {}'.format(cancer_type, tumor_site1, data_tumor_site1.shape[1]))

    print('\n{}-{} sample count: {}'.format(cancer_type, tumor_site2, data_tumor_site2.shape[0]))
    print('{}-{} gene count: {}'.format(cancer_type, tumor_site2, data_tumor_site2.shape[1]))    


    #####################################################################################################
    ########################### Histograms and Mean Cutoffs of Tumor Samples ############################
    #####################################################################################################
    
    data_tumor_site1 = remove_mean_cutoffs_tumor(data_tumor_site1, cutoff = cutoff_list[0], plotting = False)
    data_tumor_site2 = remove_mean_cutoffs_tumor(data_tumor_site2, cutoff = cutoff_list[1], plotting = False)

    print('==========Post Mean Cutoffs==========')
    print('{}-{} sample count: {}'.format(cancer_type, tumor_site1, data_tumor_site1.shape[0]))
    print('{}-{} gene count: {}'.format(cancer_type, tumor_site1, data_tumor_site1.shape[1]))

    print('\n{}-{} sample count: {}'.format(cancer_type, tumor_site2, data_tumor_site2.shape[0]))
    print('{}-{} gene count: {}'.format(cancer_type, tumor_site2, data_tumor_site2.shape[1]))    

    
    #####################################################################################################
    ############################### Differential Gene Expression Analysis ###############################
    #####################################################################################################

    filepath = '../Bladder_gene_exp_healthy.csv'
    healthy1_trunc_df = get_healthy_genes(filepath)

    tumor_site1_gene_list, _ = differential_gene_expression_analysis(data_tumor_site1, healthy1_trunc_df)
    tumor_site2_gene_list, _ = differential_gene_expression_analysis(data_tumor_site2, healthy1_trunc_df)

    print('==========Post Differential Gene Expression Analysis==========')
    print('{}-{} gene count: {}'.format(cancer_type, tumor_site1, len(tumor_site1_gene_list)))
    print('{}-{} gene count: {}'.format(cancer_type, tumor_site2, len(tumor_site2_gene_list)))

    
    #####################################################################################################
    ################################### Gene Set Enrichment Analysis ####################################
    #####################################################################################################

    # Setup ensembl/name dictionaries
    filepath = '../Ensembl_symbol_entrez.csv'
    ensembl2gene, gene2ensembl = setup_gene_name_dict(filepath)

    # Setup gene_sets libraries
    gene_sets = ['GO_Biological_Process_2018', 'Reactome_2016']

    # Run GSEA on BRCA-to-lung data
    tumor_site1_gene_enr_list = [ensembl2gene[ensembl] for ensembl in tumor_site1_gene_list]
    tumor_site1_enr = run_enrichr(gene_list = tumor_site1_gene_enr_list, gene_sets = gene_sets)
    tumor_site1_enr_results = tumor_site1_enr.results

    # Run GSEA on BRCA-to-liver data
    tumor_site2_gene_enr_list = [ensembl2gene[ensembl] for ensembl in tumor_site2_gene_list]
    tumor_site2_enr = run_enrichr(gene_list = tumor_site2_gene_enr_list, gene_sets = gene_sets)
    tumor_site2_enr_results = tumor_site2_enr.results

    # Get intersection of pathways for both metastatic sites
    tumor_site1_pw = set(tumor_site1_enr_results[tumor_site1_enr_results['Adjusted P-value'] < cutoff_list[2]]['Term'])
    tumor_site2_pw = set(tumor_site2_enr_results[tumor_site2_enr_results['Adjusted P-value'] < cutoff_list[3]]['Term'])

    # Get difference between intersection of pathways and preferred metastatic site pathway
    tumor_inter_pw = tumor_site1_pw.intersection(tumor_site2_pw)
    tumor_site1_diff_pw = tumor_site1_pw.difference(tumor_inter_pw)

    
    #####################################################################################################
    ############################## Mean Cutoffs for Healthy Tissue Samples ##############################
    #####################################################################################################

    filepath = '../Lung_gene_exp_healthy.csv'
    healthy2_trunc_df = get_healthy_genes(filepath)

    healthy2_results_df = remove_mean_cutoffs_healthy(healthy2_trunc_df, cutoff = cutoff_list[4], plotting = False)

    print('Pre-cutoff - healthy tissue gene count: {}'.format(healthy2_trunc_df.shape[1]))
    print('Post-cutoff - healthy tissue gene count: {}'.format(healthy2_results_df.shape[1]))

    
    #####################################################################################################
    ######################### Gene Set Enrichment Analysis for Healthy Samples ##########################
    #####################################################################################################

    healthy2_gene_list, missing_id_list = get_healthy2_gene_list(healthy2_results_df)

    # Run Enrichr on healthy2 gene list
    healthy2_enr = run_enrichr(gene_list = healthy2_gene_list, gene_sets = gene_sets)
    healthy2_enr_results = healthy2_enr.results

    healthy2_pw = set(healthy2_enr_results[healthy2_enr_results['Adjusted P-value'] < cutoff_list[5]]['Term'])

    # Get intersection between healthy2 and tumor_site1_diff
    tumor_site1_diff_healthy2_inter_pw = tumor_site1_diff_pw.intersection(healthy2_pw)

    # Print pre and post pathway counts
    print('Initial significant pathway count: {}'.format(len(tumor_site1_diff_pw)))
    print('Post-intersection pathway count: {}'.format(len(tumor_site1_diff_healthy2_inter_pw)))

    return list(tumor_site1_diff_healthy2_inter_pw)