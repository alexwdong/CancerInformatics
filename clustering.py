from sklearn import metrics 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl

from data_util import run_enrichr

def elbow_method(gene_array_norm, verbose = True, plotting = True):
    '''
    This function performs the elbow method using distortion or inertia to select the appropriate K for K-means
    
    gene_array_norm: NumPy Array
        Array of gene expression data, normalized
        
    verbose: Boolean
        Prints the distortion and inertia values for each K value when True
        
    plotting: Boolean
        Plots the distortion and inertia values for each K value when True
    '''
    distortions = [] 
    inertias = [] 
    mapping1 = {} 
    mapping2 = {} 
    K = range(1,20) 

    for k in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k).fit(gene_array_norm) 
        kmeanModel.fit(gene_array_norm)     

        distortions.append(sum(np.min(cdist(gene_array_norm, kmeanModel.cluster_centers_, 
                          'euclidean'),axis=1)) / gene_array_norm.shape[0]) 
        inertias.append(kmeanModel.inertia_) 

        mapping1[k] = sum(np.min(cdist(gene_array_norm, kmeanModel.cluster_centers_, 
                     'euclidean'),axis=1)) / gene_array_norm.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 
    
    if verbose == True:
    
        for key,val in mapping1.items(): 
            print(str(key)+' : '+str(val)) 

    if plotting == True:        
            
        plt.plot(K, distortions, 'bx-') 
        plt.xlabel('Values of K') 
        plt.ylabel('Distortion') 
        plt.title('The Elbow Method using Distortion') 
        plt.show() 
    
    if verbose == True:
        
        for key,val in mapping2.items(): 
            print(str(key)+' : '+str(val)) 

    if plotting == True: 
    
        plt.plot(K, inertias, 'bx-') 
        plt.xlabel('Values of K') 
        plt.ylabel('Inertia') 
        plt.title('The Elbow Method using Inertia') 
        plt.show()    
        
    return 

def run_km(cluster_count_list, gene_list, gene_array_norm, gene_array, gene2idx, verbose=False, plotting=False):
    '''
    This function runs K-means various times on the normalized gene data and returns a list of corresponding models
    Note that model selection is determined manually by visually inspecting the covariance matrix of the data when
    sorted by cluster
    
    cluster_count_list: List
        List containing the values of K to be used for clustering
        
    gene_list: Numpy Array
        Array of gene names to be clustered
    
    gene_array_norm: Numpy Array
        Array of gene expression data, normalized
        
    gene_array: Numpy Array
        Array of gene expression data
        
    gene2idx: Dict
        Dictionary that maps each gene to an index - used to reorder genes so that clusters show up together in
        the covariance matrix
        
    verbose: Boolean
        Prints the genes in each cluster when True
    
    plotting: Boolean
        Plots the covariance matrix with entries reordered based on cluster when True
        
    km_list: List
        List of stored K-Means results
    '''
    km_list = []

    for cluster_count in cluster_count_list:
    
        km2 = KMeans(n_clusters=cluster_count,init='random').fit(gene_array_norm)

        if verbose == True:

            for i in range(cluster_count):
                print('----------Cluster {}----------'.format(i))
                print(gene_list[km2.labels_==i])

        gene_list_km = []

        for i in range(cluster_count):
            gene_list_km.append(gene_list[km2.labels_==i])

        gene_index_km = [item for sublist in gene_list_km for item in sublist]
        gene_index_idx = [gene2idx[gene] for gene in gene_index_km]
        
        if plotting == True:
        
            plt.figure(figsize=(10,5))
            df = np.corrcoef(gene_array.T[gene_index_idx])
            plt.matshow(df, fignum=1)
            # plt.xticks(range(len(df.columns)), df.columns)
            # plt.yticks(range(len(df.columns)), df.columns)
            plt.colorbar()
            plt.show()
            
        km_list.append(km2)
            
    return km_list

def get_cluster_dict(cluster_count, gene_list, km_list, best_km_idx, filepath, save_file = True):

    cluster_dict = {}

    for i in range(cluster_count):
        cluster_dict[i] = gene_list[km_list[best_km_idx].labels_==i]

    if save_file == True:
        
        pkl.dump(cluster_dict, open(filepath, "wb"))    
    
    return cluster_dict

def get_cluster_pathways(cluster_dict, ensembl2gene, gene_sets, cutoff, verbose = True):
    '''
    This function takes in a dictionary of clusters and their respective genes and performs gene set enrichment
    analysis on each cluster. The function then takes all of the unique pathways above a given p-value cutoff and 
    returns a set of unique pathways among all clusters.

    cluster_dict: Dict
        Dictionary of cluster-gene pairs, keys are cluster index, values are list of genes
    
    ensembl2gene: Dict
        Dictionary that takes in an ensembl ID and returns its gene name
        
    gene_sets: List
        List of enrichr gene libaries to use
    
    verbose: Boolean
        Prints the number of added pathways for each cluster when True
        
    pw_comb: Set
        Combined set of all unique pathways identified by enrichr for each cluster
    '''
    pw_comb = set()
    
    for cluster, genes in cluster_dict.items():
        gene_enr_list = [ensembl2gene[ensembl] for ensembl in genes]
        enr = run_enrichr(gene_list = gene_enr_list, gene_sets = gene_sets)
        enr_results = enr.results
        pw = set(enr_results[enr_results['Adjusted P-value'] < cutoff]['Term'])
        pw_comb.update(pw)   
        
        if verbose == True:
            print('Cluster {} Pathways: {}'.format(cluster, len(pw)))
    
    return pw_comb