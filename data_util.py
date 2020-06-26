import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt

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