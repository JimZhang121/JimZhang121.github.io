import pandas as pd
import numpy as np
import random as rd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing
import sys
import seaborn as sns
import re
from sklearn.impute import SimpleImputer as Imputer
from sklearn.cluster import KMeans
from numpy import random

% matplotlib inline

def uniqueish_color():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())
  
colors_ = ['red', 'magenta', 'orange', 'green', 'crimson', 'grey', 'brown'


dataset_path = "proteomes_CPTAC_itraq_77_cancer.csv"
clinical_info = "clinical_data_breast_cancer.csv"
pam50_proteins = "PAM50_proteins.csv"

## Load data
data = pd.read_csv(dataset_path,header=0,index_col=0)
clinical = pd.read_csv(clinical_info,header=0,index_col=0)## holds clinical information about each patient/sample
pam50 = pd.read_csv(pam50_proteins,header=0)

## Drop unused information columns
data.drop(['gene_symbol','gene_name'],axis=1,inplace=True)
 
## Change the protein data sample names to a format matching the clinical data set
data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)

## Transpose data for the clustering algorithm since we want to divide patient samples, not proteins
data = data.transpose()
 
## Drop clinical entries for samples not in our protein data set
clinical = clinical.loc[[x for x in clinical.index.tolist() if x in data.index],:]

## Add clinical meta data to our protein data set, note: all numerical features for analysis start with NP_ or XP_
merged = data.merge(clinical,left_index=True,right_index=True)
 
## Change name to make it look nicer in the code!
processed = merged

## Numerical data for the algorithm, NP_xx/XP_xx are protein identifiers from RefSeq database
processed_numerical = processed.loc[:,[x for x in processed.columns if bool(re.search("NP_|XP_",x)) == True]]
 
## Select only the PAM50 proteins - known panel of genes used for breast cancer subtype prediction
processed_numerical_p50 = processed_numerical.loc[:,processed_numerical.columns.isin(pam50['RefSeqProteinID'])]
# processed_numerical_p50.replace([np.inf, -np.inf], np.nan)

## Impute missing values (maybe another method would work better?)
imputer = Imputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(processed_numerical_p50)
processed_numerical_p50 = imputer.transform(processed_numerical_p50)

scaled_data = preprocessing.scale(processed_numerical_p50)

# # 2D
pca = sklearnPCA(n_components=2) # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1) # get the variance explained by each component

### Doesn't function properly
# ===================================
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)
# ===================================
principalDf = pd.DataFrame(data = np.array(pca_data), index=processed['PAM50 mRNA'], columns = ['principal component 1', 'principal component 2'])

# # visualization
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1 - {0}%'.format(per_var[0]), fontsize = 15)
ax.set_ylabel('Principal Component 2 - {0}%'.format(per_var[1]), fontsize = 15)
ax.set_title('PCA for Proteomics Data - By PAM50', fontsize = 20)
ax.yaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)

PAM50_mRNA = list(set(processed['PAM50 mRNA']))
# principalDf.plot.scatter(x='principal component 1', y='principal component 2', label=principalDf.index, ax=ax);
for name, group in principalDf.groupby(principalDf.index):
    group.plot.scatter(x='principal component 1', y='principal component 2', color=colors_[PAM50_mRNA.index(name)], s=60, label=name, ax=ax)
ax.legend(fontsize=15)
ax.grid()

fig, axes = plt.subplots(1,3,sharey=True)
fig.set_size_inches((20,7))

for idx in range(3, 6):
    ax = axes[idx-3]
    clusterer_final = KMeans(n_clusters=idx, n_jobs=4)
    clusterer_final = clusterer_final.fit(processed_numerical_p50)
    # processed_p50_plot['KMeans_cluster'] = clusterer_final.labels_
    # processed_p50_plot.sort_values('KMeans_cluster',axis=0,inplace=True)

    principalDf[str(idx)+'_clusters_label'] = clusterer_final.labels_
    # principalDf.head()
    # visualization
    
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    
    
    # principalDf.plot.scatter(x='principal component 1', y='principal component 2', label=principalDf.index, ax=ax);
    for name, group in principalDf.groupby(principalDf[str(idx)+'_clusters_label']):
        group.plot.scatter(x='principal component 1', y='principal component 2', color=colors_[name], s=60, label='Cluster '+str(name+1), ax=ax)
    
    if (idx-3 == 1):
        ax.set_xlabel('Principal Component 1 - {0}%'.format(per_var[0]), fontsize = 15)
        ax.set_title('PCA for All Proteomics Data - By K-Means Clustering', fontsize = 20)
    else:
        ax.set_xlabel('')
        
    if (idx-3 == 0):
        ax.set_ylabel('Principal Component 2 - {0}%'.format(per_var[1]), fontsize = 15)
        
    ax.legend(fontsize=14)
    ax.grid()

plt.tight_layout()
plt.subplots_adjust(top=0.88,wspace = 0.03,hspace=0.12)