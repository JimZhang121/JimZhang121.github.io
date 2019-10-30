%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
import random as rd
from sklearn.cluster import KMeans
from sklearn import metrics
import re
from sklearn.impute import SimpleImputer as Imputer
from numpy import random
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def uniqueish_color():
    """There're better ways to generate unique colors, but this isn't awful."""
    return plt.cm.gist_ncar(np.random.random())
  
colors_ = ['red', 'magenta', 'orange', 'green', 'crimson', 'grey', 'brown']

dataset_path = "proteomes_CPTAC_itraq_77_cancer.csv"
clinical_info = "clinical_data_breast_cancer.csv"
pam50_proteins = "PAM50_proteins.csv"

## Load data
data = pd.read_csv(dataset_dir+dataset_path,header=0,index_col=0)
clinical = pd.read_csv(dataset_dir+clinical_info,header=0,index_col=0)## holds clinical information about each patient/sample
pam50 = pd.read_csv(dataset_dir+pam50_proteins,header=0)

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

BRCA_RNA_df = pd.read_csv(dataset_dir+"BRCA_RNA_Seq_Samples_Overlap_with_Proteomes_RSEM_est.csv", index_col=0)
BRCA_RNA_df = BRCA_RNA_df * 1e6

idx = BRCA_RNA_df.index

#将第一列的名称进行处理，删除 \ 后的数字
BRCA_RNA_df.index = list(map(lambda x: x.split('|')[0].strip(), idx))

### Filter out non-expressed genes
##删除带？ 且 为0
BRCA_RNA_df = BRCA_RNA_df.loc[BRCA_RNA_df.index != '?', :]
BRCA_RNA_df = BRCA_RNA_df.loc[BRCA_RNA_df.sum(axis=1) > 0, :]

## Filter out lowly expressed genes
mask_low_vals = (BRCA_RNA_df > 0.3).sum(axis=1) > 2
BRCA_RNA_df = BRCA_RNA_df.loc[mask_low_vals, :]

scaled_data = BRCA_RNA_df.T
# scaled_data = preprocessing.scale(scaled_data)

# # 2D
pca = PCA(n_components=2) # create a PCA object
pca.fit(scaled_data) # do the math
pca_data = pca.transform(scaled_data) # get PCA coordinates for scaled_data
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1) # get the variance explained by each component

principalDf = pd.DataFrame(data = np.array(pca_data), index=BRCA_RNA_df.columns, columns = ['principal component 1', 'principal component 2'])
principalDf = principalDf.merge(processed, left_index=True, right_index=True)
principalDf = principalDf[['principal component 1', 'principal component 2', 'PAM50 mRNA']].drop_duplicates()

# # visualization
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(1,1,1) 

# principalDf.plot.scatter(x='principal component 1', y='principal component 2', label=principalDf.index, ax=ax);
PAM50_mRNA = list(set(processed['PAM50 mRNA']))
for name, group in principalDf.groupby(principalDf['PAM50 mRNA']):
    group.plot.scatter(x='principal component 1', y='principal component 2', color=colors_[PAM50_mRNA.index(name)], s=60, label=name, ax=ax)

ax.set_xlabel('Principal Component 1 - {0}%'.format(per_var[0]), fontsize = 15)
ax.set_ylabel('Principal Component 2 - {0}%'.format(per_var[1]), fontsize = 15)
ax.set_title('PCA for RNA Data - By PAM50', fontsize = 20)
ax.yaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
# ax.set_yscale('log', basey=10)
# ax.set_xscale('log', basex=10)

ax.legend(fontsize=15)
ax.grid()

### Filter out the outliers detected by PCA
principalDf_ = principalDf[principalDf['principal component 1'] < 40000]
principalDf_ = principalDf_[principalDf_['principal component 2'] < 40000]

# # visualization
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(1,1,1) 

# principalDf.plot.scatter(x='principal component 1', y='principal component 2', label=principalDf.index, ax=ax)
PAM50_mRNA = list(set(processed['PAM50 mRNA']))
for name, group in principalDf_.groupby(principalDf_['PAM50 mRNA']):
    group.plot.scatter(x='principal component 1', y='principal component 2', color=colors_[PAM50_mRNA.index(name)], s=60, label=name, ax=ax)

ax.set_xlabel('Principal Component 1 - {0}%'.format(per_var[0]), fontsize = 15)
ax.set_ylabel('Principal Component 2 - {0}%'.format(per_var[1]), fontsize = 15)
ax.set_title('PCA for RNA Data (No Outliers) - By PAM50', fontsize = 20)
ax.yaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=14)
# ax.set_yscale('log', basey=10)
# ax.set_xscale('log', basex=10)

ax.legend(fontsize=15)
ax.grid()

