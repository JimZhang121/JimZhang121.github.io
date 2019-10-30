import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import re
from sklearn.impute import SimpleImputer as Imputer
from numpy import random
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
pam50_proteins = 'PAM50_proteins.csv'
pam50 = pd.read_csv(pam50_proteins,header=0)
proteomes_file = 'proteomes_CPTAC_itraq_77_cancer.csv'

clinical_info = 'clinical_data_breast_cancer.csv'

data = pd.read_csv(proteomes_file,header=0,index_col=0)
clinical = pd.read_csv(clinical_info,header=0,index_col=0)
data.drop(['gene_symbol','gene_name'],axis=1,inplace=True)

data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)

data = data.transpose()
 
clinical = clinical.loc[[x for x in clinical.index.tolist() if x in data.index],:]

merged = data.merge(clinical,left_index=True,right_index=True)
 
processed = merged
 
processed_numerical = processed.loc[:,[x for x in processed.columns if bool(re.search("NP_|XP_",x)) == True]]

processed_numerical_p50 = processed_numerical.loc[:,processed_numerical.columns.isin(pam50['RefSeqProteinID'])]
processed_numerical_p50.replace([np.inf, -np.inf], np.nan)
imputer = Imputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(processed_numerical_p50)
processed_numerical_p50 = imputer.transform(processed_numerical_p50)

print(set(clinical['PAM50 mRNA']))
k = 4
clusterer = KMeans(n_clusters=k, n_jobs=4)
clusterer.fit(processed_numerical_p50)
clusterer.labels_

def compare_k_means(k_list,data):
    """
        compare_k_means: Run clustering with different k and check the metrics

        Args:
            k_list (list): different number of k's to determine how many clusters we want
            data (dataframe): the data used for clustering

        Returns:
            None
    """
    for k in k_list:
        clusterer = KMeans(n_clusters=k, n_jobs=4)
        clusterer.fit(data)
        
        print("Silhouette Coefficient for k == %s: %s" % (
        k, round(metrics.silhouette_score(data, clusterer.labels_), 4)))
        
        print("Homogeneity score for k == %s: %s" % (
        k, round(metrics.homogeneity_score(processed['PAM50 mRNA'], clusterer.labels_), 4)))
        print("------------------------")
		
		
n_clusters = [2,3,4,5,6,7,8,10,20,79]
compare_k_means(n_clusters,processed_numerical_p50)

processed_numerical_random = processed_numerical.iloc[:,random.choice(range(processed_numerical.shape[1]),43)]
imputer_rnd = imputer.fit(processed_numerical_random)
processed_numerical_random = imputer_rnd.transform(processed_numerical_random)

print("====== Now it's the random proteins ======")
compare_k_means(n_clusters,processed_numerical_random)

clusterer_final = KMeans(n_clusters=3, n_jobs=4)
clusterer_final = clusterer_final.fit(processed_numerical_p50)
processed_p50_plot = pd.DataFrame(processed_numerical_p50)
processed_p50_plot['KMeans_cluster'] = clusterer_final.labels_
processed_p50_plot.sort_values('KMeans_cluster',axis=0,inplace=True)

 
fig = plt.figure(figsize = (14,10))
ax = fig.add_subplot(1,1,1)
processed_p50_plot.index.name = 'Patient'
processed_p50_plot.head()
sb.heatmap(processed_p50_plot.iloc[:,:-1], xticklabels=[], yticklabels=[]) ## The x-axis are the PAM50 proteins we used and the right-most column is the cluster marker
ax.set_xlabel('Protein', fontsize = 20)
ax.set_ylabel('Patient', fontsize = 20)
ax.set_title('Protein Expressions for Patients', fontsize = 24)
ax.yaxis.set_tick_params(labelsize=12, rotation=0)
ax.xaxis.set_tick_params(labelsize=12, rotation=0)