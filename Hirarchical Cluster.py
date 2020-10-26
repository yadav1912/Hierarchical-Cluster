# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 00:04:22 2020

@author: HP
"""

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('KMeans.csv')
X = dataset.iloc[:, [1, 2]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Distances')
plt.show();

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity ='euclidean', linkage = 'ward' )
y_hc=hc.fit_predict(X)



# Visualising the clusters
plt.scatter(X[hc == 0, 0], X[hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[hc == 1, 0], X[hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[hc == 2, 0], X[hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[hc == 3, 0], X[hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()