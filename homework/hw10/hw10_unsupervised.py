# hw10 - unsupervised learning

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from scipy.cluster import hierarchy

plt.style.use('seaborn-white');
#%%
df=pd.read_csv('../datasets/USArrests.csv', index_col=0);
print(df.head());
print(df.info());
print ('MEAN')
print(df.mean());
print ('VARIANCE');
print (df.var());

X=pd.DataFrame(scale(df),index=df.index, columns=df.columns);
pca_model=PCA().fit(X);
pca_loadings=pd.DataFrame(pca_model.components_.T, index=df.columns, columns=['V1', 'V2', 'V3', 'V4']);
print(pca_loadings);

#fit PCA model and transform X to get principal components
pca=PCA()
df_plot=pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X.index);
print(df_plot);

fig, ax1=plt.subplots(figsize=(9,7));

ax1.set_xlim(-3.5, 3.5);
ax1.set_ylim(-3.5, 3.5);

for i in df_plot.index:
    ax1.annotate(i, (df_plot.PC1.loc[i], -df_plot.PC2.loc[i]), ha='center');

ax1.hlines(0,-3.5, 3.5, linestyles='dotted', colors='grey');
ax1.vlines(0,-3.5, 3.5, linestyles='dotted', colors='grey');

ax1.set_xlabel('First Principal Component');
ax1.set_ylabel('Second Principal Component');

ax2=ax1.twinx().twiny();

ax2.set_ylim(-1,1);
ax2.set_xlim(-1,1);
ax2.tick_params(axis='y', colors='orange');
ax2.set_xlabel('Principal Component Loading Vectors', color='Orange');

#plot labels for vectors
a=1.07; #offset parameter

for i in pca_loadings[['V1', 'V2']].index:
    ax2.annotate(i, (pca_loadings.V1.loc[i]*a, -pca_loadings.V2.loc[i]*a), color='orange');

ax2.arrow(0, 0, pca_loadings.V1[0], -pca_loadings.V2[0]);
ax2.arrow(0, 0, pca_loadings.V1[1], -pca_loadings.V2[1]);
ax2.arrow(0, 0, pca_loadings.V1[2], -pca_loadings.V2[2]);
ax2.arrow(0, 0, pca_loadings.V1[3], -pca_loadings.V2[3]);

fig.savefig('USArrests_PCA_biplot.jpeg');

#variance explained
print('STD DEVIATION OF PRINCIPAL COMPONENTS')
print(np.sqrt(pca.explained_variance_));
print('VARIANCE EXPLAINED BY PRINCIPAL COMPONENTS')
print(pca.explained_variance_);
print ('FRACTIONAL VARIANCE EXPLAINED BY PRINCIPAL COMPONENETS');
print(pca.explained_variance_ratio_);

fig2, ax2=plt.subplots(figsize=(7,5));
plt.plot([1,2,3,4], pca.explained_variance_ratio_,'-o',label='Individual Component');
plt.plot([1,2,3,4], np.cumsum(pca.explained_variance_ratio_), '-s', label='Cumulative');
plt.xlim(0.75, 4.25);
plt.xlabel('Principal Component');
plt.ylim (0,1.05);
plt.xticks([1,2,3,4]);
plt.legend(loc=2);
fig2.savefig('USArrests_PCA_variance_explained.jpeg')

#%% question 2 - clustering algorithms

np.random.seed(2);
X=np.random.standard_normal((50,2));
X[:25,0]=X[:25,0]+3;
X[:25,1]=X[:25,0]-4;

#K=2
km1=KMeans(n_clusters=2, n_init=20);
km1.fit(X);
print(km1.labels_);
print(km1.inertia_);

#K=3;
np.random.seed(4);
km2=KMeans(n_clusters=3, n_init=4);
km2.fit(X);
print(km2.labels_);
print(km2.inertia_);

fig3, (ax1, ax2)=plt.subplots(1,2,figsize=(14,5));
ax1.scatter(X[:,0], X[:,1], s=40, c=km1.labels_, cmap=plt.cm.prism);
ax1.set_title('K-Means Clustering Results with K=2');
ax1.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);

ax2.scatter(X[:,0], X[:,1], s=40, c=km2.labels_, cmap=plt.cm.prism);
ax2.set_title('K-Means Clustering Results with K=3');
ax2.scatter(km2.cluster_centers_[:,0], km2.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);
fig3.savefig('KMeans_example.jpeg');

#%% part 3 - heirarchical clustering

fig4, (ax1,ax2,ax3)=plt.subplots(3,1,figsize=(15,18));

for linkage, cluster, ax in zip([hierarchy.complete(X), hierarchy.average(X), hierarchy.single(X)], ['c1', 'c2', 'c3'], [ax1, ax2, ax3]):
    cluster=hierarchy.dendrogram(linkage, ax=ax, color_threshold=0);

ax1.set_title('Complete Linkage');
ax2.set_title('Average Linkage');
ax3.set_title('Single Linkage');
fig4.savefig('heirarchical_dendrogram.jpeg')

