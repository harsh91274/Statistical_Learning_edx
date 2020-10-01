# hw10 part 2 - NCl60 data example

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
df2 = pd.read_csv('../datasets/NCI60_X.csv').drop('Unnamed: 0', axis=1);
df2.columns = np.arange(df2.columns.size);
print(df2.info());

X = pd.DataFrame(scale(df2));
print(X.shape);

y = pd.read_csv('../datasets/NCI60_y.csv', usecols=[1], skiprows=1, names=['type'])
print(y.shape);

print(y.type.value_counts());

pca2 = PCA();
df2_plot = pd.DataFrame(pca2.fit_transform(X));

fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6));
color_idx = pd.factorize(y.type)[0];
cmap = plt.cm.hsv;

#%%
# Left plot
ax1.scatter(df2_plot.iloc[:,0], -df2_plot.iloc[:,1], c=color_idx, cmap=cmap, alpha=0.5, s=50)
ax1.set_ylabel('Principal Component 2');

# Right plot
ax2.scatter(df2_plot.iloc[:,0], df2_plot.iloc[:,2], c=color_idx, cmap=cmap, alpha=0.5, s=50)
ax2.set_ylabel('Principal Component 3');

# Custom legend for the classes (y) since we do not create scatter plots per class (which could have their own labels).
handles = []
labels = pd.factorize(y.type.unique());
norm = mpl.colors.Normalize(vmin=0.0, vmax=14.0);

for i, v in zip(labels[0], labels[1]):
    handles.append(mpl.patches.Patch(color=cmap(norm(i)), label=v, alpha=0.5))

ax2.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# xlabel for both plots
for ax in fig1.axes:
    ax.set_xlabel('Principal Component 1');

fig1.savefig('NCL60_PCA.jpeg')

#%%
pd.DataFrame([df2_plot.iloc[:,:5].std(axis=0, ddof=0), pca2.explained_variance_ratio_[:5], np.cumsum(pca2.explained_variance_ratio_[:5])], index=['Standard Deviation', 'Proportion of Variance', 'Cumulative Proportion'], columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5']);
fig2,ax=plt.subplots(figsize=(7,7));
df2_plot.iloc[:,:10].var(axis=0, ddof=0).plot(kind='bar', rot=0)
plt.ylabel('Variances');
fig2.savefig('NCL60_variances.jpeg');
#%%
fig3 , (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

# Left plot
ax1.plot(pca2.explained_variance_ratio_, '-o')
ax1.set_ylabel('Proportion of Variance Explained')
ax1.set_ylim(ymin=-0.01)

# Right plot
ax2.plot(np.cumsum(pca2.explained_variance_ratio_), '-ro')
ax2.set_ylabel('Cumulative Proportion of Variance Explained')
ax2.set_ylim(ymax=1.05)

for ax in fig3.axes:
    ax.set_xlabel('Principal Component')
    ax.set_xlim(-1,65);

fig3.savefig('NCL60_variance_explained.jpeg')
#%%
X= pd.DataFrame(scale(df2), index=y.type, columns=df2.columns);

fig4, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,20))

for linkage, cluster, ax in zip([hierarchy.complete(X), hierarchy.average(X), hierarchy.single(X)],
                                ['c1','c2','c3'],
                                [ax1,ax2,ax3]):
    cluster = hierarchy.dendrogram(linkage, labels=X.index, orientation='right', color_threshold=0, leaf_font_size=10, ax=ax)

ax1.set_title('Complete Linkage')
ax2.set_title('Average Linkage')
ax3.set_title('Single Linkage');
fig4.savefig('NCL60_hierarchical.jpeg')

#%%
fig5,_= plt.subplots(figsize=(10,20))
cut4 = hierarchy.dendrogram(hierarchy.complete(X),
                            labels=X.index, orientation='right', color_threshold=140, leaf_font_size=10)
plt.vlines(140,0,plt.gca().yaxis.get_data_interval()[1], colors='r', linestyles='dashed');
fig5.savefig('NCL60_hierarchical2.jpeg');

#%%
np.random.seed(2)
km4 = KMeans(n_clusters=4, n_init=50)
km4.fit(X)
print(km4.labels_);
print(pd.Series(km4.labels_).value_counts().sort_index());

plt.figure();
cut4b = hierarchy.dendrogram(hierarchy.complete(X), truncate_mode='lastp', p=4, show_leaf_counts=True);
plt.savefig('NCL60_hierarchical3_cutp4.jpeg');

#%%
plt.figure(figsize=(10,20));
pca_cluster = hierarchy.dendrogram(hierarchy.complete(df2_plot.iloc[:,:5]), labels=y.type.values, orientation='right', color_threshold=100, leaf_font_size=10)
plt.savefig('NCL60_hierarchical4_PCA5.jpeg');
