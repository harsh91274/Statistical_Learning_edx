#Statistical Learning Ch6 Lab
#Part 3 - Principal component and partial least squares

#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
import itertools
import sklearn
from tqdm import tqdm #progress bars for iterables
from operator import itemgetter
from itertools import combinations
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression


plt.style.use('ggplot');

#%%
hitters=pd.read_csv('../datasets/Hitters.csv', index_col=0).dropna();
dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']]);
hitters = hitters.drop(['League', 'Division', 'NewLeague'], axis=1);
hitters = pd.concat([hitters, dummies[['League_N', 'Division_W', 'NewLeague_N']]],axis=1);
print(hitters.head());

X=hitters.drop('Salary', axis=1).values;
Y=hitters.Salary.values;

#Principal Component Regression
pca=PCA();
X_pcs=pca.fit_transform(scale(X));
print('Principal Component Matrix size (num_samples, num_pcs): ',X_pcs.shape, '\n');

loadings=pca.components_
print('The loadings matrix has shape (num_pcs, num_features): ', loadings.shape, '\n');

scores=[];
percent_variance=[];

num_components=list(np.arange(1,X.shape[1]+1));
num_samples=X.shape[0];

#scores.append(-np.mean(cross_val_score(LinearRegression(),np.ones(num_samples,1),y, cv=10, scoring='neg_mean_squared_error')));
scores.append(-np.mean(cross_val_score(LinearRegression(), np.ones((num_samples,1)), Y,cv=10, scoring='neg_mean_squared_error')));

for n in num_components:
    pca=PCA(n_components=n);
    pipeline=Pipeline([('scaler', StandardScaler()),('pca',pca),('linear_regression',LinearRegression())]);
    pipeline.fit(X,Y);
    scores.append(-np.mean(cross_val_score(pipeline,X,Y,scoring='neg_mean_squared_error',cv=10)));
    percent_variance.append(np.sum(pca.explained_variance_ratio_));

print('Percent Variance Explained by number of Components:\n', pd.Series(percent_variance, index=range(1,20)));

fig1, ax=plt.subplots(figsize=(12,6));
ax.plot(scores,marker='o',color='b');
ax.set_xlabel('Number of Components');
ax.set_ylabel('10-Fold CV score');
ax.set_xlim(xmin=-1);
fig1.savefig('PCA_CV.jpeg');
#%%
np.random.seed(2);
train=np.random.choice([True, False],size=len(hitters));
X_train=X[train];
X_test=X[~train];
Y_train=Y[train];
Y_test=Y[~train];

num_samples=X_train.shape[0];
scores=[];

num_components=list(np.arange(1,X_train.shape[1]+1));
num_samples=X_train.shape[0];

scores.append(-np.mean(cross_val_score(LinearRegression(),np.ones((num_samples,1)), Y_train, cv=10, scoring='neg_mean_squared_error')));

for n in num_components:
    pca=PCA(n_components=n);
    pipeline=Pipeline([('scaler',StandardScaler()), ('pca', pca), ('linear_regression',LinearRegression())]);
    pipeline.fit(X_train, Y_train);
    scores.append(-np.mean(cross_val_score(pipeline, X_train, Y_train, scoring='neg_mean_squared_error', cv=10)));

fig2, ax=plt.subplots(figsize=(12,4));
ax.plot(scores, marker='o',color='b');
ax.set_xlabel('Number of Components');
ax.set_ylabel('Training 10-Fold CV Score');
ax.set_xlim(xmin=-1);
fig2.savefig('PCA_validation.jpeg');

#5 components
pca = PCA(n_components=5);
pipeline=Pipeline([('scaler', StandardScaler()),('pca',pca),('linear_regression',LinearRegression())]);
pipeline.fit(X_train,Y_train);
Y_predict=pipeline.predict(X_test);
MSE=np.mean((Y_predict-Y_test)**2);
print('5 Component PCA MSE:',MSE);

#%% partial least squares

np.random.seed(4);

train=np.random.choice([True, False], size=len(hitters));
X_train=X[train];
X_test=X[~train];
Y_train=Y[train];
Y_test=Y[~train];

scores=[];
num_components=list(np.arange(1,X_train.shape[1]+1));
num_samples=X_train.shape[0];

scores.append(-np.mean(cross_val_score(LinearRegression(),np.ones((num_samples,1)),Y_train,cv=10, scoring='neg_mean_squared_error')));

for n in num_components:
    pls=PLSRegression(n_components=n)
    pipe=Pipeline([('scaler',StandardScaler()),('pls',pls)]);

    pipe.fit(X_train,Y_train);
    scores.append(-np.mean(cross_val_score(pipe, X_train, Y_train, scoring='neg_mean_squared_error', cv=10)));

fig3, ax=plt.subplots(figsize=(12,8));
ax.plot(scores,marker='o',color='b');
ax.set_xlabel('Number of components');
ax.set_ylabel('Training 10-Fold CV score');
ax.set_xlim(xmin=-1);
fig3.savefig('PLS_CV.jpeg');

#for 3 components
pls=PLSRegression(n_components=3, scale=False);
pipe=Pipeline([('scaler',StandardScaler()),('pls', pls)]);
pipe.fit(X_train, Y_train);
Y_predict=pipe.predict(X_test);
MSE=np.mean((Y_predict-Y_test)**2);
print('PLS 3 component MSE: ', MSE);