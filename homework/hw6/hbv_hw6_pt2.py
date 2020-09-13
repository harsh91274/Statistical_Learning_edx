#Statistical Learning Ch6 Lab
#Part 2 - lasso and ridge regression

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

plt.style.use('ggplot');

#%%
hitters=pd.read_csv('../datasets/Hitters.csv', index_col=0).dropna();
dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']]);
hitters = hitters.drop(['League', 'Division', 'NewLeague'], axis=1);
hitters = pd.concat([hitters, dummies[['League_N', 'Division_W', 'NewLeague_N']]],axis=1);
print(hitters.head());

X=hitters.drop('Salary', axis=1);
Y=hitters.Salary;

#ridge regression
regr=[];
alphas=10**np.linspace(-4,2,1000);

for index, alph in enumerate(alphas):
    model=Ridge(alpha=alph, normalize=True, fit_intercept=True);
    regr=np.append(regr,model.fit(X,Y));

print('sklearn alpha=', alphas[89],'\n');
df=pd.Series(data=np.hstack([regr[89].intercept_,regr[89].coef_]),index=['Intercept']+list(X.columns));
print(df,'\n');
print('L2 norm of Betas=', np.sqrt(sum(df.apply(lambda x: x**2).iloc[1:])));

ridge_coefs=np.empty((len(alphas),X.shape[1]));

for index, model in enumerate(regr):
    ridge_coefs[index]=model.coef_[np.newaxis];

f1, ax=plt.subplots(figsize=(16,10));
ax.plot(alphas, ridge_coefs,linewidth=2.0);
ax.set_xscale('log');
ax.set_xlabel('alpha (log-scale)', fontsize=14);
ax.set_ylabel('Ridge Coeffs', fontsize=14);
ax.legend(X.columns.tolist(),loc='best', bbox_to_anchor=(0.5, 1.4), ncol=6);
f1.savefig('ridge_regression_plot.jpeg');

np.random.seed(0);
train=np.random.choice([True, False], size=len(hitters));

X_train=X[train];
X_test=X[~train];
Y_train=Y[train];
Y_test=Y[~train];

#choosing a lambda=4,alpha=4/(2*N)=0.09

model=Ridge(alpha=0.09, normalize=True).fit(X_train, Y_train);
Y_predicted=model.predict(X_test);
MSE_validation=np.mean((Y_test-Y_predicted)**2);
print('MSE for alpha=0.09 (lambda=4)', MSE_validation);

print('MSE Of Intercept Only Model = ', np.mean((np.mean(Y[train])-Y_test)**2));
#for a large value of lambda
model = Ridge(alpha=1e10, normalize=True).fit(X_train,Y_train);
Y_predicted=model.predict(X_test);
MSE_validation=np.mean((Y_predicted-Y_test)**2);
print('Large Alpha MSE: ', MSE_validation);

#for least squares, alpha=lambda=0;
model=Ridge(alpha=0, normalize=True).fit(X_train, Y_train)
Y_predicted=model.predict(X_test);
MSE_validation=np.mean((Y_predicted-Y_test)**2);
print('Least Squares MSE: ', MSE_validation);

#%% k fold cross validation

alphas = 10**np.linspace(-4,2, 100);
kf=KFold(n_splits=10);
#kf=kf0.split(X);
cvs=[];

for alpha in alphas:
    error=[];
    for train, test in kf.split(X):
        X_train=X.values[train];
        Y_train=Y.values[train];
        X_test=X.values[test];
        Y_test=Y.values[test];

        model=Ridge(alpha=alpha, normalize=True).fit(X_train,Y_train);
        error=np.append(error, model.predict(X_test)-Y_test);

    cvs=np.append(cvs, np.mean(error**2));

min_index,min_cvs=min(enumerate(cvs), key=itemgetter(1));
print('Min Alpha, Min CV = ', alphas[min_index],min_cvs);

f2, ax= plt.subplots(figsize=(8,6));
ax.plot(alphas, cvs, color='b');
ax.plot(alphas[min_index],min_cvs, marker='x', color='r', markersize=15);

ax.set_xscale('log');
ax.set_ylabel('10 fold CV error', fontsize=14);
ax.set_xlabel('alpha (log-scale)',fontsize=14);
f2.savefig('ridge_k_fold.jpeg');

#%% lasso

grid=10**np.linspace(-4,2,1000);
np.random.seed(0);
train=np.random.choice([True, False], size=len(hitters));

X_train=X.values[train];
Y_train=Y.values[train];
X_test=X.values[~train];
Y_test=Y.values[~train];

coefficients=np.empty((len(grid),X.shape[1]));

for index, alpha in enumerate(grid):
    lasso=Lasso(alpha=alpha, normalize=True, max_iter=10000);
    # pipeline=Pipeline([('lasso'),lasso]);
    # pipleline.fit(X_train,Y_train);
    lasso.fit(X_train,Y_train)
    coefficients[index]=lasso.coef_[np.newaxis];


# Make a plot of the coeffecients
f3, ax = plt.subplots(figsize=(12,5))
ax.plot(grid, coefficients, linewidth =2.0);

ax.set_xscale('log')
ax.set_xlabel('alpha (log-scale)', fontsize=14)
ax.set_ylabel('Lasso Coeffecients', fontsize=14)
ax.legend(X.columns.tolist(),loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=6);
f3.savefig('lasso.jpeg');

scores = list();
np.random.seed(0);

# compute cross validation using MSE scoring
for alpha in grid:
    # for each alpha make a new model
    lasso = Lasso(alpha=alpha, normalize=True, max_iter=10000)

    # create our pipeline for later potential scaling of predictors
    pipeline = Pipeline([('lasso', lasso)])

    # get the cross-val score
    this_scores = -cross_val_score(pipeline, X, Y, scoring='neg_mean_squared_error', cv=10)
    scores.append(np.mean(this_scores));


min_index, min_score = min(enumerate(scores), key=itemgetter(1))
print('Min Alpha = ', grid[min_index])

# Plot the CV Errors as a funtion of alpha and plot minimum
f4,ax = plt.subplots(figsize=(8,6))

ax.plot(grid, scores, color='b')
ax.plot(grid[min_index], min_score, marker='x', color='r', markersize=15)

ax.set_xscale('log')
ax.set_ylabel('10-Fold CV Error', fontsize=14);
ax.set_xlabel('alpha (log-scale)', fontsize=14);
f4.savefig('lasso_cv.jpeg');

#%% using lassocv

np.random.seed(0)
lasso_cv = LassoCV(alphas=grid, normalize=True, max_iter=10000, cv=10)
lasso_cv.fit(X,Y)
print(lasso_cv.alpha_)

np.random.seed(0)

# split the data into test and validation sets
train = np.random.choice([True, False], size=len(hitters))

X_train = X[train]
Y_train = Y[train]
X_test = X[~train]
Y_test = Y[~train]

lasso = Lasso(alpha=grid[min_index], normalize=True, max_iter=10000)
lasso.fit(X_train, Y_train)

mse = np.mean((lasso.predict(X_test)-Y_test)**2)
df = pd.Series(data = np.hstack([lasso.intercept_, lasso.coef_]), index=['Intercept'] + X.columns.tolist())
print(df)

