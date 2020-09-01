#Statistical Learning Lab 5 - Resampling by HBV

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
import sklearn
#import scikits.bootstrap as bootstrap

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#from sklearn import cross_validation
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score

plt.style.use('ggplot');
np.set_printoptions(precision=4);
#%%
#auto dataset
auto=pd.read_csv('../data/auto.csv',na_values='?');
auto=auto.dropna();
print(len(auto), ' rows');
print(auto.head());

#portfolio dataset
portfolio=pd.read_csv('../datasets/Portfolio.csv');
print('\n',len(portfolio), ' rows');
print(portfolio.head());

#%% CROSS VALIDATION
#predict mpg from horesepower using with quadratic++ fit

np.random.seed(0);

#split data, make boolean mask
training=np.random.choice([False, True], size=auto.shape[0]);

y_train=auto.mpg[training];
x_train=sm.add_constant(auto.horsepower[training]);
#validation set
y_test=auto.mpg[~training];
x_test=sm.add_constant(auto.horsepower[~training]);

#linear model
linear_model=sm.OLS(y_train,x_train);
linear_results=linear_model.fit();

y_predictions=linear_results.predict(x_test);
print('Linear Model MSE=',np.mean((y_test.values-y_predictions)**2));

#quadratic model
x_train2=sm.add_constant(np.column_stack((auto.horsepower[training],auto.horsepower[training]**2)));
x_test2=sm.add_constant(np.column_stack((auto.horsepower[~training],auto.horsepower[~training]**2)));

quad_model=sm.OLS(y_train,x_train2);
quad_model_results=quad_model.fit();
y_predictions2=quad_model_results.predict(x_test2);
print('Quadratic Model MSE=',np.mean((y_test.values-y_predictions2)**2));

#cubic model
x_train3=sm.add_constant(np.column_stack((auto.horsepower[training],auto.horsepower[training]**2,auto.horsepower[training]**3)));
x_test3=sm.add_constant(np.column_stack((auto.horsepower[~training],auto.horsepower[~training]**2,auto.horsepower[~training]**3)));

cubic_model=sm.OLS(y_train,x_train3);
cubic_model_results=cubic_model.fit();
y_predictions3=cubic_model_results.predict(x_test3);
print('Cubic Model MSE=', np.mean((y_test.values-y_predictions3)**2));

#%% LOOCV

X=auto.horsepower.values;
X=X[:,np.newaxis];
y=auto.mpg.values;
#LOOCV for 5 polynomial models
orders=np.arange(1,6);
mse_est=np.array([])

for index,order in enumerate(orders):
    print(order);
    poly=PolynomialFeatures(degree=order, interaction_only=False, include_bias=False);
    regress=LinearRegression();
    regress.fit(poly.fit_transform(X),y);
    print('Coefficients: Intercept, Beta(s)',regress.intercept_, regress.coef_);

    #evaluate using LOOCV
    mse_est=np.append(mse_est,-np.mean(cross_val_score(regress,poly.fit_transform(X),y,scoring='neg_mean_squared_error',cv=len(X))));
    print('Estimated test MSE:', mse_est);

f1=plt.figure();
fig, ax= plt.subplots(figsize=(6,4));
ax.plot(orders, mse_est, linestyle='-.',marker='o',color='k');
ax.set_xlabel('Polynomial Order');
ax.set_ylabel('LOOCV Error Rate');
plt.savefig('LOOCV_Error_rate.jpeg');
plt.close();

#%% k-Fold Cross Validation
#models of order 1-5, fit and perform k-foldcv

#k-fold CV for 5 polynomial models
orders_k=np.arange(1,6);
mse_est_k=np.array([])

for index,order in enumerate(orders_k):
    print(order);
    poly_k=PolynomialFeatures(degree=order, interaction_only=False, include_bias=False);
    regress_k=LinearRegression();
    regress_k.fit(poly_k.fit_transform(X),y);
    print('Coefficients: Intercept, Beta(s)',regress_k.intercept_, regress_k.coef_);

    #evaluate using kfold CV
    mse_est_k=np.append(mse_est_k,-np.mean(cross_val_score(regress_k,poly_k.fit_transform(X),y,scoring='neg_mean_squared_error',cv=10)));
    print('Estimated test MSE:', mse_est_k);

f2=plt.figure();
fig, ax= plt.subplots(figsize=(6,4));
ax.plot(orders_k, mse_est_k, linestyle='-.',marker='o',color='k');
ax.set_xlabel('Polynomial Order');
ax.set_ylabel('10 fold CV Error Rate');
plt.savefig('10_fold_CV_Error_rate.jpeg')
plt.close();

#%% The Bootstrap

def alpha(df, num_samples=100):
    """Returns the alpha statistic for num_sampels of elements from dataframe df"""
    indices=np.random.choice(df.index,num_samples,replace=True);
    X=df.X[indices].values;
    Y=df.Y[indices].values;

    return (np.var(Y)-np.cov(X,Y)[0][1])/(np.var(X)+np.var(Y)-2*np.cov(X,Y)[0][1]);


def auto_coeffs(data, num_samples=auto.shape[0]):
    indices=np.random.choice(data.index,num_samples,replace=True);
    X=sm.add_constant(data.horsepower[indices]);
    Y=data.mpg;

    results=sm.OLS(Y,X).fit();
    return results.params;

#%%
np.random.seed(0);
print(alpha(portfolio));

X=sm.add_constant(auto.horsepower);
Y=auto.mpg;

model_bs=sm.OLS(y,X);
results_bs=model_bs.fit();
print(results_bs.summary());
#auto_coeffs(auto);








