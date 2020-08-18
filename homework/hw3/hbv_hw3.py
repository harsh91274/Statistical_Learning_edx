#applied problems ch4 - statistical learning by Harsh Vora
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy as stats
from statsmodels.stats.outliers_influence import OLSInfluence

def predict1(res,new):
    fit=pd.DataFrame(res.predict(new),columns=['fit']);
    ci=res.conf_int().rename(columns={0:'lower', 1:'upper'});
    ci=ci.T.dot(new.T).T;
    return pd.concat([fit, ci], axis=1);

#%%
advertising=pd.read_csv('../datasets/Advertising.csv');
print(advertising.head());

credit=pd.read_csv('../datasets/Credit.csv');
print(credit.head());

auto=pd.read_csv('../data/Auto.csv');
print(auto.head());

#%% 3.1 - least squares plot
plt.figure();
sns.regplot(advertising.TV, advertising.sales, order=1, ci=None, scatter_kws={'color':'r', 's':9});
plt.xlim(-10,310); plt.ylim(ymin=0);
plt.savefig('f1.jpeg');

#%% 3.4 and 3.6 - regression coefficients
est=smf.ols('sales~TV+radio+newspaper',data=advertising).fit();
print(est.summary());

#%% 3.7 - interaction b/w predictors
credit['Student2'] = credit.Student.map({'No':0, 'Yes':1});
est1=smf.ols('Balance~Income+Student2',data=credit).fit();
reg1=est1.params;
print(reg1);
est2=smf.ols('Balance~Income+Income*Student2',data=credit).fit();
reg2=est2.params;
print(reg2);

#%% 3.8
X=auto['horsepower'].reset_index(drop=True);
X=sm.add_constant(X); #add bias;
y=auto['mpg'].reset_index(drop=True);

res1=smf.ols('mpg~horsepower',auto);
results1=res1.fit();
print(results1.summary());

#%% LABWORK
print ('linear regression lab ch3');

boston=pd.read_csv('../data/Boston.csv');
print(boston.head());

#predictor is lstat and medv is response

lm1=sm.OLS.from_formula('medv~lstat',data=boston);
result1=lm1.fit();
print(result1.summary());

#new predictions
new1=pd.DataFrame([[1,5], [1,10], [1,15]], columns=['Intercept', 'lstat']);
result1.predict(new1);

print(predict1(result1, new1));

plt.figure();
sns.regplot('lstat','medv',boston, line_kws={'color':'r'}, ci=None);
plt.savefig('lab_single_regression.jpeg');

plt.figure();
fitted_values=pd.Series(result1.fittedvalues, name='fitted_values');
residuals=pd.Series(result1.resid, name='Residuals');
sns.regplot(fitted_values, residuals, fit_reg=False);
plt.savefig('lab_single_regression_residuals.jpeg');

plt.figure();
s_residuals=pd.Series(result1.resid_pearson,name='S. Residuals');
sns.regplot(fitted_values, s_residuals, fit_reg=False);
plt.savefig('lab_single_regression_sresiduals.jpeg');

#points with influence
plt.figure();
leverage=pd.Series(OLSInfluence(result1).influence, name='Leverage');
sns.regplot(leverage, s_residuals, fit_reg=False);
plt.savefig('lab_single_regression_influence.jpeg');

#MULTIPLE LINEAR REGRESSION

model=sm.OLS.from_formula('medv~lstat+age',boston);
result=model.fit();
print(result.summary());

#using all predictors
model2=sm.OLS.from_formula('medv~'+'+'.join(boston.columns.difference(['medv'])), boston);
result2=model2.fit();
print(result2.summary());

#interaction terms
print (sm.OLS.from_formula('medv~lstat*age',boston).fit().summary());

#non-linear transformations
result3=sm.OLS.from_formula('medv~lstat+np.square(lstat)',boston).fit();
print(result3.summary());

#ANOVA
result0=sm.OLS.from_formula('medv~lstat',boston).fit();
print(sm.stats.anova_lm(result0, result3));

plt.figure();
fitted_values3=pd.Series(result3.fittedvalues, name='fitted_values_quadratic');
s_residuals3=pd.Series(result3.resid_pearson, name='S.residuals_quadratic');
sns.regplot(fitted_values3, s_residuals3, fit_reg=False);
plt.savefig('linear_quadratic_sresidual.jpeg')

#%%
#qualitative predictors
df2=pd.read_csv('../data/Carseats.csv');
print(df2.head());

print(sm.OLS.from_formula('Sales ~ Income:Advertising+Price:Age + ' + "+".join(df2.columns.difference(['Sales'])), df2).fit().summary())

#Assignment Q10
# Pre-processing
# Convert quantitive datatypes to numerics
datatypes = {'quant': ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education'],
             'qual': ['ShelveLoc', 'Urban', 'US']}
# Use floats for all quantitive values
quants = df2[datatypes['quant']].astype(np.float_)
carseats_df = pd.concat([quants, df2[datatypes['qual']]], axis=1)

print(carseats_df.head());

model=sm.OLS.from_formula('Sales~Price+Urban+US', carseats_df).fit();
print(model.summary());

model2=sm.OLS.from_formula('Sales~Price+US',carseats_df).fit();
print(model2.summary());

#residuals
plt.figure();
fitted_values=pd.Series(model2.fittedvalues, name='fitted_values');
s_residuals=pd.Series(model2.resid_pearson, name='S_residuals');
sns.regplot(fitted_values, s_residuals, fit_reg=True);
plt.savefig('carseats_residual.jpeg');

conf_int_95=model2.conf_int(alpha=0.95);
print(conf_int_95);













