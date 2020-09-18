#statistical learning lab7 - nonlinear methods
#harsh vora 09172020
#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from patsy import bs, dmatrix
from matplotlib import pyplot as plt

plt.style.use('ggplot');
#%% polynomial regression
wages=pd.read_csv('../datasets/Wage.csv');
print(wages.head());

model = smf.ols('wage ~ age + I(age**2)+ I(age**3) + I(age**4)', data=wages);
estimate = model.fit()
print(estimate.summary());
#predictions
new_ages=np.linspace(wages.age.min(),wages.age.max(),num=1000);
predictions=estimate.predict(exog=dict(age=new_ages));

simpleTable, data, column_names=summary_table(estimate, alpha=0.05);
print('column names: ', column_names);
predicted_mean_ci_low, predicted_mean_ci_high=data[:,4:6].T;

fig1,ax=plt.subplots(1,1, figsize=(10,6));
ax.scatter(wages.age, wages.wage, facecolors='none', edgecolors='darkgray', label='data');
ax.plot(new_ages, predictions, 'b-', lw=2, label='Prediction');

order=np.argsort(wages.age.values);
ax.plot(wages.age.values[order], predicted_mean_ci_low[order], 'b--', label='95% CI');
ax.plot(wages.age.values[order], predicted_mean_ci_high[order], 'b--');
ax.legend(loc='best');
ax.set_xlabel('age');
ax.set_ylabel('wage');
fig1.savefig('age_vs_wage_4deg.jpeg');
plt.close(fig1);

#%% ANOVA to obtain polynomial degree
est1=smf.ols('wage~age', data=wages).fit();
est2=smf.ols('wage~age+I(age**2)', data=wages).fit();
est3 = smf.ols('wage ~ age + I(age**2)+ I(age**3)', data=wages).fit()
est4 = smf.ols('wage ~ age + I(age**2)+ I(age**3) + I(age**4)', data=wages).fit()
est5 = smf.ols('wage ~ age + I(age**2)+ I(age**3) + I(age**4) + I(age**5)', data=wages).fit()

anova = sm.stats.anova_lm(est1,est2,est3,est4,est5);
print(anova);

#p-value for 1st order to 2nd order is <2e-16 - thus, quadratic term is significant
#p-value for 2nd order to 3rd order is 1.6e-3 - thus, cubic term is significant
#p-value for 3rd order to 4th order is 5.1e-2 - thus, quadratic term is significant
#p-value for 4th order to 5th order is 3.69e-1 - thus, not significant
# 4th order polynomial is probably best, 3rd order polynomial might be sufficient

#%% logistic model for high earner prediction

wages['high_earner']=(wages.wage>250).astype(float);
model=smf.logit('high_earner~age+I(age**2)+I(age**3)+I(age**4)',data=wages);

estimate=model.fit();
#make OOS out of sample predictions

new_ages=np.linspace(wages.age.min(), wages.age.max(), num=1000);
predictions=estimate.predict(exog=dict(age=new_ages));

#get confidence intervals
std_err=np.array([]);
for age in new_ages:
    poly_age=np.array([[1, age, age**2, age**3, age**4]]);
    temp=np.dot(estimate.cov_params(), poly_age.T);
    #std_err=np.append(std_err, np.sqrt(np.dot(poly_age,temp));
    std_err=np.append(std_err, np.sqrt(np.dot(poly_age, temp)))

# compute critical value t_alpha/2(n-1) ~alpha=5%
crit_value = stats.t.isf(.05/2,len(wages)-1);
widths=crit_value*std_err;
linear_fit_vals=estimate.predict(exog=dict(age=new_ages),linear=True);

#confidence intervals
ui_linear=linear_fit_vals+widths;
li_linear=linear_fit_vals-widths;

ui=np.exp(ui_linear)/(1+np.exp(ui_linear));
li=np.exp(li_linear)/(1+np.exp(li_linear));

fig2, ax=plt.subplots(figsize=(12,8));
def rand_jitter(arr):
    stdev=0.01*(max(arr)-min(arr));
    return arr+np.random.randn(len(arr))*stdev;

ax.scatter(rand_jitter(wages.age.values),wages.high_earner.values/5, marker='|',color='darkgray',s=200);
ax.plot(new_ages,predictions, color='b', label='logit');
ax.plot(new_ages, ui, color='b', linestyle='--', label='95% CI');
ax.plot(new_ages, li, color='b', linestyle='--');
ax.set_ylim([-0.01, 0.21]);
ax.set_xlabel('Age');
ax.set_ylabel('P(Wage>250 | Age)');
plt.legend(loc='best', prop={'size':15});
fig2.savefig('wages_logistic.jpeg')
plt.close(fig2);
#%% regression splines using patsy

design = dmatrix("bs(age, knots=(25,40,60), degree=3, include_intercept=False)", data={"age":wages.age}, return_type="dataframe")
model=sm.GLS(endog=wages.wage, exog=design);
estimate=model.fit();

age_grid = np.linspace(wages.age.min(), wages.age.max(), num=1000)
prediction_design = dmatrix("bs(age_grid, knots=(25,40,60), degree=3, include_intercept=False)", {"age_grid": age_grid}, return_type='dataframe')
predictions = estimate.predict(prediction_design);

std_err = np.array([])
for row in prediction_design.values:
    # This row is a polynomial of the spline basis functions
    t = np.dot(estimate.cov_params(), row.T)
    # compute X.T*Sigma*X to get SE**2 then we can use this to get CIs
    std_err = np.append(std_err,np.sqrt(np.dot(row, t)))

# Compute the critical value t_alpha/2,n-1 ~ alpha = 5%
crit_value = stats.t.isf(.05/2,len(wages)-1);
# compute the confidence interval width
widths = crit_value*std_err;

# constuct upper and lower CIs
ui = predictions + widths;
li = predictions - widths;

#matrix with 6 degrees of freedom
design = dmatrix("bs(age, df=6, degree=3, include_intercept=False)", data={"age":wages.age});

model2 = sm.GLM(endog=wages.wage, exog=design);
estimate2 = model2.fit();

prediction2_design = dmatrix("bs(age_grid, df=6, degree=3, include_intercept=False)", {"age_grid": age_grid}, return_type='dataframe')
predictions2 = estimate2.predict(prediction2_design);

std_err2 = np.array([])
for row in prediction2_design.values:
    # This row is a polynomial of the spline basis functions
    t = np.dot(estimate2.cov_params(), row.T)
    # compute X.T*Sigma*X to get SE**2 then we can use this to get CIs
    std_err2 = np.append(std_err2,np.sqrt(np.dot(row, t)))

# Compute the critical value t_alpha/2,n-1 ~ alpha = 5%
crit_value2 = stats.t.isf(.05/2,len(wages)-1);
# compute the confidence interval width
widths2 = crit_value*std_err;

# constuct upper and lower CIs
ui2 = predictions2 + widths2;
li2 = predictions2 - widths2;

#Natural Spline DoF=4
design3 = dmatrix("cr(age, df=4)", data={"age":wages.age},return_type='dataframe');
model3 = sm.GLM(endog=wages.wage, exog=design3);
estimate3 = model3.fit();
prediction3_design = dmatrix("cr(age_grid, df=4)", {"age_grid": age_grid},return_type='dataframe');
predictions3 = estimate3.predict(prediction3_design);

std_err3 = np.array([]);
for row in prediction3_design.values:
    # This row is a polynomial of the spline basis functions
    t = np.dot(estimate3.cov_params(), row.T);
    # compute X.T*Sigma*X to get SE**2 then we can use this to get CIs
    std_err3 = np.append(std_err3,np.sqrt(np.dot(row, t)));

# Compute the critical value t_alpha/2,n-1 ~ alpha = 5%
crit_value3 = stats.t.isf(.05/2,len(wages)-1)
# compute the confidence interval width
widths3 = crit_value*std_err

# constuct upper and lower CIs
ui3 = predictions3 + widths3
li3 = predictions3 - widths3

fig3, ax = plt.subplots(1,1,figsize=(12,6))
# plot data
ax.scatter(wages.age, wages.wage,facecolors='none', edgecolors='darkgray', label="wage");

# plot the prediction when knots are specified
ax.plot(age_grid, predictions, 'b-', lw=2, label='Cubic Spline Knots Specified');
# add 95% CI
ax.plot(age_grid, ui, color='b', linestyle='--');
ax.plot(age_grid, li, color= 'b', linestyle='--');

# plot the prediction when the dof is specified
ax.plot(age_grid, predictions2, 'g-', lw=2, label='Cubic Spline DoF=6');
# add 95% CI
ax.plot(age_grid, ui2, color='g', linestyle='--');
ax.plot(age_grid, li2, color= 'g', linestyle='--');

# plot the prediction when the dof is specified and Natural Spline
ax.plot(age_grid, predictions3, 'r-', lw=2, label='Natural Spline DoF=4');
# add 95% CI
ax.plot(age_grid, ui3, color='r', linestyle='--');
ax.plot(age_grid, li3, color= 'r', linestyle='--');

# Labels
ax.set_xlabel('Age')
ax.set_ylabel('Wage');
plt.legend(loc='best');
fig3.savefig('wage_splines.jpeg')
plt.close(fig3);

#%% local regression
predictions20 = lowess(wages.wage, wages.age, frac=0.2, delta=0);
predictions50 = lowess(wages.wage, wages.age, frac=0.5, delta=0);

fig4, ax = plt.subplots(1,1,figsize=(12,6))
ax.scatter(wages.age, wages.wage,facecolors='none', edgecolors='darkgray', label="wage");
ax.plot(np.sort(wages.age), predictions20[:,1], 'r-', lw=2, label='Span=0.2')
ax.plot(np.sort(wages.age), predictions50[:,1], 'b-', lw=2, label='Span=0.5')

# Labels
ax.set_xlabel('Age')
ax.set_ylabel('Wage');
plt.legend(bbox_to_anchor=(1.3, 1.0));
fig4.savefig('wages_lowess.jpeg');
plt.close(fig4);

#%% part 5 - generalized additive models

design = dmatrix("cr(year, df=4, constraints='center') + cr(age, df=5, constraints='center') + education", data={"year":wages.year, "age":wages.age, "education":wages.education}, return_type="dataframe");

gam=sm.OLS(endog=wages.wage, exog=design).fit();
print(gam.summary());
print(gam.params[5:9]);
print(design.head(2));
year_basis=design[design.columns[5:9]];
print(year_basis.head());



