#Statistical Learning Ch6 Lab
#Part 1 - subset selection - best subset, forward and backward stepsize, validation and CV
#Part 2 -

#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
import itertools
from tqdm import tqdm #progress bars for iterables
from operator import itemgetter
from itertools import combinations
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split

plt.style.use('ggplot');
np.set_printoptions(precision=4);
#hitters dataset

#%%
hitters=pd.read_csv('../datasets/Hitters.csv');

# print('no of players: ', hitters.shape[0]);
# num_missing=np.sum(hitters.isnull().any(axis=1));
# print('missing data for ', num_missing, ' players');

# #remove missing players from dataframe

# hitters=hitters.dropna();
# print('no of players after drop: ', hitters.shape[0]);
# print(hitters.head());

# #create dummy variables for categoricals
# dummies=pd.get_dummies(hitters[['League','Division','NewLeague']]);
# print(dummies.head());

# y=hitters.Salary;
# df=hitters.drop(['Unnamed: 0','Salary','League','Division','NewLeague'],axis=1);
# df=pd.concat([df, dummies[['League_N','Division_W','NewLeague_N']]],axis=1);
# df.head();

# X=df;

#%%
#best subset selection
#this forms every possible model from the predictors,
#compares models with same number of predictors using RSS
#once the best mmodel is picked for a number of predictors,
#models with different number of predictors are compared using AIC, BIC or adjR2
#or by CV to select the model with lowest test MSE
#this method has a high computational cost, 2^p models
#hitters = pd.read_csv('../../data/Hitters.csv',index_col=0)

# Get the number of players  and the number of players with missing values
print('Hitters contains', len(hitters), 'players.')
num_missing = np.sum(hitters.isnull().any(axis=1))
print('We are missing data for', num_missing, 'players.')

# now remove the missing players for dataframe
hitters = hitters.dropna()
print('After removal Hitters contains', len(hitters), 'players.')
print('Shape=', hitters.shape)
hitters.head()
dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']])
dummies.head(2)
df = hitters.drop(['Unnamed: 0','League', 'Division', 'NewLeague'], axis=1)
# add new dummy variables
df = pd.concat([df, dummies[['League_N', 'Division_W', 'NewLeague_N']]],axis=1)
df.head(2)

def best_subsets(dataframe, predictors, response, max_features=8):

    def process_linear_model(features):
        """
        Constructs Linear Model Regression of response onto features.
        """
        # Create design Matrix
        X = sm.add_constant(dataframe[features])
        y = dataframe[response]

        model = sm.OLS(y,X.astype(float)).fit()
        RSS = model.ssr
        return (model, RSS)

    def get_best_kth_model(k):
        """
        Returns the model from all models with k-predictors with the lowest RSS.
        """
        results = []

        for combo in combinations(predictors, k):
            # process linear model with this combo of features
            results.append(process_linear_model(list(combo)))

        # sort the models and return the one with the smallest RSS
        return sorted(results, key= itemgetter(1)).pop(0)[0]

    models =[]
    for k in tqdm(range(1,max_features+1)):
        models.append(get_best_kth_model(k))

    return models

predictors = list(df.columns)
predictors.remove('Salary')
models = best_subsets(df, predictors, ['Salary'], max_features=19)
print(models[1].params);
#%%
# def fit_ols(X,y):
#     ols = LinearRegression()
#     ols_model =ols.fit(X,y)
#     RSS= round(mean_squared_error(y,ols_model.predict(X)) * len(y),2)
#     R_square = round(ols_model.score(X,y),2)
#     return RSS, R_square;

# k=5;
# RSS_list, R_squared_list, feature_list = [],[], []
# numb_features = []

# #Looping over k = 1 to k = 11 features in X
# for k in range(1,len(X.columns) + 1):
#     #Looping over all possible combinations: from 11 choose k
#     for combo in itertools.combinations(X.columns,k):
#         tmp_result = fit_ols(X[list(combo)],y)   #Store temp result b
#         RSS_list.append(tmp_result[0])                  #Append lists
#         R_squared_list.append(tmp_result[1])
#         feature_list.append(combo)
#         numb_features.append(len(combo));

# df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list});
# df['min_RSS'] = df.groupby('numb_features')['RSS'].transform(min)
# df['max_R_squared'] = df.groupby('numb_features')['R_squared'].transform(max)
# #%%
# f1=plt.figure(figsize=(16,6));
# ax=f1.add_subplot(1,2,1);

# ax.scatter(df.numb_features,df.RSS,alpha=0.2,color='darkblue');
# ax.set_xlabel('#Features');
# ax.set_ylabel('RSS');
# ax.set_title('RSS-Best subset selection');
# ax.plot(df.numb_features,df.min_RSS,color='r',label='best subset');
# ax.legend();

# ax=f1.add.subplot(1,2,2);
# ax.scatter(df.numb_features,df.R_squared,alpha=0.2,color='darkblue');
# ax.plot(df.numb_features,df.max_R_squared,color='r',label='best subset');
# ax.set_xlabel('#features');
# ax.set_ylabel*('R squared');
# ax.set_title('R_squared-Best subset selection');
# ax.legend();

# plt.show();
# f1.savefig('best_subset_selection.jpeg');
# plt.close();
#%%
#sequential forward stepwise selection

# ols = LinearRegression();

# sfs1 = sfs(ols, k_features=1, forward=True, floating=False, verbose=2, scoring='neg_mean_squared_error',cv=5)
# sfs1.fit(X, y);

# sfs2 = sfs(ols,k_features=2,forward=True,floating=False,verbose=2,scoring='neg_mean_squared_error',cv=5)
# sfs2.fit(X, y)

# sfs3=sfs(ols,k_features=3, forward=True, floating=False, verbose=2, scoring='neg_mean_squared_error', cv=5);
# sfs3.fit(X,y);

# sfs4 = sfs(ols, k_features =4, forward =True, floating =False, verbose= 2, scoring = 'neg_mean_squared_error', cv=5)
# sfs4.fit(X,y);

# dat= [['one variable', sfs1.k_feature_names_],['two variables', sfs2.k_feature_names_], ['three variables', sfs3.k_feature_names_], ['four variables', sfs4.k_feature_names_]]
# dat_df= pd.DataFrame(dat, columns=['Number of Variables','Forward stepwise Selection']);
# print(dat_df);
#%%
# create plots of these statistics to find the best model for baseball player salary.
aics = [models[x].aic for x in range(len(models))]
bics = [models[x].bic for x in range(len(models))]
r_adj = [models[x].rsquared_adj for x in range(len(models))]

# find the mins/maxes
min_aic_index, min_aic = min(enumerate(aics), key=itemgetter(1))
min_bic_index, min_bic = min(enumerate(bics), key=itemgetter(1))
max_radj_index, max_radj = max(enumerate(r_adj), key=itemgetter(1))

num_predictors = np.linspace(1,len(models),len(models))
# Create a plot
fig1,(ax1, ax2) = plt.subplots(1,2,figsize=(16,4))
# Add test MSE estimates
ax1.plot(num_predictors, aics, 'r', marker='o', label='AIC');
ax1.plot(num_predictors, bics, 'b', marker='o', label='BIC')

# add the minimums to the axis
ax1.plot(min_aic_index+1, min_aic, 'gx', markersize=20, markeredgewidth=1)
ax1.plot(min_bic_index+1, min_bic, 'gx', markersize=20, markeredgewidth=1)

# Labels and Legend
ax1.set_xlabel('Number of Predictors');
ax1.set_ylabel('Test MSE');
ax1.legend(loc='best');

# Add Adj R**2
ax2.plot(num_predictors, r_adj,'k', marker='o')
ax2.plot(max_radj_index+1, max_radj, 'gx', markersize=20, markeredgewidth=1)
ax2.set_xlabel('Number of Predictors');
ax2.set_ylabel('Adjusted R**2');

fig1.savefig('best_subset.jpeg');

print(models[5].params);

#%%
def forward_step_select(df, predictors, response, max_features=len(predictors)):

    def process_linear_model(features):
        """
        Constructs Linear Model Regression of response onto features.
        """
        # Create design Matrix
        X = sm.add_constant(df[features])
        y = df[response]

        model = sm.OLS(y,X.astype(float)).fit()
        RSS = model.ssr
        return (model, RSS)

    def update_model(best_features, remaining_features):
        """
        Computes the RSS of possible new models and returns the model with the lowest RSS.
        """
        results = []

        for feature in remaining_features:
            results.append(process_linear_model(best_features + [feature]))

        # select model with the lowest RSS
        new_model = sorted(results, key= itemgetter(1)).pop(0)[0]
        new_features = list(new_model.params.index)[1:]

        return new_features, new_model

    # Create list to hold models, model features and the remaining features to test
    models = []
    best_features = []
    remaining_features = predictors

    while remaining_features and len(best_features) < max_features:

        # get the best new feature set from update_model
        new_features, new_model = update_model(best_features, remaining_features)
        # update the best features to include the one we just found
        best_features = new_features
        # reduce the available features for the next round
        remaining_features =  [feature for feature in predictors if feature not in best_features]

        # append the new_features and model so we can compare models with different features later
        models.append((new_features,new_model))

    return models

predictors = list(df.columns)
predictors.remove('Salary')
# call forward_step_select
mods = forward_step_select(df,predictors,['Salary'],max_features=19)

print(mods[6][1].params);

#%% test, train
np.random.seed(0)
index = np.random.choice([True, False], size=len(df));
df_train = df[index];
df_test = df[~index];

predictors = list(df_train.columns);
predictors.remove('Salary');
models = best_subsets(df_train, predictors, ['Salary'], max_features=19);

mses = np.array([])
for model in models:
    # get the predictors for this model, ignore constant
    features = list(model.params.index[1:])

    # get the corresponding columns of df_test
    X_test = sm.add_constant(df_test[features])

    # make prediction for this model
    salary_pred = model.predict(X_test)

    # get the MSE for this model
    mses = np.append(mses, np.mean((salary_pred - df_test.Salary.values)**2))
print('MSEs =', mses)

min_index, min_mse = min(enumerate(mses), key=itemgetter(1));
print(min_index, min_mse);

min_index, min_mse = min(enumerate(mses), key=itemgetter(1))
print(min_index, min_mse)

print('10 Variable Model:', list(models[9].params.index));
print('14-Variable Model:', list(models[12].params.index));

num_predictors = np.linspace(1,len(models),len(models))

fig2, ax1 = plt.subplots(figsize=(8,4));
# add the mse and mimimum mse to the plot
ax1.plot(num_predictors, mses, 'r', marker='o', label='MSE')
ax1.plot(min_index+1, min_mse, 'gx', markersize=20, markeredgewidth=2)

# Labels and Legend
ax1.set_xlabel('Number of Predictors');
ax1.set_ylabel('Validation MSE');
ax1.legend(loc='best');
fig2.savefig('MSE_vs_predictors.jpeg');

predictors = list(df.columns)
predictors.remove('Salary')
models = best_subsets(df, predictors, ['Salary'], max_features=19)

# Print out the Coeffecients of the 14 predictor model determined as Best by Validation approach above.
print(models[13].params);

# Create the 10 folds using sklearn KFolds
kf = KFold(len(df),n_folds=10, random_state=1)

mses = np.zeros([10, len(predictors)])

for fold, (train, test) in enumerate(kf):

    # split data for this fold
    df_train = df.iloc[train]
    df_test = df.iloc[test]

    # compute the best model subsets using our function
    models = best_subsets(df_train, predictors, ['Salary'], max_features=19)

    # compute the MSE of each model
    for idx, model in enumerate(models):
        # get the predictors for this model, ignore constant
        features = list(model.params.index[1:])

        # get the corresponding columns of df_test
        X_test = sm.add_constant(df_test[features])

        # make prediction for this model
        salary_pred = model.predict(X_test)

        # get the MSE for this model and fold
        mses[fold, idx] = np.mean((salary_pred - df_test.Salary.values)**2)

# now we can compute the mean MSE across folds, one per model with idx features
cvs = np.mean(mses, axis=0)

# set predictors for x-axis
num_predictors = np.linspace(1,len(models),len(models))

fig3, ax1 = plt.subplots(figsize=(8,4));
# get the minimum in the CV
min_index, min_CV = min(enumerate(cvs), key=itemgetter(1))

# add the mse and mimimum mse to the plot
ax1.plot(num_predictors, cvs, 'b', marker='o', label='Test MSE')
ax1.plot(min_index+1, min_CV, 'rx', markersize=20, markeredgewidth=2)

# Labels and Legend
ax1.set_xlabel('Number of Predictors');
ax1.set_ylabel('CV Error');
ax1.legend(loc='best');
fig3.savefig('CVError_vs_predictors.jpeg');

models = best_subsets(df, predictors, ['Salary'], max_features=19);
print(models[10].params);
