#statistical learning hw8 - hbv - tree based methods


import numpy as np
import pandas as pd
import graphviz

from sklearn.tree import tree, export_graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from IPython.display import Image, display
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence

plt.style.use('ggplot');

#%% Q1 - classification trees

df=pd.read_csv('../data/Carseats.csv');
print (df.head());
df['High']=df.Sales>8;
df2=pd.get_dummies(df,columns=['ShelveLoc', 'Urban', 'US']);
df3=df2.drop(['ShelveLoc_Bad', 'Urban_No', 'US_No'],axis=1)
print(df.head());

#classification tree
predictors=df3.columns.tolist();
predictors.remove('Sales');
predictors.remove('High');
X=df3[predictors].values;
y=df3.High.values.reshape(-1,1);

clf = tree.DecisionTreeClassifier(min_samples_split=20);
tree_est=clf.fit(X,y);

dot_data=export_graphviz(tree_est, out_file='seat_tree.dot', feature_names=predictors, class_names=['True', 'False'], filled=True, rounded=True, special_characters=True);

with open('seat_tree.dot') as f:
    dot_graph=f.read();

I=graphviz.Source(dot_graph, format='png', engine='dot');
Image(I.render());
I.view();

feature_importances=tree_est.feature_importances_;
fid=pd.DataFrame(data=feature_importances, index=predictors, columns=['Importance']).sort_values(by=['Importance'], ascending=False);
print(fid);

cmatrix=confusion_matrix(y_true=y, y_pred=tree_est.predict(X), labels=[True, False]);
print(cmatrix);

error_rate=(cmatrix[0,1]+cmatrix[1,0])/cmatrix.sum();
print('Training Error Rate: ', error_rate);

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=0);

tree_est=clf.fit(X_train, y_train);
ypred=tree_est.predict(X_test);
cmatrix_test=confusion_matrix(y_true=y_test, y_pred=ypred, labels=[True, False]);
print(cmatrix_test);
error_rate=(cmatrix_test[0,1]+cmatrix_test[1,0])/cmatrix_test.sum();
print('Test Error Rate: ', error_rate);

#%% Q2 - regression trees

boston=pd.read_csv('../data/Boston.csv', index_col=0);
print(boston.head());
X=boston[boston.columns[0:-1]];
y=boston['medv'].values;

X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=0.5, random_state=0);
#mse criterion, all features, split if more than 10 samples at a node
tree=DecisionTreeRegressor(criterion='mse', max_features=None, min_samples_split=2);
tree_est=tree.fit(X_train, y_train);

dot_data=export_graphviz(tree_est, out_file='boston_tree.dot', feature_names=boston.columns[0:-1], filled=True, rounded=True, special_characters=True);

with open('boston_tree.dot') as f:
    dot_graph2=f.read();

I=graphviz.Source(dot_graph, format='png', engine='dot');
#Image(I.render());

feature_importances=pd.Series(data=tree.feature_importances_, index=list(boston.columns[0:-1]));
fid=feature_importances.sort_values(axis=0, ascending=False);
print(fid);

ypred=tree_est.predict(X_test);

fig , ax=plt.subplots(1,1, figsize=(8,6));
ax.scatter(ypred, y_test, facecolor='None', edgecolor='b');
ax.plot([min(ypred),max(ypred)],[min(y_test), max(y_test)], linestyle='--',color='k');
ax.set_xlabel('y_predicted');
ax.set_ylabel('y_actual');
fig.savefig('boston_regression_tree.jpeg')

print('Test MSE: ', np.mean((ypred-y_test)**2));

#%% Q3 - Bagging and Random Forest
#bagged ensemble, with 500 trees (n=500) using all the predictors and bootstrap

bagger=RandomForestRegressor(n_estimators=500, criterion='mse', max_features='auto', bootstrap=True, oob_score=True, random_state=0);
bag_est=bagger.fit(X_train, y_train);

y_pred=bag_est.predict(X_test);

fig2, ax=plt.subplots(1,1,figsize=(8,6));
ax.scatter(y_pred, y_test, facecolor='none', edgecolor='r');
ax.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], linestyle='--',color='k');
ax.set_xlabel('y_predicted');
ax.set_ylabel('y_actual');
fig2.savefig('boston_bagged_ensemble.jpeg');

print('Bagged Ensemble Test MSE: ', np.mean((y_pred-y_test)**2));

#%% - Random Forest Prediction
# max features set at sqrt of number of features
forest=RandomForestRegressor(n_estimators=500, criterion='mse', max_features=6, bootstrap=True, oob_score=True, random_state=0);
forest_est=forest.fit(X_train, y_train);

y_pred=forest_est.predict(X_test);

fig3, ax=plt.subplots(1,1,figsize=(8,6));
ax.scatter(y_pred, y_test, facecolor='none', edgecolor='r');
ax.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], linestyle='--',color='k');
ax.set_xlabel('y_predicted');
ax.set_ylabel('y_actual');
fig3.savefig('boston_random_forest.jpeg');

print('Random Forest Test MSE: ', np.mean((y_pred-y_test)**2));

feature_importances=pd.Series(data=forest_est.feature_importances_, index=list(boston.columns[0:-1]));
fid=feature_importances.sort_values(axis=0, ascending=False);
print(fid);

fig4, ax1 =plt.subplots(1,1, figsize=(8,6));
feature_importances.plot(kind='barh', ax=ax1);
ax1.set_xlabel('Importance');
ax1.set_ylabel('Predictor');
fig4.savefig('boston_random_forest_feature_importance.jpeg')

#%% part 5 - boosted trees
# least squares fit, learning rate of 0.001, 5000 iterations, depth=4.
booster=GradientBoostingRegressor(loss='ls', learning_rate=0.001, n_estimators=5000, max_depth=4, random_state=0);
booster_est=booster.fit(X_train, y_train);

y_pred=booster_est.predict(X_test);

fig4, ax=plt.subplots(1,1,figsize=(8,6));
ax.scatter(y_pred, y_test, facecolor='none', edgecolor='r');
ax.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], linestyle='--',color='k');
ax.set_xlabel('y_predicted');
ax.set_ylabel('y_actual');
fig4.savefig('boston_boosted_trees.jpeg');

print('Boosted Trees Test MSE: ', np.mean((y_pred-y_test)**2));

feature_importances = pd.Series(data=booster_est.feature_importances_, index=list(boston.columns[0:-1]));
sorted_feature_importances = feature_importances.sort_values(axis=0, ascending=False);
print(sorted_feature_importances)

feature_idxs=np.argsort(feature_importances.values)[-3:];
fig5, axs = plt.subplots(1,1, figsize=(12,4));
plot_partial_dependence(booster_est, X_train, features=feature_idxs, feature_names=feature_importances.index.tolist(), ax=axs);
fig5.savefig('boston_partial_dependence_plot.jpeg')