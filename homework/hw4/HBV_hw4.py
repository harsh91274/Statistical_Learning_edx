#Statistical Learning
#Lab Ch4 - Logistic Regression, lda, qda, KNN

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from scipy import stats

plt.style.use('ggplot');
#%%
#Question 10

df=pd.read_csv('../datasets/Weekly.csv');
print(df.head());

#A - summaries of the data

correlations=df.corr(method='pearson');
print(correlations);

#strongest correlation is between year and volume

# B - logistic regression for direction as response and five lag variables + vol as predictors

fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(18,4));

ax1.scatter(df.Year.values, df.Volume.values, facecolors='None',edgecolors='b');
ax1.set_xlabel('year');
ax1.set_ylabel('volume in billions');

ax2.scatter(df.Lag1.values,df.Today.values,facecolors='none',edgecolors='b');
ax2.set_xlabel('lag1 percent return');
ax2.set_ylabel('todays percent return');

ax3.scatter(df.Lag2.values,df.Today.values,facecolors='none',edgecolors='b');
ax3.set_xlabel('lag2 percent return');
ax3.set_ylabel('todays percent return');

fig.savefig('q10_f1.jpeg');

#strong correlation between year and trading volume, and a weak relationship between %return and lag
#thus logistic regression may not perform too well
#C
predictors=df.columns[1:7];
X=sm.add_constant(df[predictors]); #noise

y=np.array([1 if el=='Up' else 0 for el in df.Direction.values]);
logit=sm.Logit(y,X);
results=logit.fit();
print(results.summary());

#only variable lag2 has a small p-value

y_predicted=results.predict(X);
y_predicted=np.array(y_predicted>0.5, dtype=float);

#confusion matrix
table=np.histogram2d(y_predicted,y, bins=2)[0];
print(pd.DataFrame(table,['Down_predicted','Up_predicted'],['Down','Up']));
print('\n')
print('Error Rate=',1-(table[0,0]+table[1,1])/np.sum(table));
print('Precision=',(table[1,1]/(table[1,0]+table[1,1])));
print('Type 1 error (False Positive)=',(table[1,0]/(table[1,0]+table[0,0])));
print('Type 2 error (False Negatives)=',(table[0,1]/(table[0,1]+table[1,1])))

#D - split data and re-examine using lag2 as the only predictor

X_train=sm.add_constant(df[df.Year<=2008].Lag2);
response_train=df[df.Year<=2008].Direction;

#convert responses to 0 and 1's
y_train=np.array([1 if el=='Up' else 0 for el in response_train]);

X_test=sm.add_constant(df[df.Year>2008].Lag2);
response_test=df[df.Year>2008].Direction;
y_test=np.array([1 if el=='Up' else 0 for el in response_test]);

#classifier and fit
logit=sm.Logit(y_train,X_train);
results=logit.fit();
print(results.summary());

#predicting test responses
y_predicted=results.predict(X_test);
y_predicted=np.array(y_predicted>0.5, dtype='float');

#confustion matrix
table=np.histogram2d(y_predicted,y_test,bins=2)[0];
print('Confusion Matrix');
print(pd.DataFrame(table, ['Down_predicted','Up_predicted'],['Down','Up']));
print('\n')
print('Error Rate=',1-(table[0,0]+table[1,1])/np.sum(table));
print('Precision=',(table[1,1]/(table[1,0]+table[1,1])));
print('Type 1 error (False Positive)=',(table[1,0]/(table[1,0]+table[0,0])));
print('Type 2 error (False Negatives)=',(table[0,1]/(table[0,1]+table[1,1])))

#E - LDA

clf=LDA(solver='lsqr',store_covariance=True);
X_train=df[df.Year<=2008].Lag2.values;
X_train = X_train.reshape((len(X_train),1))

X_test=df[df.Year>2008].Lag2.values;
X_test = X_test.reshape((len(X_test),1))

clf.fit(X_train, y_train);
print('priors=',clf.priors);
print('class means=',clf.means_[0],clf.means_[1]);
print('coefficients=',clf.coef_);
print('\n');
#predict test respones
y_predicted=clf.predict(X_test);
y_predicted=np.array(y_predicted>0.5, dtype='float');

#confustion matrix
table=np.histogram2d(y_predicted,y_test,bins=2)[0];
print('LDA Confusion Matrix');
print(pd.DataFrame(table, ['Down_predicted','Up_predicted'],['Down','Up']));
print('\n')
print('Error Rate=',1-(table[0,0]+table[1,1])/np.sum(table));
print('Precision=',(table[1,1]/(table[1,0]+table[1,1])));
print('Type 1 error (False Positive)=',(table[1,0]/(table[1,0]+table[0,0])));
print('Type 2 error (False Negatives)=',(table[0,1]/(table[0,1]+table[1,1])))


#F - Quadratic discriminant analysis
qclf= QDA (store_covariance=True);
qclf.fit(X_train,y_train);

print('priors=',qclf.priors);
print('class means=',qclf.means_[0],clf.means_[1]);
print('coefficients=',qclf.covariance_);
print('\n');

y_predict=qclf.predict(X_test);
y_predicted=np.array(y_predict>0.5,dtype='float');

#confustion matrix
table=np.histogram2d(y_predicted,y_test,bins=2)[0];
print('QDA Confusion Matrix');
print(pd.DataFrame(table, ['Down_predicted','Up_predicted'],['Down','Up']));
print('\n')
print('Error Rate=',1-(table[0,0]+table[1,1])/np.sum(table));
print('Precision=',(table[1,1]/(table[1,0]+table[1,1])));
print('Type 1 error (False Positive)=',(table[1,0]/(table[1,0]+table[0,0])));
print('Type 2 error (False Negatives)=',(table[0,1]/(table[0,1]+table[1,1])))

#KNN logistic regression

clf=KNeighborsClassifier(n_neighbors=1);
clf.fit(X_train,y_train);
y_predicted=clf.predict(X_test);
#y_predicted=np.array(y_predict>0.5,dtype='float');

#confustion matrix
table=np.histogram2d(y_predicted,y_test,bins=2)[0];
print('QDA Confusion Matrix');
print(pd.DataFrame(table, ['Down_predicted','Up_predicted'],['Down','Up']));
print('\n')
print('Error Rate=',1-(table[0,0]+table[1,1])/np.sum(table));
print('Precision=',(table[1,1]/(table[1,0]+table[1,1])));
print('Type 1 error (False Positive)=',(table[1,0]/(table[1,0]+table[0,0])));
print('Type 2 error (False Negatives)=',(table[0,1]/(table[0,1]+table[1,1])))


# Build KNN Classifier and Fit #
################################
clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(X_train, y_train)

# Predict Test Set Responses #
##############################
y_predicted = clf.predict(X_test)

table = np.histogram2d(y_predicted, y_test , bins=2)[0]
print(pd.DataFrame(table, ['Down', 'Up'], ['Down', 'Up']))
print('')
print('Error Rate =', 1-(table[0,0]+table[1,1])/np.sum(table))