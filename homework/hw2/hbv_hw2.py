# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:05:07 2020
Statistical Learning Assignment #1 and 2
@author: harshv
"""
#%% import libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#%%
def scale(df):
    """Xi = Xi - mu(Xi) / s(Xi)
    mu -> mean
    s -> standard deviation"""
    return (df-df.mean())/(df.std())

#%% college dataset

df_c=pd.read_csv('../datasets/College.csv');

#change college names to index
df_college=df_c.set_index(['Unnamed: 0'], append=True, verify_integrity=True);
df_college.rename_axis([None,'College'], inplace=True);
print(df_college.head()); #see first few rows
print(df_college.describe()); #statistical summary

plt.figure();
sns.pairplot(df_college.iloc[:,1:11]);
plt.savefig('college_pairplot.jpeg');

#side-by-side plots of outstate vs private
plt.figure()
sns.boxplot(x=df_college['Private'], y=df_college['Outstate']);
plt.savefig('box_sbs.jpeg');

# 8c iv. Create a new qualitative variable, called Elite, by binning the Top10perc variable. 

df_college['Elite']=df_college['Top10perc']>50;
plt.figure();
sns.boxplot(x=df_college['Elite'], y=df_college['Outstate']);
plt.savefig('box_sbs_elite_vs_outstate.jpeg');

#8c 5
feature_count = 12;
df_norm = scale(df_college.iloc[:, 1:feature_count+1]);
df_meltd = df_norm.melt(var_name='cols', value_name='vals')

# Plot grid of plots
plt.figure();
g = sns.FacetGrid(df_meltd, col='cols', col_wrap=4);
g.map(sns.distplot, 'vals');
g.set(xlim=(-4, 4));
plt.savefig('melt.jpeg');




