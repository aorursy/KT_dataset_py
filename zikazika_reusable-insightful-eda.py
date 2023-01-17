import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import datetime

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import scipy


train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
sns.pairplot(train_df.iloc[:,197:])
sns.pairplot(test_df.iloc[:,198:])
sns.pairplot(train_df.iloc[:,:5])
pd.isna(train_df).sum().sum()
sns.pairplot(pd.isna(train_df.iloc[:,198:]))
# Before excluding certain values in handle_outliers function underneath, we are going to compare two methods and different paramaeters

# All to see which number of outliers seems reasonable, than we are going to exclude entire row that has this outlier

#It will be only a few since we will opt for the most extreme case, where deviation from the mean is really ridiculous.



def out_std(s, nstd=3.0, return_thresholds=False):



    data_mean, data_std = s.mean(), s.std()

    cut_off = data_std * nstd

    lower, upper = data_mean - cut_off, data_mean + cut_off

    if return_thresholds:

        return lower, upper

    else:

        return [False if x < lower or x > upper else True for x in s]

    



    

    

std2 = train_df.iloc[:,198:].apply(out_std, nstd=2.0)

std3 = train_df.iloc[:,198:].apply(out_std, nstd=3.0)

std4 = train_df.iloc[:,198:].apply(out_std, nstd=4.0)



    

    

f, ((ax1, ax2, ax3)) = plt.subplots(ncols=3, nrows=1, figsize=(22, 12));

ax1.set_title('Outliers with 2 standard deviations');

ax2.set_title('Outliers using 3 standard deviations');

ax3.set_title('Outliers using 4 standard deviations');



sns.heatmap(std2, cmap='Blues', ax=ax1);

sns.heatmap(std3, cmap='Blues', ax=ax2);

sns.heatmap(std4, cmap='Blues', ax=ax3);





plt.show()
melted = pd.melt(train_df.iloc[:,194:])

melted["value"] = pd.to_numeric(melted["value"])

sns_plot1=sns.boxplot(x="variable", y="value", data=melted)

sns_plot1.set_xticklabels(sns_plot1.get_xticklabels(), rotation = 90, fontsize = 10)
corr = train_df.iloc[:,190:].corr()



# plot the heatmap

sns_plot2=sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(5,10,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(5,10,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();
t0 = train_df.loc[train_df['target'] == 0]

t1 = train_df.loc[train_df['target'] == 1]

features = train_df.columns.values[2:52]

plot_feature_distribution(t0, t1, '0', '1', features)
features = train_df.columns.values[2:52]

plot_feature_distribution(train_df, test_df, 'train', 'test', features)