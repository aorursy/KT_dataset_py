# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import statistics

import scipy.stats as stats

import statsmodels.api as sm

from statsmodels.formula.api import logit

from scipy.stats import chi2_contingency

from scipy.stats import kurtosis 

from scipy.stats import skew

from statistics import stdev 

import seaborn as sns

from tqdm import tqdm

from PIL import Image

import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')



train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

# Look at dimension of data set and types of each attribute

train.info()
test.info()
# Summarize attribute distributions of the data frame

train.describe(include='all')
test.describe(include='all')
# Take a peek at the first rows of the data

train.head(10)
test.head(10)
# Check missing values both to numeric features and categorical features

train.isnull().sum()/train.shape[0]*100
test.isnull().sum()/test.shape[0]*100
# Data Imputation

# Input missing values with median or mode depending of features class

train['sex'].fillna(train['sex'].mode()[0], inplace=True)

train['age_approx'].fillna(train['age_approx'].median(), inplace=True)

train['anatom_site_general_challenge'].fillna(train['anatom_site_general_challenge'].mode()[0], inplace=True)

test['anatom_site_general_challenge'].fillna(test['anatom_site_general_challenge'].mode()[0], inplace=True)



# Summarize the class distribution 

count = pd.crosstab(index = train['target'], columns="count")

percentage = pd.crosstab(index = train['target'], columns="frequency")/pd.crosstab(index = train['target'], columns="frequency").sum()

pd.concat([count, percentage], axis=1)
# Plot the target variable

ax = sns.countplot(x=train['target'], data=train, order=[0,1]).set_title("Target Variable Distribution")
# Univariate analysis looking at Standard Deviation, Skewness and Kurtosis for train set



print('\nStandard Deviation :', stdev(train['age_approx']), 

      '\nSkewness :', skew(train['age_approx']), 

        '\nKurtosis :', kurtosis(train['age_approx']))
# Univariate analysis looking at Standard Deviation, Skewness and Kurtosis for test set



print('\nStandard Deviation :', stdev(test['age_approx']), 

      '\nSkewness :', skew(test['age_approx']), 

        '\nKurtosis :', kurtosis(test['age_approx']))
# graphical function for univariate analysis



def num_plot(dataframe, feature):

    plt.figure(figsize=(15, 5))



    # histogram

    plt.subplot(1, 3, 1)

    sns.distplot(train[feature], bins=30, color='g')

    plt.title('Histogram')

    # Q-Q plot

    plt.subplot(1, 3, 2)

    stats.probplot(train[feature], dist="norm", plot=plt)

    plt.ylabel('Variable quantiles')

    # boxplot

    plt.subplot(1, 3, 3)

    x=train[feature]

    sns.boxplot(x,linewidth=1.5, color='g')

    plt.title('Boxplot')



    plt.show()
# train

# age_approx

num_plot(train, 'age_approx')
# test

# age_approx

num_plot(test, 'age_approx')
# Feature selection with Kendall's Test



alpha = 0.05

var = 'age_approx'

p = stats.kendalltau(train['target'],train[var])[1]

if p <= alpha:

    print('{0} Dependent (reject H0)'.format(var))

else:

    print('{0} Independent (fail to reject H0)'.format(var))
# Univariate analysis with frequency and barplots for train set

sns.set( rc = {'figure.figsize': (5, 5)})

fcat_tr = ['sex','anatom_site_general_challenge','diagnosis','benign_malignant']



for col in fcat_tr:

    count = pd.crosstab(index = train[col], columns="count")

    percentage = pd.crosstab(index = train[col], columns="frequency")/pd.crosstab(index = train[col], columns="frequency").sum()

    tab = pd.concat([count, percentage], axis=1)

    plt.figure()

    sns.countplot(x=train[col], data=train, palette="Set1")

    plt.xticks(rotation=45)

    print(tab)

    plt.show()
# Univariate analysis with frequency and barplots for test set

sns.set( rc = {'figure.figsize': (5, 5)})

fcat_te = ['sex','anatom_site_general_challenge']



for col in fcat_te:

    count = pd.crosstab(index = test[col], columns="count")

    percentage = pd.crosstab(index = test[col], columns="frequency")/pd.crosstab(index = test[col], columns="frequency").sum()

    tab = pd.concat([count, percentage], axis=1)

    plt.figure()

    sns.countplot(x=test[col], data=test, palette="Set1")

    plt.xticks(rotation=45)

    print(tab)

    plt.show()
# Bivariate analysis with barplots for train set

sns.set( rc = {'figure.figsize': (5, 5)})



for col in fcat_tr:

    plt.figure()

    sns.countplot(x=train[col], hue=train['target'], data=train, palette="Set2")

    plt.xticks(rotation=45)

    plt.show()
# Feature Selection with Chi-Square Test 

alpha = 0.05

for var in fcat_tr:

    X = train[var].astype(str)

    Y = train['target'].astype(str)

    dfObserved = pd.crosstab(Y,X)

    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)

    if p <= alpha:

    	print('{0} Dependent (reject H0)'.format(var))

    else:

        print('{0} Independent (fail to reject H0)'.format(var))
img = []

for i, image_id in enumerate(tqdm(train['image_name'].head(20))):

    im = Image.open(f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg')

    im = im.resize((128, )*2)

    img.append(im)
img[0]
img[5]
img[10]
img[15]
benign = train[train['benign_malignant']=='benign']

malign = train[train['benign_malignant']=='malignant']
img_b = []

for i, image_id in enumerate(tqdm(benign['image_name'].tail(20))):

    im = Image.open(f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg')

    im = im.resize((128, )*2)

    img_b.append(im)
img_b[1]
img_b[6]
img_b[11]
img_b[16]
img_m = []

for i, image_id in enumerate(tqdm(malign['image_name'].head(20))):

    im = Image.open(f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg')

    im = im.resize((128, )*2)

    img_m.append(im)
img_m[2]
img[7]
img[12]
img[17]