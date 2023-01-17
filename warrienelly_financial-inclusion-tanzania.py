# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from datetime import datetime

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

# from ipywidgets import widgets

from IPython.display import display, clear_output, Image

# from plotly.widgets import GraphWidget

#%matplotlib inline

import seaborn as sns

import warnings

sns.set()



from scipy.stats import probplot # for a qqplot
# %reset -f 
train=pd.read_csv('/kaggle/input/Train_v2.csv')

test=pd.read_csv('/kaggle/input/Test_v2.csv')

# submission_file=pd.read_csv('/kaggle/input/SubmissionFile.csv')

# Variables=pd.read_csv('VariableDefinitions.csv')
all_data = pd.concat([train, test])
all_data.shape, test.shape, train.shape
all_data.isnull().sum()
## no missing values in the data
all_data.describe()
# submission_file.to_csv('all_1 values.csv', index=False)
all_data.nunique()
train.info()
for col in all_data.columns:

    print(col)

    print(all_data[col].value_counts())

    print('==========================')
for col in train.columns:

    print(col)

    print(train[col].value_counts())

    print('==========================')
test.education_level.value_counts()
cat_col = train.select_dtypes(exclude=np.number).drop(['uniqueid', 'bank_account'], axis=1)

num_col = train.select_dtypes(include=np.number)
## count plot function

def cat_plot (col):

    

    ## create 2 plot one for train and another for test

    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))

    

    f = sns.countplot(x=col, data=train, ax=ax)

        ## write ontop of box snippet

    for p in f.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f.title.set_text('Bar plot of train ' + col)

    f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")

   

    ## plot for test

    f1 = sns.countplot(x=col, data=test, ax=ax1)

        ## write ontop of box snippet

    for p in ax1.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f1.annotate('{:.1f}%'.format(100.*p.get_height()/test.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f1.title.set_text('Bar plot of test ' + col)

    f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")

    fig.tight_layout()

    fig.show()  
## Gui of countplot

@interact

def show_cat_plot(col=cat_col.columns):

    cat_plot(col)
num_col.nunique()
def num_plot(col):

    

    

    ## create 2 plot one for train and another for test

    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))

    

   ## Am using the min and max value of the col to set x and y axis

    if col == 'age_of_respondent':

        ax.set_xlim(13,100)

        ax.set_xticks(range(13,110,4))

        ax1.set_xlim(13,110)

        ax1.set_xticks(range(13,101,4))

    if col == 'household_size':

        ax.set_xlim(0,25)

        ax.set_xticks(range(0,26))

        ax1.set_xlim(0,26)

        ax1.set_xticks(range(0,26))

    f = sns.distplot(train[col], rug=True, ax=ax)

        ## write ontop of box snippet

    f.title.set_text('hist plot of train ' + col)

#     f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")

   

    ## plot for test

    f1 = sns.distplot(test[col],  rug=True, ax=ax1)

    f1.title.set_text('hist plot of test ' + col)

#     f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")

    fig.tight_layout()

    fig.show()  

    plt.show()
# num_col.household_size.describe()
# num_col['age_of_respondent'].describe()
@interact

def show_num_plot(col=num_col.columns):

    num_plot(col)
sns.distplot(np.log(train.age_of_respondent))
# Look deeply into this data since it is categorical in nature, i can use value count
## get counto 10 highest age of train

test.age_of_respondent.value_counts()[:10]
## get counto 10 highest age of test

num_col.age_of_respondent.value_counts()[:10]
num_col.household_size.value_counts()
num_bins = 20

counts, bin_edges = np.histogram (num_col.age_of_respondent, bins=num_bins, normed=True)

cdf = np.cumsum (counts)

plt.plot (bin_edges[1:], cdf/cdf[-1])



# statsmodels Q-Q plot on model residuals

for col in num_col.columns:

    probplot(num_col[col], dist="norm", plot=plt)

    plt.show()
# statsmodels Q-Q plot on model residuals

for col in num_col.columns:

    probplot(np.log(train[col]), dist="norm", plot=plt)

    plt.show()
# Possible solution to numeric col

# log age

# make year and child num cat

## count plot function

def cat_plot (col):

    

    ## create 2 plot one for train and another for test

    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))

    

    f = sns.countplot(x=col, data=train, ax=ax, hue='bank_account')

        ## write ontop of box snippet

    for p in f.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f.title.set_text('Bar plot of train ' + col)

    f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")

   

    ## plot for test

    f1 = sns.countplot(x=col, data=test, ax=ax1)

        ## write ontop of box snippet

    for p in ax1.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f1.annotate('{:.1f}%'.format(100.*p.get_height()/test.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f1.title.set_text('Bar plot of test ' + col)

    f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")

    fig.tight_layout()

    fig.show()  
@interact

def cat_with_targrt(col=cat_col.columns):

    cat_plot(col)
train.job_type.unique()
# Old people dont have cell_access but have bank_account, why??

## This happens mostly in kenya 

## mostly in rural area
for col in train.columns:

    print(col)

    print (train.loc[(train.bank_account == 'Yes') & (train.cellphone_access == 'No') & (train.age_of_respondent > 69)][col].value_counts())

    print('===================')
@interact

def cat_with_targrt(col=num_col.columns):

    cat_plot(col)
test.job_type.value_counts()
def num_plot(col):

    

    

    ## create 2 plot one for train and another for test

    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))

    

   ## Am using the min and max value of the col to set x and y axis

    if col == 'age_of_respondent':

        ax.set_xlim(13,100)

        ax.set_xticks(range(13,110,4))

        ax1.set_xlim(13,110)

        ax1.set_xticks(range(13,101,4))

    if col == 'household_size':

        ax.set_xlim(0,25)

        ax.set_xticks(range(0,26))

        ax1.set_xlim(0,26)

        ax1.set_xticks(range(0,26))

    f = sns.distplot(train.loc[train.bank_account == 'No'][col],hist=False, rug=True, ax=ax, label="No")

    f = sns.distplot(train.loc[train.bank_account == 'Yes'][col],hist=False, rug=True, ax=ax, label="Yes")

    

        ## write ontop of box snippet

    f.title.set_text('hist plot of train ' + col)

#     f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")

   

    ## plot for test

    f1 = sns.distplot(test[col],  rug=True, ax=ax1)

    f1.title.set_text('hist plot of test ' + col)

#     f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")

    fig.tight_layout()

    fig.show()  

    plt.show()
@interact

def num_wrt_target(col=num_col.columns):

    num_plot(col)
## does houeshold size really affect acct number??

## Those in their 20's tend to use a bank.

# What if i analyze those in 70's who have and analyse those in 20's who do not have
## count plot function

def cat_plot (col):

    

    ## create 2 plot one for train and another for test

    fig, ((ax),(ax1)) = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(15,5))

    

    f = sns.countplot(x=col, data=train, ax=ax, hue='country')

        ## write ontop of box snippet

    for p in f.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f.title.set_text('Bar plot of train ' + col)

    f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")

   

    ## plot for test

    f1 = sns.countplot(x=col, data=test, ax=ax1,hue='country')

        ## write ontop of box snippet

    for p in ax1.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f1.annotate('{:.1f}%'.format(100.*p.get_height()/test.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f1.title.set_text('Bar plot of test ' + col)

    f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")

    fig.tight_layout()

    fig.show()  
@interact

def cat_with_targrt(col=cat_col.columns):

    cat_plot(col)
def num_plot(col):

    

    

    ## create 2 plot one for train and another for test

    fig, ((ax),(ax1),(ax2),(ax3)) = plt.subplots(2, 2, sharex=False, sharey=False,figsize=(15,5))

    

   ## Am using the min and max value of the col to set x and y axis

    if col == 'age_of_respondent':

        ax.set_xlim(13,100)

        ax.set_xticks(range(13,110,4))

        ax1.set_xlim(13,110)

        ax1.set_xticks(range(13,101,4))

    if col == 'household_size':

        ax.set_xlim(0,25)

        ax.set_xticks(range(0,26))

        ax1.set_xlim(0,26)

        ax1.set_xticks(range(0,26))

    f = sns.distplot(train.loc[train.country == 'Rwanda'][col],hist=False, rug=True, ax=ax, label="Rwanda")

    f = sns.distplot(train.loc[train.country == 'Tanzania'][col],hist=False, rug=True, ax=ax, label="Tanzania")

    f = sns.distplot(train.loc[train.country == 'Kenya'][col],hist=False, rug=True, ax=ax, label="Kenya")

    f = sns.distplot(train.loc[train.country == 'Uganda'][col],hist=False, rug=True, ax=ax, label="Uganda")



    

        ## write ontop of box snippet

    f.title.set_text('hist plot of train ' + col)

#     f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")

   

    ## plot for test

    f1 = sns.distplot(train.loc[train.country == 'Tanzania'][col],  rug=True, ax=ax1)

    f1.title.set_text('hist plot of test ' + col)

#     f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")

    fig.tight_layout()

    fig.show()  

    plt.show()
@interact

def num_wrt_target(col=num_col.columns):

    num_plot(col)
@interact

def cat_with_targrt(col=num_col.columns):

    cat_plot(col)
train.loc[train.country == 'Uganda']['age_of_respondent'].value_counts()

# train.loc[train.country == 'Tanzania'][col],hist=False, rug=True, ax=ax, label="Tanzania")

#     f = sns.distplot(train.loc[train.country == 'Kenya'][col],hist=False, rug=True, ax=ax, label="Kenya")

#     f = sns.distplot(train.loc[train.country == 'Uganda'][col],hist=False, rug=True, ax=ax, label="Uganda")

train.country.value_counts()
def num_plot(col):

    

    

    ## create 2 plot one for train and another for test

    fig, ((ax,ax1),(ax2,ax3)) = plt.subplots(2, 2, sharex=False, sharey=False,figsize=(15,5))

    

   ## Am using the min and max value of the col to set x and y axis

    if col == 'age_of_respondent':

        ax.set_xlim(13,100)

        ax.set_xticks(range(13,110,4))

        ax1.set_xlim(13,110)

        ax1.set_xticks(range(13,101,4))

    if col == 'household_size':

        ax.set_xlim(0,25)

        ax.set_xticks(range(0,26))

        ax1.set_xlim(0,26)

        ax1.set_xticks(range(0,26))

#     f = sns.distplot(train.loc[train.country == 'Rwanda'][col],hist=False, rug=True, ax=ax, label="Rwanda")

#     f = sns.distplot(train.loc[train.country == 'Tanzania'][col],hist=False, rug=True, ax=ax, label="Tanzania")

#     f = sns.distplot(train.loc[train.country == 'Kenya'][col],hist=False, rug=True, ax=ax, label="Kenya")

#     f = sns.distplot(train.loc[train.country == 'Uganda'][col],hist=False, rug=True, ax=ax, label="Uganda")

    f = sns.countplot(x=train.loc[train.country == 'Rwanda'][col], data=train, ax=ax, hue='bank_account')

        ## write ontop of box snippet

    for p in f.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f.title.set_text('Bar plot of train Rwanda ' + col)

    f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")

    

        ## write ontop of box snippet

#     f.title.set_text('hist plot of train ' + col)

#     f.set_xticklabels(f.get_xticklabels(), rotation=40, ha="right")

   

    ## plot for tanzania

    f1 = sns.countplot(x=train.loc[train.country == 'Tanzania'][col], data=train, ax=ax1, hue='bank_account')

        ## write ontop of box snippet

    for p in f1.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f1.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f1.title.set_text('Bar plot of train Tanzania ' + col)

    f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")

    

      

#     f1.set_xticklabels(f1.get_xticklabels(), rotation=40, ha="right")



    ## plot for tanzania

    f2 = sns.countplot(x=train.loc[train.country == 'Kenya'][col], data=train, ax=ax2, hue='bank_account')

        ## write ontop of box snippet

    for p in f2.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f2.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f2.title.set_text('Bar plot of train Kenya ' + col)

    f2.set_xticklabels(f2.get_xticklabels(), rotation=40, ha="right")

    



    

    f3 = sns.countplot(x=train.loc[train.country == 'Uganda'][col], data=train, ax=ax3, hue='bank_account')

        ## write ontop of box snippet

    for p in f3.patches:

        ## Get box location

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ## write percentage ontop of box

        f3.annotate('{:.1f}%'.format(100.*p.get_height()/train.shape[0]), (x.mean(), y.mean()), 

                ha='center', va='bottom') # set the alignment of the text

    f3.title.set_text('Bar plot of train Uganda ' + col)

    f3.set_xticklabels(f3.get_xticklabels(), rotation=40, ha="right")

    

    fig.tight_layout()

    fig.show()  

    plt.show()
@interact

def num_wrt_target(col=train.columns):

    num_plot(col)
@interact

def num_wrt_target(col=num_col.columns):

    num_plot(col)
train.loc[(train.country == 'Uganda') & (train.bank_account == 'Yes')]['job_type'].value_counts()