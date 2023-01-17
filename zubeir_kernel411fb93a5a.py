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
%matplotlib inline 

import numpy as np 

import scipy as sp 

import matplotlib as mpl

import matplotlib.cm as cm 

import matplotlib.pyplot as plt

import pandas as pd 

#from pandas.tools.plotting import scatter_matrix

pd.set_option('display.width', 500)

pd.set_option('display.max_columns', 100)

pd.set_option('display.notebook_repr_html', True)

import seaborn as sns

sns.set(style="whitegrid")

import warnings

warnings.filterwarnings('ignore')

import string

import math

import sys

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import sklearn

from IPython.core.interactiveshell import InteractiveShell



InteractiveShell.ast_node_interactivity = "all"

train = pd.read_csv('../input/titanic/train.csv')

Test = pd.read_csv("../input/titanic/test.csv")



combine = [train, Test]

combined = pd.concat(combine)
train.tail()

train.info()



print('_'*80)



Test.info()

figure, survive_bar = plt.subplots(figsize=(7, 7))

sns.barplot(x= train["Survived"].value_counts().index, y = train["Survived"].value_counts(), ax = survive_bar)

survive_bar.set_xticklabels(['Not Survived', 'Survived'])

survive_bar.set_ylabel('Frequency Count')

survive_bar.set_title('Count of Survival', fontsize = 16)



for patch in survive_bar.patches:

    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle

    label_y = patch.get_y() + patch.get_height()/2

    survive_bar.text(label_x, label_y,

                #left - freq below - rel freq wrt population as a percentage

               str(int(patch.get_height())) + '(' +

               '{:.0%}'.format(patch.get_height()/len(train.Survived))+')',

               horizontalalignment='center', verticalalignment='center')
figure, embarked_bar = plt.subplots(figsize=(7, 7))

sns.barplot(x= train["Embarked"].value_counts().index, y = train["Embarked"].value_counts(), ax = embarked_bar)

embarked_bar.set_xticklabels(['Southampton', 'Chernboug', 'Queenstown'])

embarked_bar.set_ylabel('Frequency Count')

embarked_bar.set_title('Where did the passengers board the Titanic?', fontsize = 16)
null_ages = pd.isnull(train.Age)

known_ages = pd.notnull(train.Age)

preimputation = train.Age[known_ages]

sns.distplot(preimputation)

#here we show a distribution of ages before imputation.
figure, myaxis = plt.subplots(figsize=(10, 7.5))





sns.barplot(x = "Sex", 

            y = "Survived", 

            data=train, 

            ax = myaxis,

            estimator = np.mean,

            palette = {'male':"green", 'female':"Pink"},

            linewidth=2)



myaxis.set_title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 20)

myaxis.set_xlabel("Sex",fontsize = 15)

myaxis.set_ylabel("Proportion of passengers survived", fontsize = 15)



for patch in myaxis.patches:

    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle

    label_y = patch.get_y() + patch.get_height()/2

    myaxis.text(label_x, label_y,

                #left - freq below - rel freq wrt population as a percentage

                '{:.3%}'.format(patch.get_height()),

               horizontalalignment='center', verticalalignment='center')
#Plot 1: We can use a bar plot:



figure, pclass_bar = plt.subplots(figsize = (8,10))

sns.barplot(x = "Pclass", 

            y = "Survived", 

            estimator = np.mean,

            data=train, 

            ax = pclass_bar,

            linewidth=2)

pclass_bar.set_title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 18)

pclass_bar.set_xlabel("Passenger class (Pclass)", fontsize = 15);

pclass_bar.set_ylabel("% of Passenger Survived", fontsize = 15);

labels = ['Upper (1)', 'Middle (2)', 'Lower (3)']

#val = sorted(train.Pclass.unique())

val = [0,1,2] ## this is just a temporary trick to get the label right. 

pclass_bar.set_xticklabels(labels);
