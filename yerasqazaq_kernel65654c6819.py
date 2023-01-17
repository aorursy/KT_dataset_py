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
train_data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

test_data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')



train_data.head()
test_data.head()
train_data.shape
test_data.shape
train_data.isnull().sum()
test_data.isnull().sum()
X_train = train_data.iloc[:, 2: ].values

y_train = train_data.iloc[:,0]

X_train
y_train
y_train = train_data.iloc[:,0].values
y_train
X_test = test_data.iloc[:, 1:].values
X = train_data.iloc[:, 2:]

y = train_data.iloc[:, 1]
X_test
X
y
train_data.nunique()
test_data.nunique()
import seaborn as seaborn

import matplotlib.pyplot as plt
seaborn.countplot(train_data.target)
# ScatterPlot for train and test Dataset



def plot_feature_scatter(df1 , df2 ,features):

    i = 0

    seaborn.set_style('whitegrid')

    plt.figure()

    fig , ax = plt.subplots(3, 3,figsize = (20,20))



    for feature in features:

        i += 1

        plt.subplot(3, 3,i)

        plt.scatter(df1[feature], df2[feature],marker = '+')

        plt.xlabel(feature , fontsize = 9)

    plt.show();
features = ['var_0','var_1','var_2','var_3','var_4','var_5','var_6','var_7','var_8']



plot_feature_scatter(train_data[::20],test_data[::20],features)
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50)
from sklearn.preprocessing import StandardScaler
stScal = StandardScaler()

X_train = stScal.fit_transform(X_train)

X_test = stScal.transform(X_test)

X = stScal.transform(X)
def plot_feat_graf(df1, df2, label1, label2, features):

    i = 0

    seaborn.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(10,10,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(10,10,i)

        seaborn.distplot(df1[feature], hist=False,label=label1)

        seaborn.distplot(df2[feature], hist=False,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show();
train0 = train_data.loc[train_data['target'] == 0]

train1 = train_data.loc[train_data['target'] == 1]



feat = train_data.columns.values[2:102]

plot_feat_graf(train0 , train1 , '0','1',feat)
feat  = train_data.columns.values[102:202]

plot_feat_graf(train0 , train1 , '0','1',feat)
#Distribution of data 
feat = train_data.columns.values[2:102]

plot_feat_graf(train_data , test_data , 'train','test', feat)
feat = train_data.columns.values[102:202]

plot_feat_graf(train_data , test_data , 'train','test', feat)
plt.figure(figsize = (16,6))

plt.title('distribution of mean values per column in the train set')

seaborn.distplot(train0[feat].mean(axis = 0),color = 'green',kde = True , bins = 120,label = 'target = 0')

seaborn.distplot(train1[feat].mean(axis = 0),color = 'blue',kde = True , bins = 120 , label = 'target = 1')

plt.legend()

plt.show();
# Check the distribution of min per row in the train and test set



plt.figure(figsize = (16,6))

features = train_data.columns.values[2:202]

plt.title('Distribution of min values per row in the train and test set')

seaborn.distplot(train_data[feat].min(axis = 1),color = 'red',kde = True , bins = 120 , label = 'train')

seaborn.distplot(test_data[feat].min(axis = 1),color = 'orange',kde = True , bins = 120 , label = 'test')

plt.legend()

plt.show();
# check the distribtuion of min per column in the train and test set



plt.figure(figsize =(16,6))

features = train_data.columns.values[2:202]

plt.title('Distribution of min values per column in the train and test set')

seaborn.distplot(train_data[feat].min(axis = 0),color = 'magenta',kde = True , bins = 120, label = 'train')

seaborn.distplot(test_data[feat].min(axis = 0),color = 'darkblue',kde = True , bins =120,label = 'test')

plt.legend()

plt.show()
# Check the distribution of max values per rows for train and test set

plt.figure(figsize = (16,6))

features = train_data.columns.values[2:202]

plt.title('Distribution of max values per row in the train and test set')

seaborn.distplot(train_data[feat].max(axis = 1),color = 'brown',kde = True , bins =120 , label = 'train')

seaborn.distplot(test_data[feat].max(axis = 1),color = 'yellow',kde = True , bins =120 , label = 'test')

plt.legend()

plt.show()
# Check max distribution on columns for train and test set



plt.figure(figsize = (16,6))

features = train_data.columns.values[2:202]

plt.title('Distribution of max values per column in the train and test set')

seaborn.distplot(train_data[feat].max(axis = 0), color = 'blue',kde = True , bins = 120 , label = 'train')

seaborn.distplot(test_data[feat].max(axis = 0),color = 'red',kde = True , bins =120 , label = 'test')

plt.legend()

plt.show()
# Check distribution of min values per row in train set



plt.figure(figsize = (16,6))

plt.title('Distribution of min values per column in the train set')

seaborn.distplot(train0[feat].min(axis = 0),color = 'red',kde = True , bins =120 , label = 'target =0')

seaborn.distplot(train1[feat].min(axis = 0),color = 'blue',kde = True , bins = 120 , label = 'target = 1')

plt.legend()

plt.show();
plt.figure(figsize = (16,6))

plt.title('Distribution of skew per row in the train and test set')

seaborn.distplot(train_data[feat].skew(axis = 1),color = 'red',kde = True , bins = 120,label = 'train')

seaborn.distplot(test_data[feat].skew(axis = 1),color = 'orange',kde = True , bins = 120 ,label = 'test')

plt.legend()

plt.show()
plt.figure(figsize = (16,6))

plt.title('Distribution of skew per column in the train and test set')

seaborn.distplot(train_data[feat].skew(axis = 0),color = 'magenta',kde = True , bins = 120 , label = 'train')

seaborn.distplot(test_data[feat].skew(axis = 0), color = 'darkblue',kde = True , bins = 120 , label = 'test')

plt.legend()

plt.show()
train0 = train_data.loc[train_data['target'] == 0]

train1 = train_data.loc[train_data['target'] == 1]

plt.figure(figsize=(16,6))

plt.title("Distribution of kurtosis values per column in the train set")

seaborn.distplot(train0[feat].kurtosis(axis=0),color="red", kde=True,bins=120, label='target = 0')

seaborn.distplot(train1[feat].kurtosis(axis=0),color="blue", kde=True,bins=120, label='target = 1')

plt.legend();

plt.show()



##

##https://www.kaggle.com/saeedtqp/customer-transaction-predict