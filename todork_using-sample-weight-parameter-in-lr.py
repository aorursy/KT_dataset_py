# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#First load train dataset

train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
dummy_cols=['bin_0','bin_1','bin_2','bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9','ord_0','ord_1','ord_2','ord_3','ord_4','ord_5','month', 'day']

train = pd.get_dummies(train, columns=dummy_cols, sparse=True)
#validation dataset

validate = pd.concat([train[train['target']==0].head(3000),train[train['target']==1].head(3000)]).reset_index(drop=True)



#drop validation rows from train

train=train[~train.id.isin(validate.id)].reset_index(drop=True)



#get target column, then drop 'id' and  'target' from the two dataframes

target = train['target']

train = train.drop(['id','target'], axis=1)

target_val = validate['target']

validate = validate.drop(['id','target'], axis=1)



#convert to sparse 

train = train.sparse.to_coo().tocsr()

validate = validate.sparse.to_coo().tocsr()

def lr_classifier(x, y, x_val, y_val, sample_weight):

    lr = LogisticRegression(solver = 'lbfgs', C = 0.1, max_iter=1000)

    lr.fit(x,y,sample_weight) 

    y_pred = lr.predict(x_val)

    

    cm = confusion_matrix(y_val, y_pred )

    cm = cm.astype('float') / cm.sum(axis=1)



    plt.matshow(cm)

    plt.title('Confusion matrix')

    for (i, j), z in np.ndenumerate(cm):

        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.show()
lr_classifier(train,target,validate,target_val,sample_weight = None)

sample_weight= target.apply( lambda x: 2.27 if x==1 else 1)

lr_classifier(train,target,validate,target_val,sample_weight)