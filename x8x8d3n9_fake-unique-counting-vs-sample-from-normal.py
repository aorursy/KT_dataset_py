import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

import  warnings

warnings.simplefilter('ignore')
#######################################load data################################################################

train=pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

test=pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

pr_b=np.load('../input/list-of-fake-samples-and-public-private-lb-split/synthetic_samples_indexes.npy')

################################Split syn and true##############################################################

train.drop(['ID_code','target'],axis=1,inplace=True)

test=test.drop('ID_code',axis=1)

test1=test.iloc[pr_b]

data1=pd.concat([train,test1])
np.random.seed(9)

# true values unique number  VS random generated samples

for col in test.columns:

    print(col)

    a=data1[col].map(data1[col].value_counts())

    sns.distplot(np.round(a,4),bins=100)

    plt.show()

    a=pd.DataFrame(np.round(np.random.normal(loc=np.mean(data1[col]), scale=np.std(data1[col]), size=300000),4))

    a.columns=['x']

    b=a.x.map(a.x.value_counts())

    sns.distplot(b,bins=100)

    plt.show()

    