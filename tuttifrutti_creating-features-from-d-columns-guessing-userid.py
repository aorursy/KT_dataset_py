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
#import data

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
import matplotlib.pyplot as plt
SEED = 12345



#creating Day feature

train_transaction['TransactionDTday'] = (train_transaction['TransactionDT']/(60*60*24)).map(int)

test_transaction['TransactionDTday'] = (test_transaction['TransactionDT']/(60*60*24)).map(int)



#stolen from alijs and slightly modified: https://www.kaggle.com/alijs1/ieee-transaction-columns-reference

def timehist1_2(col,product):

    N = 8000 if col in ['TransactionAmt'] else 9999999999999999 # clip trans amount for better view

    train_transaction[(train_transaction['isFraud'] == 0) & (train_transaction['ProductCD'] == product)].set_index('TransactionDT')[col].clip(0, N).plot(style='.', title='Hist ' + col, figsize=(15, 3))

    train_transaction[(train_transaction['isFraud'] == 1) & (train_transaction['ProductCD'] == product)].set_index('TransactionDT')[col].clip(0, N).plot(style='.', title='Hist ' + col, figsize=(15, 3))

    test_transaction[test_transaction['ProductCD'] == product].set_index('TransactionDT')[col].clip(0, N).plot(style='.', title=col + ' values over time (blue=no-fraud, orange=fraud, green=test)', figsize=(15, 3))

    plt.show()
products=train_transaction.ProductCD.unique().tolist()

col='D1'

for prod in products: 

    print("Product code:", prod)

    timehist1_2(col, prod)
train_transaction['D1minusday'] = train_transaction['D1'] - train_transaction['TransactionDTday']

test_transaction['D1minusday'] = test_transaction['D1'] - test_transaction['TransactionDTday']
col='D1minusday'

for prod in ['S']: 

    print("Product code:", prod)

    timehist1_2(col, prod)
train_transaction[(train_transaction.isFraud==1) & (train_transaction.ProductCD=='S') & (train_transaction.D1minusday>50)]['D1minusday'].value_counts()
import seaborn as sns

print('Blue: Frauds, Orange: Non-Fraud')

sns.distplot(train_transaction[(train_transaction.isFraud==1) & (train_transaction.ProductCD=='S')]['D1minusday'], hist=False, rug=False);

sns.distplot(train_transaction[(train_transaction.isFraud==0) & (train_transaction.ProductCD=='S')]['D1minusday'], hist=False, rug=False);
train_transaction[(train_transaction.isFraud==1) & (train_transaction.D1minusday==78)][['card1','card2','card3','card4','card5','card6','addr1',

 'addr2',

 'dist1',

 'dist2','P_emaildomain','R_emaildomain','TransactionDTday']]