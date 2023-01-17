# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

transactions = pd.read_csv('../input/transactions_v2.csv')

members = pd.read_csv('../input/members_v2.csv')

train = pd.read_csv('../input/train_v2.csv')

users = pd.read_csv('../input/user_logs_v2.csv')
df1 = members.merge(transactions, how='inner', on = 'msno', copy=False)

df2 = df1.merge(users, how= 'inner', on = 'msno', copy=False)

df = df2.merge(train, how = 'inner', on = 'msno', copy = False)

df.head(10)
df.describe()
df.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
num_cols = ['bd', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100','num_unq',

           'total_secs']
facet = None

for i in range(0,len(num_cols),2):

    if len(num_cols) > i+1:

        plt.figure(figsize=(10,4))

        plt.subplot(121)

        sns.boxplot(facet, num_cols[i],data = df)

        plt.subplot(122)            

        sns.boxplot(facet, num_cols[i+1],data = df)

        plt.tight_layout()

        plt.show()



    else:

        sns.boxplot(facet, num_cols[i],data = df)
df = df.drop(columns=['msno','transaction_date', 'membership_expire_date', 'gender',

                      'registration_init_time', 'date', 'payment_method_id', 'actual_amount_paid'], axis = 1)

df.head(10)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1]

y = df.iloc[:,-1]

X.head(10)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape)
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix as cm

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error
acc = cm(y_test, y_pred)

acc
r2_score(y_test, y_pred)
abserr = mean_absolute_error(y_test, y_pred)
abserr