import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
data_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

data_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

data_sample = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
data_train
data_train.describe()
data_test.columns
data_test.describe()
data_sample
data_sample.describe()
num_bins = 10

plt.hist(data_train['var_1'], num_bins, density=1, facecolor='blue', alpha=0.5)

plt.show()
num_bins = 10

plt.hist(data_train['var_199'], num_bins, density=1, facecolor='blue', alpha=0.5)

plt.show()
num_bins = 10

plt.hist(data_test['var_0'], num_bins, density=1, facecolor='blue', alpha=0.5)

plt.show()
num_bins = 10

plt.hist(data_test['var_199'], num_bins, density=1, facecolor='blue', alpha=0.5)

plt.show()
x = data_train.iloc[:, 2:].values

y = data_train.target.values

x_test = data_test.iloc[:, 1:].values
#creating new useful variables

x_train = x

y_train = y
#Let's see x train data after calculation

x_test
y
#our shape of x test and y test

x_test.shape

x_train.shape
from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB
gauss = GaussianNB()

y_prediction = gauss.fit(x_train, y_train).predict(x_test)
print(classification_report(y_train,y_prediction))


sub_df = pd.DataFrame({'ID_code':data_test.ID_code.values})

sub_df['target'] = y_prediction

sub_df.to_csv('submission.csv', index=False)


