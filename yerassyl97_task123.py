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
num_bins = 10

plt.hist(data_train['var_1'], num_bins, density=1, facecolor='blue', alpha=0.5)

plt.show()
num_bins = 10

plt.hist(data_test['var_0'], num_bins, density=1, facecolor='blue', alpha=0.5)

plt.show()
data_test.info()
data_train.info()
data_sample.info()
x = data_train.iloc[:, 2:].values

y = data_train.target.values

x_test = data_test.iloc[:, 1:].values
x_train = x

y_train = y
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train, y_train)
#so general interesting result is coming

predictions = logmodel.predict(x_test)
from sklearn.metrics import classification_report

print(classification_report(y_train,predictions))
sub_result = pd.DataFrame({'ID_code':data_test.ID_code.values})

sub_result['target'] = predictions

sub_result.to_csv('submission.csv', index=False)