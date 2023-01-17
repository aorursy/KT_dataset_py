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
data_train.info()
data_test.info()
data_sample.info()
data_test.describe()

data_train.info()
data_train32 = data_train.drop(['ID_code', 'target'], axis = 1).astype('float32')
data_train32.info()
data_train32.head()
# main algorithms and functions to use

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
## checking the balance of the data by plotting the count of outcomes by their value



data_train.target.value_counts().plot(kind="bar")



## this  is just for fun result, nothing primary result =)
x_train = data_train.iloc[:, 2:].values

y_train = data_train.target.values

x_test = data_test.iloc[:, 1:].values

data_train.target.value_counts()
x_test
x_train.shape
y_train.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)
print(classification_report(y_train,predictions))
sub_df = pd.DataFrame({'ID_code':data_test.ID_code.values})

sub_df['target'] = predictions

sub_df.to_csv('submission_logreg.csv', index=False)
data_train.info()
data_test.info()
x_train = data_train.iloc[:, data_train.columns != 'target'].values

y_train = data_train.iloc[:, 1].values

x_test = data_test.values
x_train
x_test
data_train.describe()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()





x_train[:,0] = label_encoder.fit_transform(x_train[:,0])

x_test[:,0] = label_encoder.fit_transform(x_test[:,0])

knn = KNeighborsClassifier(50)
knn.fit(x_train, y_train)
y_preds = knn.predict(x_test)
y_preds
sub_df = pd.DataFrame({'ID_code':data_test.ID_code.values})

sub_df['target'] = y_preds

sub_df.to_csv('submission_knn_result_100.csv', index=False)