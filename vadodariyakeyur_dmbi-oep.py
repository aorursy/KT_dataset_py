# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
data = pd.read_csv('/kaggle/input/bank-term-deposit/dataset.csv',header=0)

data = data.dropna()

print(data.shape)

print(list(data.columns))
data.head()
print(data['education'].unique())

data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])

data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])

data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])

print(data['education'].unique())
data['y'].value_counts()
sns.countplot(x='y',data=data)
count_no_sub = len(data[data['y']==0])

count_sub = len(data[data['y']==1])

pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)

print("percentage of no subscription is", pct_of_no_sub*100)

pct_of_sub = count_sub/(count_no_sub+count_sub)

print("percentage of subscription", pct_of_sub*100)
data.groupby('y').mean()
data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()
%matplotlib inline

pd.crosstab(data.job,data.y).plot(kind='bar')

plt.title('Purchase Frequency for Job Title')

plt.xlabel('Job')

plt.ylabel('Frequency of Purchase')
table=pd.crosstab(data.marital,data.y)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar')

plt.title('Marital Status vs Purchase')

plt.xlabel('Marital Status')

plt.ylabel('Proportion of Customers')
table=pd.crosstab(data.education,data.y)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar')

plt.title('Education vs Purchase')

plt.xlabel('Education')

plt.ylabel('Proportion of Customers')
pd.crosstab(data.day_of_week,data.y).plot(kind='bar')

plt.title('Purchase Frequency for Day of Week')

plt.xlabel('Day of Week')

plt.ylabel('Frequency of Purchase')
pd.crosstab(data.month,data.y).plot(kind='bar')

plt.title('Purchase Frequency for Month')

plt.xlabel('Month')

plt.ylabel('Frequency of Purchase')
data.age.hist()

plt.title('Histogram of Age')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('hist_age')
pd.crosstab(data.poutcome,data.y).plot(kind='bar')

plt.title('Purchase Frequency for Poutcome')

plt.xlabel('Poutcome')

plt.ylabel('Frequency of Purchase')

plt.savefig('pur_fre_pout_bar')
pd.crosstab(data.poutcome,data.y).plot(kind='bar')

plt.title('Purchase Frequency for Poutcome')

plt.xlabel('Poutcome')

plt.ylabel('Frequency of Purchase')
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(data[var], prefix=var)

    data1=data.join(cat_list)

    data=data1

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

data_vars=data.columns.values.tolist()

to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]

data_final.columns.values
X = data_final.loc[:, data_final.columns != 'y']

y = data_final.loc[:, data_final.columns == 'y']

from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

columns = X_train.columns

os_data_X,os_data_y=os.fit_sample(X_train, y_train)

os_data_X = pd.DataFrame(data=os_data_X,columns=columns )

os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

# we can Check the numbers of our data

print("length of oversampled data is ",len(os_data_X))

print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))

print("Number of subscription",len(os_data_y[os_data_y['y']==1]))

print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))

print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))
os_data_X.head()
os_data_y.head()
import keras

from keras.models import Sequential

from keras.layers import Dense



# Neural network

model = Sequential()

model.add(Dense(128, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=100,batch_size=32)
y_pred = model.predict(X_test)
Y_pred = []

for val in y_pred:

    Y_pred.append(np.argmax(val))
from sklearn.metrics import classification_report

print(classification_report(y_test, Y_pred))