#Import packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

#Read and check the data

data =pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(data.shape)

data.head()

data.info()
data.describe()
#check missing data

data.isnull().sum()
#calculate how many are fraud and not fraud data? here 0 is not fraud and 1 is fraud

data.Class.value_counts()
# visualization of fraud and not fraud

LABELS=['Normal','Fraud']

sns.countplot(x = 'Class',data=data)

plt.xticks(range(2), LABELS)
amount_0 = data[data['Class']==0]['Amount']

amount_1 = data[data['Class']==1]['Amount']
time_0=data[data['Class']==0]['Time']

time_1 = data[data['Class']==1]['Time']
#Transactions in time

sns.kdeplot(time_0,label='Not_fraud')

sns.kdeplot(time_1,label='Fraud')

plt.title('Credit Card Transactions Time Density Plot',size = 20)
fig,(ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize = (8,9))

fig.suptitle('Time of transaction vs Amount by class')

ax1.scatter(time_0, amount_0)

ax1.set_title('Not Fraud')

ax2.scatter(time_1, amount_1)

ax2.set_title('Fraud')

plt.xlabel('Time[s]')

plt.ylabel('Amount')

plt.show()

#Feature correlation

corre = data.corr()

top_fea = corre.index

plt.figure(figsize=(20,20))

sns.heatmap(data[top_fea].corr(),annot = True,cmap="RdYlGn")
df1 = data[['Amount','Time']]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled= scaler.fit_transform(df1)
data1 = pd.DataFrame(scaled ,columns= ['Amount_Scale','Time_Scale'])

data.drop(['Amount','Time'],axis= 1,inplace = True)

data = pd.concat([data1,data],axis=1)
data.head()
data = pd.concat([data1,data],axis=1)
X = data.drop('Class',axis = 1)

Y = data['Class']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size =0.7,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

lr = LogisticRegression()

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

confusion_matrix(y_test,y_pred)
from sklearn import metrics

metrics.accuracy_score(y_test,y_pred)