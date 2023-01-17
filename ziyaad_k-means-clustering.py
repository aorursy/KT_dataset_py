import pandas as pd

import numpy as np
data = pd.read_csv("../input/train.csv")



data.head()
data.isnull().sum()
data['Age'] = data['Age'].fillna(data['Age'].mean())
data.isnull().sum()
data[['Pclass','Survived']].groupby(['Pclass']).mean()
data[['Sex','Survived']].groupby(['Sex']).mean()
data[['SibSp','Survived']].groupby(['SibSp']).mean()
import seaborn as sns

import matplotlib.pyplot as plt



a = sns.FacetGrid(data,col='Survived')

a.map(plt.hist,'Age',bins=20)

plt.show()
import seaborn as sns

import matplotlib.pyplot as plt



a = sns.FacetGrid(data,col='Survived',row='Pclass')

a.map(plt.hist,'Age',bins=20)

a.add_legend()

plt.show()
data.head()
data = data.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
y = data['Survived']
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

data['Sex'] = le.fit_transform(data['Sex'])
x=data

x = x.drop(['Survived'],axis=1)
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=2)

kmeans.fit(x)
x.iloc[0]
a = kmeans.predict(x)
from sklearn.metrics import accuracy_score



accuracy_score(y,a)
correct = 0

for i in range(len(x)):

    predict = np.array(x.iloc[i])

    print("Row : ",predict)

    predict = predict.reshape(-1,len(predict))

    prediction = kmeans.predict(predict)

    if(prediction[0] == y[i]):

        correct += 1

        

print(correct/len(x))
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

x_scale = scaler.fit_transform(x)



kmeans.fit(x_scale)



a = kmeans.predict(x)
accuracy_score(y,a)