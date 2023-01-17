import numpy as np

import pandas as pd

from sklearn import tree



fields=['Survived','Pclass','Sex','Age','SibSp','Parch','Cabin','Fare','Embarked']

fields_test=['Pclass','Sex','Age','SibSp','Parch','Cabin','Fare','Embarked']



mymap = {'male':1, 'female':2, 'C':1, 'Q':2, 'S':3}



df=pd.read_csv('../input/train.csv',usecols=fields)

df['Age'].fillna(df['Age'].mean(),inplace=True)

df.fillna(0,inplace=True)

df=df.applymap(lambda s: mymap.get(s) if s in mymap else s)

for i in range(891):

    if(df.iloc[i,7]):

        df.iloc[i,7]=1        



df2=pd.read_csv('../input/test.csv',usecols=fields_test)

df2['Age'].fillna(df2['Age'].mean(),inplace=True)

df2.fillna(0,inplace=True)

df2=df2.applymap(lambda s: mymap.get(s) if s in mymap else s)

for i in range(418):

    if(df2.iloc[i,6]):

        df2.iloc[i,6]=1



testdata=np.array(df2)

Y=np.array(df.iloc[:,0])

X=np.array(df.iloc[:,1:])

clf = tree.DecisionTreeClassifier(min_samples_split=20)

clf=clf.fit(X, Y)



pred = clf.predict(testdata)

print(pred)

df2=pd.read_csv('../input/test.csv',usecols=['PassengerId'])

dff = pd.concat([df2, pd.DataFrame(pred)], axis = 1)

dff.columns = ['PassengerId', 'Survived']

dff.to_csv('abc.csv', index = False)

print(dff.head())
