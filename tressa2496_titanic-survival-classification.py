import numpy as np
import pandas as pd

train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
train_dataset.describe()
train_dataset.info()
import matplotlib as plt

#Bar graph Surived Vs Dead according to Pclass and Sex
def bar_chart(feature):
    survived = train_dataset[train_dataset['Survived'] == 1][feature].value_counts()
    dead = train_dataset[train_dataset['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar', stacked = True, figsize=(3,3))
    
bar_chart('Pclass')
bar_chart('Sex')
#Encoding categorical data
train_dataset['Sex'] = train_dataset['Sex'].map({'male':0, 'female':1})
test_dataset['Sex'] = test_dataset['Sex'].map({'male':0, 'female':1})
#print(train_dataset.values[:,4])
#print(test_dataset.values[:,3])
#Fitting DecisionTreeClassifier to the training set
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
X_train = train_dataset[['Pclass','Sex']]
y = train_dataset['Survived']
X_test = test_dataset[['Pclass','Sex']]
dtree.fit(X_train,y)

#Predicting Test set results
prediction = dtree.predict(X_test)

#Creating csv file of predicion
passengers_id = test_dataset['PassengerId']
dfPrediction = pd.DataFrame({'PassengerId':passengers_id, 'Survived':prediction})
dfPrediction.to_csv('submission.csv', index=False)
#print(dfPrediction)