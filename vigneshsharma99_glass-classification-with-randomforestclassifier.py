import pandas as pd

import numpy as np
df = pd.read_csv('../input/glass/glass.csv')
df
df.head()
df.isnull().sum()
df.shape
df.describe
X = df.drop(['Type'], axis = 1)
X
y = df['Type']
y
import seaborn as sns

import matplotlib.pyplot as plt
corr = df.corr()

plt.figure(figsize=(15,13))

sns.heatmap(corr,annot=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
n_trees = [10,50,100,200,300]

for i in n_trees:

    ran_for = RandomForestClassifier(n_estimators=i)

    ran_for.fit(X_train,y_train)

    pred = ran_for.predict(X_test)

    

    print('n of trees: {}'.format(i))

    correct_pred = 0

    for j,k in zip(y_test,pred):

        if j == k:

            correct_pred += 1

            print('correct predictions: {}'.format(correct_pred/len(y_test) *100))
model = RandomForestClassifier()

model.fit(X_train, y_train)
prediction = model.predict(X_test)
acc_score = accuracy_score(y_test,prediction)

acc_score