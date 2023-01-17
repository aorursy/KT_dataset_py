import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/mushrooms.csv')
df.columns
df.head(5)
labelencoder=LabelEncoder()

for column in df.columns:

    df[column] = labelencoder.fit_transform(df[column])
df.describe()
df=df.drop(["veil-type"],axis=1)
plt.figure()

pd.Series(df['class']).value_counts().sort_index().plot(kind = 'bar')

plt.ylabel("Count")

plt.xlabel("class")

plt.title('Number of poisonous/edible mushrooms (0=edible, 1=poisonous)')
plt.figure(figsize=(14,12))

sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)

plt.yticks(rotation=0)
df[['class', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='class', ascending=False)
df[['class', 'population']].groupby(['population'], as_index=False).mean().sort_values(by='class', ascending=False)
df[['class', 'spore-print-color']].groupby(['spore-print-color'], as_index=False).mean().sort_values(by='class', ascending=False)
X=df.drop(['class'], axis=1)

Y=df['class']
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1, random_state=29)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import subprocess



clf = DecisionTreeClassifier()

clf = clf.fit(X_train, Y_train)
import graphviz

dot_data = export_graphviz(clf, out_file=None,  

                         feature_names=X.columns, 

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
features_list = X.columns.values

feature_importance = clf.feature_importances_

sorted_idx = np.argsort(feature_importance)

 

plt.figure(figsize=(5,7))

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()

plt.show()
y_pred=clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print("Decision Tree Classifier report \n", classification_report(Y_test, y_pred))
confusion_matrix(Y_test, y_pred)