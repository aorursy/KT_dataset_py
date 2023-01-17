# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score, recall_score, f1_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
df.info()
plt.hist(df['Age'])
plt.show()
pd.crosstab(df.SibSp, df.Survived).plot(kind='bar')
pd.crosstab(df.Survived, df.Sex).plot(kind='bar')
sns.violinplot(df['Sex'], df['Age'])
sns.violinplot(df['Survived'], df['Age'])
plt.show()
df[(df['Age'].isnull()==True)&(df['Sex']=='male')]
df['Call'] = df['Name'].str.extract(r' ([a-zA-Z]+)\.', expand=False)
df['Age'] = df['Age'].fillna('0')
# Fill missing values
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Fill Age message
age_data = df.groupby('Call').Age.describe().iloc[:, 1:2]
age_map = age_data.T.to_dict('records')[0]
print(age_map)
for x in range(len(df["Age"])):
    if df["Age"][x] == '0':
        df["Age"][x] = int(age_map[df["Call"][x]])
df.describe()
df[df['Survived']==0].describe()
df[df['Survived']==1].describe()
sex_map = {'male': 0, 'female': 1}
df['Sex'] = df['Sex'].map(sex_map)
embarked_map = {'S': 0, 'C': 1, 'Q': 2}
df['Embarked'] = df['Embarked'].map(embarked_map)
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
sns.heatmap(df.corr())
df['Fare'] = MinMaxScaler().fit_transform(df['Fare'].values.reshape(-1,1)).reshape(1,-1)[0]
df
df['Age'] = MinMaxScaler().fit_transform(df['Age'].values.reshape(-1,1)).reshape(1,-1)[0]
label = df['Survived']
df = df.drop('Survived', axis=1)
features = df
X_train, X_validation, Y_train, Y_validation = train_test_split(features.values, label.values, test_size=0.2)
knn = DecisionTreeClassifier()
knn.fit(X_train, Y_train)
y_val_pred = knn.predict(X_validation)
print('acc:', accuracy_score(Y_validation, y_val_pred))
print('recall:', recall_score(Y_validation, y_val_pred))
print('f1:', f1_score(Y_validation, y_val_pred))