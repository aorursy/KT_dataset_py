import numpy as np

import pandas as pd

from sklearn import ensemble

# from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# train.shape
train_data = train.replace("C",0).replace("Q",1).replace("S",2)

train_data = train_data.replace("male",0).replace("female",1)

train_data.head()
train.isnull().sum()

# 年齢の欠損値データを補完

train_data.Age = train_data.fillna(train.Age.mean)

# train_data.Embarked = train_data.fillna(train.Embarked.mean)

train_data.isnull().sum()
train_data.head()
data =  train_data[['Survived','PassengerId','Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']] 

data = data.dropna()

data_learn, data_classify = train_test_split(data, test_size=0.3, random_state=123)

print ("Dimension of learn data {}".format(data_learn.shape))

print ("Dimension of classify data {}".format(data_classify.shape))
#学習用データセット用意

stdsc = StandardScaler()

y1 = data_learn['Survived']

x1 = data_learn[['PassengerId','Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]

x1_std = stdsc.fit_transform(x1)

#data_classify_std = stdsc.transform(data_classify)
#学習

rf = ensemble.RandomForestClassifier()

rf.fit(x1_std, y1)

py = rf.predict(x1_std)

from sklearn.metrics import confusion_matrix

confusion_matrix(py, y1)

from sklearn.metrics import accuracy_score

#accuracy_score(py, y1)
#テストデータを用いて識別率を求める

y2 = data_classify['Survived']

x2 = data_classify[['PassengerId','Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]

x2_std = stdsc.fit_transform(x2)

py2 = rf.predict(x2_std)

confusion_matrix(py2, y2)
accuracy_score(py2, y2)
# それぞれの変数の重要度を記述する

print(rf.feature_importances_)