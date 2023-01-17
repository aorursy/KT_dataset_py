# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

np.random.seed(43)

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

from sklearn.preprocessing import LabelEncoder
nameEncoder = LabelEncoder()
data['Name']=nameEncoder.fit_transform(data['Name'].transform(lambda n: n[n.index(',')+2:n.index('.')]))
data['Age']=data['Age'].fillna(data['Age'].median())
data['Sex']=data['Sex'].transform(lambda s: 0 if s=='female' else 1)
data['Cabin']=data['Cabin'].transform(lambda c: 0 if isinstance(c, float) else (int(c[1:]) if c[1:].isdigit() else 0))
embarkedEncoder = LabelEncoder()
data['Embarked']=embarkedEncoder.fit_transform(data['Embarked'].transform(lambda e: e if isinstance(e, str) else 'NA'))
data['FamilySize']=data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
data = data.drop('Ticket', axis=1)
data.info()
data.corr()['Survived'].sort_values(ascending=False)
# %matplotlib inline
# import matplotlib.pyplot as plt
# data.hist(bins=50, figsize=(20,15))
# plt.show()
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from IPython.display import Image as PImage
# from subprocess import check_call
# from PIL import Image, ImageDraw, ImageFont

# dt = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=0.005)
# dt.fit(data.drop('Survived', axis=1).drop('PassengerId', axis=1), data['Survived'])

# with open("tree1.dot", 'w') as f:
#      f = export_graphviz(dt,
#                               out_file=f,
#                               impurity = True,
#                               feature_names = list(data.drop('Survived', axis=1).drop('PassengerId', axis=1)),
#                               class_names = ['Died', 'Survived'],
#                               rounded = True,
#                               filled = True )
        
# #Convert .dot to .png to allow display in web notebook
# check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
# PImage("tree1.png")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

scaler = StandardScaler()
scaler.fit(data.drop('Survived', axis=1).drop('PassengerId', axis=1))

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(64, input_dim=data.drop('Survived', axis=1).drop('PassengerId', axis=1).shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.15))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.15))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.15))
model.add(Dense(2, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("Training...")
model.optimizer.lr=0.01
model.fit(scaler.transform(data.drop('Survived', axis=1).drop('PassengerId', axis=1)), np_utils.to_categorical(data['Survived']), epochs=20, batch_size=32, validation_split=0.05, verbose=2)

# dt = Pipeline((("scaler", StandardScaler()), ('lr', LogisticRegression())))
# dt = Pipeline((("scaler", StandardScaler()), ("lsvc", SVC(C=0.1))))
# dt = VotingClassifier(estimators=[('lr', LogisticRegression()), ('svc', SVC(probability=True)), ('dt', DecisionTreeClassifier(criterion='gini', min_impurity_decrease=0.005))], voting='soft')
# dt = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, oob_score=True)
# dt = ExtraTreesClassifier(n_estimators=100, max_leaf_nodes=16, random_state=27)
dt = VotingClassifier(estimators=[('1', RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, oob_score=True)), 
                                  ('2', ExtraTreesClassifier(n_estimators=100, max_leaf_nodes=16, random_state=27)),
                                  ('3', DecisionTreeClassifier(criterion='gini', min_impurity_decrease=0.005)),
                                  ('4', AdaBoostClassifier()),
                                  ('5', GradientBoostingClassifier(max_depth=3, n_estimators=100, random_state=42, n_iter_no_change=10)),
                                  ('6', SVC(C=1, probability=True))
                                 ], voting='soft')
dt.fit(scaler.transform(data.drop('Survived', axis=1).drop('PassengerId', axis=1)), data['Survived'])
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.metrics import confusion_matrix
# scores = cross_val_score(dt, data.drop('Survived', axis=1).drop('PassengerId', axis=1), data['Survived'], scoring='accuracy')
# print(scores.mean(), scores.std())
# dy = cross_val_predict(dt, data.drop('Survived', axis=1).drop('PassengerId', axis=1), data['Survived'])
# print(confusion_matrix(data['Survived'], dy))
# len(data[(dy==0)&(data['Survived']==1)&(data['Sex']==1)])
test = pd.read_csv('../input/test.csv')
test['Name']=nameEncoder.transform(test['Name'].transform(lambda n: n[n.index(',')+2:n.index('.')]).transform(lambda n: 'Don' if n == 'Dona' else n))
test['Age']=test['Age'].fillna(data['Age'].median())
test['Fare']=test['Fare'].fillna(data['Fare'].median())
test['Sex']=test['Sex'].transform(lambda s: (0 if s=='female' else 1))
test['Cabin']=test['Cabin'].transform(lambda c: 0 if isinstance(c, float) else (int(c[1:]) if c[1:].isdigit() else 0))
test['Embarked']=embarkedEncoder.transform(test['Embarked'].transform(lambda e: e if isinstance(e, str) else 'NA'))
test['FamilySize']=test['SibSp'] + test['Parch'] + 1
test['IsAlone'] = 0
test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

test = test.drop('Ticket', axis=1)
test.info()

y1 = model.predict(scaler.transform(test.drop('PassengerId', axis=1)))
y2 = dt.predict_proba(scaler.transform(test.drop('PassengerId', axis=1)))
y = (y1+y2)/2
ty = [1 if yy[1] > 0.5 else 0 for yy in y]
print(sum(ty)/len(ty), sum(data['Survived'])/len(data['Survived']))
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': ty })
submission.to_csv("dtSubmission2.csv", index=False)
