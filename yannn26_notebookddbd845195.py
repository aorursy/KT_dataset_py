### import data

import numpy as np

import pandas as pd

train_origin = pd.read_csv('../input/train.csv')

test_origin = pd.read_csv('../input/test.csv')

train = train_origin

print(train.head())
### process raw data(1): male-0, female-1; Embarked(C,Q,S)-(0,1,2)

train = train.replace("C",0).replace("Q",1).replace("S",2)

train = train.replace("male",0).replace("female",1)

train.head()
### preprocess raw data(2):

# 1.the null value of Age to mean value

# 2.delete the ROW where "Embarked" is null

print('### Before Preprocessing: ### \n',train.isnull().sum(),'\n')

train.Age = train.fillna(train.Age.mean)

train = train.dropna(subset=['Embarked'])

print('### After Preprocessing: ### \n',train.isnull().sum())
### prepare for the data: 70% is data_learn, 30% is data_classify

data =  train[['Survived','PassengerId','Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]

from sklearn.model_selection import train_test_split

data_learn, data_test = train_test_split(data, test_size=0.3, random_state=123)

print ("Dimension of learn data {}".format(data_learn.shape))

print ("Dimension of test data {}".format(data_test.shape))
y1_data_learn = data_learn['Survived']

x1_data_learn = data_learn[['PassengerId','Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]

y2_data_test = data_test['Survived']

x2_data_test = data_test[['PassengerId','Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]

### Standardization

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

x1_data_learn_std = stdsc.fit_transform(x1_data_learn)

x2_data_test_std = stdsc.transform(x2_data_test)

print("Dimension of learn data {}".format(x1_data_learn_std.shape))

print("Dimension of test data {}".format(x2_data_test_std.shape))
### Using SVC

from sklearn import svm, model_selection

from sklearn.metrics import confusion_matrix

parameters = [{'kernel':('linear', 'rbf'), 'C':np.logspace(-4, 4, 6), 'gamma':np.logspace(-4, 4, 6)}]

#parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3, 1e-4],'C': [1, 10, 100, 1000]},

#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf = model_selection.GridSearchCV(svm.SVC(), parameters, n_jobs = -2)

clf.fit(x1_data_learn_std, y1_data_learn)

print("mean score for cross-validation:\n")

for params, mean_score, all_scores in clf.grid_scores_:

    print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))