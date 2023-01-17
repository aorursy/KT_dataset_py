

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.metrics import precision_recall_fscore_support as score

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print(train_data.shape)

train_data.head()

# test_data.shape

# test_data.head()
gender_submission.head()
# Recall ของแต่ละ class

# Precision ของแต่ละ class

# F-Measure ของแต่ละ class

# Average F-Measure ของทั้งชุดข้อมูล

   

def evaluate(y_test, y_pred):

    precision, recall, fscore, support = score(y_test, y_pred)



    print('recall: {}'.format(recall))

    print('precision: {}'.format(precision))

    print('fscore: {}'.format(fscore))

    print('support: {}'.format(support))

    return fscore

# X = train_data[]

# X = train_data[["Pclass", "Sex", "SibSp", "Parch"]]

# y = train_data[["Survived"]]

feature = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Cabin", "Embarked"]

feature = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[feature])

y = train_data["Survived"]

print(X.shape)

print(y.shape)
import numpy as np

from sklearn.model_selection import KFold



kf = KFold(n_splits=5)



for train_index, test_index in kf.split(X):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y.values[train_index], y.values[test_index]

    # Model Here

    

    # Evaluate Model
from sklearn import tree

model = tree.DecisionTreeClassifier()



import numpy as np

from sklearn.model_selection import KFold



kf = KFold(n_splits=5)

fscores = []

loop = 1

for train_index, test_index in kf.split(X):

    print(" *** loop :", loop)

    loop +=1

    X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y.values[train_index], y.values[test_index]

    # Model Here

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    # Evaluate Model

    fscore = evaluate(y_test, y_pred)

    fscores.append(fscore)

    

print(" *** Average F-Measure :", np.array(fscores).mean(axis=0))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)



import numpy as np

from sklearn.model_selection import KFold



kf = KFold(n_splits=5)

fscores = []

loop = 1

for train_index, test_index in kf.split(X):

    print(" *** loop :", loop)

    loop +=1

    X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y.values[train_index], y.values[test_index]

    # Model Here

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    # Evaluate Model

    fscore = evaluate(y_test, y_pred)

    fscores.append(fscore)

    

print(" *** Average F-Measure :", np.array(fscores).mean(axis=0))

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()



import numpy as np

from sklearn.model_selection import KFold



kf = KFold(n_splits=5)

fscores = []

loop = 1

for train_index, test_index in kf.split(X):

    print(" *** loop :", loop)

    loop +=1

    X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y.values[train_index], y.values[test_index]

    # Model Here

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    # Evaluate Model

    fscore = evaluate(y_test, y_pred)

    fscores.append(fscore)

    

print(" *** Average F-Measure :", np.array(fscores).mean(axis=0))

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)



import numpy as np

from sklearn.model_selection import KFold



kf = KFold(n_splits=5)

fscores = []

loop = 1

for train_index, test_index in kf.split(X):

    print(" *** loop :", loop)

    loop +=1

    X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y.values[train_index], y.values[test_index]

    # Model Here

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    # Evaluate Model

    fscore = evaluate(y_test, y_pred)

    fscores.append(fscore)

    

print(" *** Average F-Measure :", np.array(fscores).mean(axis=0))

X_test = pd.get_dummies(test_data[feature])

y_pred = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
output.head()