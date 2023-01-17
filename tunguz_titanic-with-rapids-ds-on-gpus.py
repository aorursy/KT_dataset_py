%%time

# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)

import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cupy, cudf, cuml

from cuml.linear_model import LogisticRegression

from cuml.ensemble import RandomForestClassifier

from cuml.svm import SVC

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = cudf.read_csv('/kaggle/input/titanic/train.csv')

test = cudf.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
train = train.drop(columns= ['Name','Ticket','Cabin'])

test = test.drop(columns= ['Name','Ticket','Cabin'])
train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)

train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)

train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)

train['Gender'] = (train['Sex'] == 'male').astype(int)
test['Embarked_S'] = (test['Embarked'] == 'S').astype(int)

test['Embarked_C'] = (test['Embarked'] == 'C').astype(int)

test['Embarked_Q'] = (test['Embarked'] == 'Q').astype(int)

test['Gender'] = (test['Sex'] == 'male').astype(int)
train = train.drop(columns= ['Embarked','Sex',])

test = test.drop(columns= ['Embarked','Sex',])

train.fillna(0,inplace=True)

test.fillna(0,inplace=True)
X = train.drop(columns = ['Survived'])

y = train['Survived'].astype('int32')
model = RandomForestClassifier(n_estimators = 100, max_depth = 6)

model.fit(X, y) 
yhat_train = model.predict(X, predict_model = 'CPU')

yhat_test = model.predict(test, predict_model = 'CPU') 
print(sum(y == yhat_train) / len(y)) 
submission = cudf.DataFrame({'PassengerId': test.PassengerId, 'Survived': yhat_test})

submission.to_csv('submission.csv', index = False) 