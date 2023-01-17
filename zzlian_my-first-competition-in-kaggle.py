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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
print(print(train_data.shape))
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
print(test_data.shape)
test_data.head()
import plotly.express as px
sur = train_data.groupby('Survived')['PassengerId'].count().reset_index(name = 'count')
fig = px.pie(sur, values='count', names='Survived', title='Surviver Spread')
fig.show()
pclass = train_data.groupby("Pclass")['PassengerId'].count().reset_index(name = 'count')
fig = px.pie(pclass, values='count', names='Pclass', title='Pclass Spread')
fig.show()
sex = train_data.groupby("Sex")['PassengerId'].count().reset_index(name = 'count')
fig = px.pie(sex, values='count', names='Sex', title='Sex Spread')
fig.show()
counts, bins = np.histogram(train_data.Age, bins=range(0, 80, 5))
bins = 0.5 * (bins[:-1] + bins[1:])
fig = px.bar(x=bins, y=counts, labels={'x':'Age', 'y':'count'})
fig.show()
sib = train_data.groupby("SibSp")['PassengerId'].count().reset_index(name = 'count')
fig = px.bar(sib, x='SibSp', y='count', labels={'x':'Siblings', 'y':'count'})
fig.show()
par = train_data.groupby("Parch")['PassengerId'].count().reset_index(name = 'count')
fig = px.bar(par, x='Parch', y='count', labels={'x':'Siblings', 'y':'count'})
fig.show()
train_data.Fare.value_counts()
counts, bins = np.histogram(train_data.Fare, bins=range(0, 600, 10))
bins = 0.5 * (bins[:-1] + bins[1:])
fig = px.bar(x=bins, y=counts, labels={'x':'Fare', 'y':'count'})
fig.show()
emb = train_data.groupby("Embarked")['PassengerId'].count().reset_index(name = 'count')
fig = px.pie(emb, values='count', names='Embarked', title='Embarked Spread')
fig.show()
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X
from sklearn.ensemble import RandomForestClassifier
features = ["Pclass", "Sex", "SibSp", "Parch"]
y_train = train_data["Survived"][:750]
X_train = pd.get_dummies(train_data[features])[:750]

y_validate = train_data["Survived"][750:]
x_validate = pd.get_dummies(train_data[features])[750:]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(x_validate)
result = y_pred == np.array(y_validate)
the_right_answer_num = len(result[result])
the_wrong_answer_num = len(result) - the_right_answer_num

print("the right answer num is : ", the_right_answer_num)
print("the wrong answer num is : ", the_wrong_answer_num)
print("the score is : ", the_right_answer_num / len(result))
X_test = pd.get_dummies(test_data[features])
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
