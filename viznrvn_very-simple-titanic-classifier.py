import pandas as pd

import numpy as np

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics.classification import accuracy_score, classification_report
data = pd.read_csv('../input/train.csv')
data_len = len(data)

for col in data.columns:

    print('{}: {}'.format(col, 100*sum(data[col].isna())/data_len))
sns.distplot(data['Age'].dropna())
data['Age'] = data['Age'].fillna(data['Age'].mean())
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
data.head()
gender = {

    'female': 0,

    'male': 1

}

data['Sex'] = data['Sex'].map(lambda x: gender.get(x))
data_len = len(data)

for col in data.columns:

    print('{}: {}'.format(col, 100*sum(data[col].isna())/data_len))
X_train, X_test, Y_train, Y_test = train_test_split(data.drop('Survived', axis=1), data['Survived'], test_size=0.2)

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
prediction = classifier.predict(X_test)
'Accuracy: {}'.format(accuracy_score(prediction, Y_test))
print('Classification report')

print(classification_report(Y_test, prediction))