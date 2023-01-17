import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
train_data.shape
train_data.describe()
train_data.info()
train_data.isnull().sum()
sb.heatmap(train_data.isnull())
#Missing value imputation

def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train_data['Age'] = train_data[['Age', 'Pclass']].apply(impute_age, axis = 1)
sb.heatmap(train_data.isnull())
train_data.drop('Cabin', axis = 1, inplace = True)
train_data.info()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
test_data.shape
test_data.describe()
test_data.info()
test_data.isnull().sum()
sb.heatmap(test_data.isnull())
test_data['Age'] = test_data[['Age', 'Pclass']].apply(impute_age, axis = 1)
sb.heatmap(test_data.isnull())
sb.countplot(x = 'Survived', data = train_data)
sb.countplot(x = 'Survived', hue = 'Sex', data = train_data)
sb.countplot(x = 'Survived', hue = 'Pclass', data = train_data)
plt.figure(figsize = (12, 7))

sb.boxplot(x = 'Survived', y = 'Age', data = train_data)
train_data.head()
train_data = train_data.set_index('PassengerId')

train_data = train_data.drop(columns=['Name', 'Ticket', 'Fare'])

train_data = pd.get_dummies(train_data, columns=['Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked'])
train_data.head()
test_data.head()
test_data = test_data.set_index('PassengerId')

test_data = test_data.drop(columns=['Name', 'Ticket', 'Fare', 'Cabin'])

test_data = pd.get_dummies(test_data, columns=['Pclass','SibSp', 'Parch', 'Sex', 'Embarked'])
test_data = test_data.drop(['Parch_9'], axis=1)

test_data.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data.drop('Survived', axis = 1),

                                                   train_data['Survived'], test_size = 0.20,

                                                   random_state = 101)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(x_train, y_train)

predictions = logmodel.predict(x_test)
logmodel.score(x_test, y_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color = 'orange', label = "ROC")

    plt.plot([0, 1],[0, 1], color = 'darkblue', linestyle = '--' )

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristics (ROC) Curve')

    plt.legend()

    plt.show()
probs = logmodel.predict_proba(x_test)

probs
probs = probs[:, 1]

probs
auc = roc_auc_score(y_test, probs)

print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = test_data.index

submission_df['Survived'] = logmodel.predict(test_data)

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)