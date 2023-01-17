import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



from sklearn import datasets, linear_model

from sklearn.model_selection  import  train_test_split

from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')        
train = pd.read_csv(os.path.join(dirname, 'train.csv'))

test = pd.read_csv(os.path.join(dirname, 'test.csv'))

submission = pd.read_csv(os.path.join(dirname, 'gender_submission.csv'))
train.head()
test.head()
def get_title(name):

    title = re.search(' ([A-Za-z]+)\.', name)

    if title:

        return title.group(1)

    return '0'
def process_data(df):

    #preprocessing based https://www.kaggle.com/paulorzp/titanic-gp-model-training

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['Alone'] = 0

    df.loc[df['FamilySize'] == 1, 'Alone'] = 1

    df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    

    df['Cabin'].fillna('0', inplace=True)

    df['Cabin'] = df['Cabin'].str[0]

    df['Cabin'] = df['Cabin'].map({'0':0, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8})

    df['Cabin'] = df['Cabin'].astype(int)    



    title_mapping = {'0':0, 'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

    title_replace = {'0':'0', 'Mlle': 'Rare', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Rare', 'Rev': 'Mr',

                     'Don': 'Mr', 'Mme': 'Rare', 'Jonkheer': 'Mr', 'Lady': 'Mrs',

                     'Capt': 'Mr', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare'}

    df['Title'] = df['Name'].map(get_title)

    df.replace({'Title': title_replace}, inplace=True)

    df['Title'] = df['Title'].map(title_mapping)

    df['Title'].fillna('0', inplace=True)

    df['Title'] = df['Title'].astype(int)

    

    df.fillna(0, inplace=True)

    return df
train = process_data(train)

test = process_data(test)
train.head()
train = train.drop(columns=['Name', 'Ticket'])

test = test.drop(columns=['Name', 'Ticket'])
train.info()
train.describe().T
fig = plt.subplots(figsize = (10,10))

sb.set(font_scale=1.5)

sb.heatmap(train.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})

plt.show()
sb.countplot(x='Survived',data=train)
X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['Survived']), pd.DataFrame(train.Survived))
y_test.describe().T
X_train.describe().T
train.head()
lm = linear_model.LinearRegression()

lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
print('Coefficients: \n', lm.coef_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print('Total True: %d' % lm.predict(X_test).round().sum())
y_test['Survived'] = lm.predict(X_test).round()

y_test.classe = y_test['Survived'].astype(int)
sb.countplot(x='Survived',data= y_test)
submission.head()
submission['Survived'] = lm.predict(test).round().astype(int)
submission.to_csv('Submission.csv', index=False)