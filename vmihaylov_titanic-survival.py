# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df
def enrich(df):
    df = pd.concat([df, pd.get_dummies(df['Sex'], prefix='sex')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='embarked')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['SibSp'], prefix='sibsp')], axis=1)
    df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='pclass')], axis=1)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Age'] = df['Age'] / df['Age'].max()
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Fare'] = df['Fare'] / df['Fare'].max()
    df = pd.concat([df, pd.get_dummies(df['Parch'], prefix='parch')], axis=1)
    df['title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')
    df = pd.concat([df, pd.get_dummies(df['title'], prefix='title')], axis=1)
    df['cabin_letter'] = df['Cabin'].str.extract(r'([A-Za-z]+)')
    df['cabin_letter'].fillna('N', inplace=True)
    df = pd.concat([df, pd.get_dummies(df['cabin_letter'], prefix='cabin')], axis=1)
    return df

df = enrich(df)
pd.options.display.max_columns = None
df
df["Parch"].isna().sum()
def extract_features(df):
    return df[[
        'sex_male', 'sex_female', 
        'pclass_1', 'pclass_2', 'pclass_3', 
        'embarked_C', 'embarked_Q', 'embarked_S', 
        'sibsp_0', 'sibsp_1', 'sibsp_2', 'sibsp_3', 'sibsp_4', 'sibsp_5', 'sibsp_8', 
        'Age', 
        'Fare', 
        'parch_1', 'parch_2', 'parch_3', 'parch_4', 'parch_5', 'parch_6',
        'title_Capt', 'title_Col', 'title_Countess', 'title_Don', 'title_Dr', 'title_Jonkheer', 'title_Lady', 'title_Major', 'title_Master', 'title_Miss', 
        'title_Mlle', 'title_Mme', 'title_Mr', 'title_Mrs', 'title_Ms', 'title_Rev', 'title_Sir',
        'cabin_A', 'cabin_B', 'cabin_C', 'cabin_D', 'cabin_E', 'cabin_F', 'cabin_G', 'cabin_N', 'cabin_T'
    ]]
    
X = extract_features(df)
y = df['Survived']
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)
X_train = X
y_train = y
X_train
df[[col for col in df if col.startswith('sibsp')]].sum()
from tensorflow import keras
from keras.layers import Input, Dense, Dropout
from tensorflow.keras import Sequential


model = Sequential()
model.add(Input(shape=X_train.shape))
model.add(Dense(units=4, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y=y_train, verbose=2, epochs=100)
model.evaluate(x=X, y=y)
df_test = pd.read_csv('../input/test.csv')

df_test = enrich(df_test)
df_test
for col in ['title_Mlle', 'title_Jonkheer', 'title_Countess', 'title_Lady', 'title_Major', 'title_Don', 'title_Mme', 'title_Sir', 'title_Capt', 'cabin_T']:
    df_test[col] = 0
    
X_test = extract_features(df_test)

    
y_pred = model.predict(X_test)
y_pred[y_pred <= 0.5] = 0
y_pred[y_pred > 0.5] = 1

y_pred
df_test['Survived'] = y_pred
df_test['Survived'] = df_test['Survived'].astype(int)

df_test[['PassengerId', 'Survived']]
df_test.to_csv('output.csv', mode='w', columns=['PassengerId','Survived'], index=False)