import pandas as pd

import warnings

warnings.filterwarnings('ignore')
%ls ../input
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_gender_submission = pd.read_csv('../input/gender_submission.csv')
genders = {'male': 0, 'female': 1} # 辞書を作成

# Sexをgendersを用いて変換

df_train['Sex'] = df_train['Sex'].map(genders)

df_test['Sex'] = df_test['Sex'].map(genders)



# ダミー変数化

df_train = pd.get_dummies(df_train, columns=['Embarked'])

df_test = pd.get_dummies(df_test, columns = ['Embarked'])



# 不要な列の削除

df_train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
X_train = df_train.iloc[:, 1:]

Y_train = df_train['Survived']
X_train.to_csv("train_X_data_after_preprocessing.csv", index=False)

Y_train.to_csv("train_Y_data_after_preprocessing.csv", index=False)