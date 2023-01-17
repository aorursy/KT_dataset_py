import numpy as np

import pandas as pd
# NotebookでLinuxコマンドを入力する場合は、文頭に!を付けましょう。

!ls ../input
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

gender_submission = pd.read_csv("../input/gender_submission.csv")
gender_submission.head()
train.head()
test.head()
data = pd.concat([train, test], sort=True)
data.head()
print(len(train), len(test), len(data))
data_form = pd.DataFrame(data)

data_form.dtypes
data.isnull().sum()
data['Pclass'].value_counts()
data['Sex'].replace(['male','female'],[0, 1], inplace=True)
data['Embarked'].value_counts()
data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
# Pandasでは、存在しないcolumns（列名）を指定すると勝手に生成される



data['FamilySize'] = data['SibSp'] + data['Parch'] + 1    # 最小値が0になると不都合な場合がある（除算できないとか）ため、よく+1したりする



delete_columns = ['SibSp', 'Parch']

data.drop(delete_columns, axis = 1, inplace = True)    # axisは配列の行か列の指定（0:行、1:列。今回はこの2列を落としたいので'1'）
data.drop('Ticket', axis = 1, inplace = True)
data.drop('Cabin', axis = 1, inplace = True)
delete_columns = ['Name', 'PassengerId']

data.drop(delete_columns, axis = 1, inplace = True)
train = data[:len(train)]    # 配列[:n]：配列の最初からn行目まで

test = data[len(train):]     # 配列[m:]：配列のm行目から最後まで
# Survivedだけ抽出＝正解ラベルのデータ

y_train = train['Survived']



# Survivedi以外のデータ＝説明変数x

X_train = train.drop('Survived', axis = 1)

X_test = test.drop('Survived', axis = 1)
X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)    # インスタンス化を忘れずに！
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred[:20]
sub = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

sub['Survived'] = list(map(int, y_pred))

sub.to_csv("submission.csv", index = False)