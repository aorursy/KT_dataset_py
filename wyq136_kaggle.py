import pandas as pd

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.info())

print(test.info())
selected_features=['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

X_train = train[selected_features]

X_test = test[selected_features]



y_train = train['Survived']
print(X_train['Embarked'].value_counts())

print(X_test['Embarked'].value_counts())
# 填充缺失值

X_train['Embarked'].fillna('S', inplace=True)

#X_test['Embarked'].fillna('S', inplace=True)



X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)

X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
print(X_train.info())

print(X_test.info())
#X_train.to_dict(orient='record')
# 特征向量化

from sklearn.feature_extraction import DictVectorizer

dict_vec=DictVectorizer(sparse=False)

X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))

dict_vec.feature_names_
X_test=dict_vec.fit_transform(X_test.to_dict(orient='record'))
# 随机深林

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

rfc_y_predict = rfc.predict(X_test)

rfc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':rfc_y_predict})

# 保存成文件

rfc_submission.to_csv('rfc_submission.csv', index=False)

rfc_submission.head()
from xgboost import XGBClassifier

xgbc = XGBClassifier()
# 使用5折交叉验证进行评估

from sklearn.cross_validation import cross_val_score

cross_val_score(rfc, X_train, y_train, cv=5).mean()



cross_val_score(xgbc, X_train, y_train, cv=5)
from xgboost import XGBClassifier



xgbc = XGBClassifier()
from sklearn.cross_validation import cross_val_score



cross_val_score(rfc, X_train, y_train, cv=5).mean()
cross_val_score(xgbc, X_train, y_train, cv=5).mean()
xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)

xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})

xgbc_submission.to_csv('xgbc_submission.csv', index=False)