import pandas as pd

from sklearn.naive_bayes import GaussianNB



X_train = pd.read_csv('../input/titanic/train.csv',usecols=['PassengerId','Pclass','Sex','Age']).fillna('0.0')

X_train_enc = pd.get_dummies(X_train, columns=['Sex']) 

y_train = pd.read_csv('../input/titanic/train.csv')['Survived']



model = GaussianNB()

model.fit(X_train_enc, y_train)



X_test = pd.read_csv('../input/titanic/test.csv',usecols=['PassengerId','Pclass','Sex','Age']).fillna('0.0')

X_test_enc = pd.get_dummies(X_test, columns=['Sex'])

y_model = model.predict(X_test_enc)



X_test['Survived'] = y_model

df = X_test[['PassengerId','Survived']]

df.to_csv('titanic_test_results.csv',index=False)