import pandas as pd

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
def sex_map(sex):

    if sex == 'female':

        return 0

    else:

        return 1
combine = [train_df, test_df]

for d in combine:

    d['Sex'] = d['Sex'].map(sex_map)
fields = ['PassengerId','Sex']

X_train = train_df.select(lambda f : f in fields, axis=1)

Y_train = train_df['Survived']

X_test = test_df.select(lambda f : f in fields, axis=1)

X_train.shape,X_test.shape,Y_train.shape
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic-output.csv', index=False)