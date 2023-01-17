import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
test.head()
train.head()
women = train.loc[train.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print(rate_women)
men = train.loc[train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print(rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



model = RandomForestClassifier(n_estimators = 100, max_depth=5, random_state=242)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived':predictions})

output.to_csv('my_submission.csv', index=False)

print('saved')

