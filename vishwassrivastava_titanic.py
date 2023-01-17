import pandas as pd
import numpy as np

# read data sets
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
train_data.head()
print(train_data.shape)
train_data.info()
# check null in train data
train_data.isnull().sum()
bool_series = pd.isna(train_data['Age'])
train_data.Age.unique()
bool_series.unique()
train_data = train_data[~ bool_series]
train_data.Cabin.unique()
### checking percentage of missing values
round((train_data.isnull().sum()/ len(train_data)) * 100, 2)
# Since cabin has 74% data missing we can drop this column
train_data.drop(columns=["Cabin"], inplace=True)
round((train_data.isnull().sum()/ len(train_data)) * 100, 2)
bool_series = pd.isna(train_data['Embarked'])
bool_series.unique()
train_data = train_data[~ bool_series]
round((train_data.isnull().sum()/ len(train_data)) * 100, 2)
# loss of data due to cleaning
round( (1-len(train_data)/891)*100 ,2)
train_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
