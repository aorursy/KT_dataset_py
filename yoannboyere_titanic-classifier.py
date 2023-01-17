import pandas as pd
data = pd.read_csv("/kaggle/input/titanic/train.csv")

data.head()
data.count()
data.Age = data.Age.fillna(data.Age.mean())

data.Embarked = data.Embarked.fillna('S')
dummy_sex = pd.get_dummies(data["Sex"])

dummy_embarked = pd.get_dummies(data["Embarked"])



new_data = pd.concat([data,dummy_sex,dummy_embarked], axis=1)



new_data = new_data.rename(columns={'female':'isFemale'})
clean_data = new_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'male', 'Embarked'], axis=1)
clean_data.head()
clean_data.corr()
from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier
X = clean_data[['Pclass','Fare','isFemale','C','S']]

Y = clean_data.Survived



clf = KNeighborsClassifier()

clf.fit(X, Y)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.count()
#There is one Nan value for the Fare column

test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean())



dummy_sex_test = pd.get_dummies(test_data["Sex"])

dummy_embarked_test = pd.get_dummies(test_data["Embarked"])



new_test_data = pd.concat([test_data,dummy_sex_test,dummy_embarked_test], axis=1)



new_test_data = new_test_data.rename(columns={'female':'isFemale'})



X_test = new_test_data[['Pclass','Fare','isFemale','C','S']]
predicted_data = clf.predict(X_test)
predicted_data
final_data = pd.DataFrame(

                            {

                                'PassengerId':test_data.PassengerId,

                                'Survived':predicted_data

                            },

                            columns=['PassengerId','Survived']

                        )
final_data.to_csv('my_submission.csv', index=False)

print("File saved !")