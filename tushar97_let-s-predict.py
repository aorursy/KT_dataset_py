import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
Train_data = pd.read_csv('../input/train.csv')
Test_data = pd.read_csv('../input/test.csv')
Train_data.info()
Train_data['Survived'].value_counts(normalize = True)
Train_data['Survived'].value_counts(normalize = True)
#now we will convert non numeric values to numeric values and remove the NAN values
Train_data.loc[Train_data['Sex'] == 'male', 'Sex'] = 0
Train_data.loc[Train_data['Sex'] == 'female', 'Sex'] = 1
Train_data['Age'].fillna(Train_data['Age'].mean(), inplace = True)
#Now we will fill all the empty Embarked feature values
print(Train_data["Embarked"].unique())
Train_data["Embarked"] = Train_data["Embarked"].fillna("S")
# now we will convert non numeric values in embarked into numeric ones
Train_data.loc[Train_data['Embarked'] == 'S','Embarked'] = 0
Train_data.loc[Train_data['Embarked'] == 'C','Embarked'] = 1
Train_data.loc[Train_data['Embarked'] == 'Q','Embarked'] = 2
Train_data.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked"]
X_train, X_test, y_train, y_test = train_test_split(Train_data[predictors], Train_data["Survived"])
Forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
Forest.fit(X_train, y_train)
print("Random Forest score: {0:.2}".format(Forest.score(X_test, y_test)))
#now we will do the same thing on test set
Test_data.loc[Test_data['Sex'] == 'male', 'Sex'] = 0
Test_data.loc[Test_data['Sex'] == 'female', 'Sex'] = 1

Test_data['Age'].fillna(Test_data['Age'].mean(), inplace = True)
Test_data['Fare'].fillna(Test_data['Fare'].mean(), inplace = True)

#now plotting the graps with Features and pick the best feature as good contributor to model
# we will take the main five big values

plt.bar(np.arange(len(predictors)), Forest.feature_importances_)
plt.xticks(np.arange(len(predictors)), predictors)

predictors = ["Sex", "Fare", "Pclass", "Age", "SibSp"]
clf = RandomForestClassifier(n_estimators=100,
                             criterion='gini',
                             max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             random_state=0)
clf.fit(Train_data[predictors], Train_data["Survived"])
prediction = clf.predict(Test_data[predictors])
Submission = pd.DataFrame({"PassengerId": Test_data["PassengerId"], "Survived": prediction})
Submission.to_csv("submission.csv", index=False)