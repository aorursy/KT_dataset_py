import pandas as pd

import numpy as np



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv("../input/train.csv")

test_df  = pd.read_csv("../input/test.csv")



train_df.head()
train_df.info()

print('*************************************************')

test_df.info()
train_df = train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

test_df  = test_df.drop(["Name", "Ticket", "Cabin"], axis=1)
# fill in the missing value of embarked column with the most occurred value "S"

train_df["Embarked"].fillna("S", inplace=True) 

test_df["Embarked"].fillna("S", inplace=True)



# fill in the missing value of Fare column with the median

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



# convert the type of Fare values into int

train_df["Fare"] = train_df["Fare"].astype("int")

test_df["Fare"]  = test_df["Fare"].astype("int")



# acquire the mean, std and number of null values of "Age" column in train_df

mean_age_train = train_df["Age"].mean()

std_age_train  = train_df["Age"].std()

num_nan_age_train = train_df["Age"].isnull().sum()



# acquire the mean, std and number of null values of "Age" column in test_df

mean_age_test = test_df["Age"].mean()

std_age_test  = test_df["Age"].std()

num_nan_age_test = test_df["Age"].isnull().sum()



# generate random age values between (mean-std) and (mean+std)

rand_age_train = np.random.randint(mean_age_train - std_age_train, mean_age_train + std_age_train, 

                              size=num_nan_age_train)

rand_age_test = np.random.randint(mean_age_test - std_age_test, mean_age_test + std_age_test, 

                              size=num_nan_age_test)



# fill in the NaN with the random value generated above

train_df["Age"][np.isnan(train_df["Age"])] = rand_age_train

test_df["Age"][np.isnan(test_df["Age"])] = rand_age_test



# convert the type of age values into int

train_df["Age"] = train_df["Age"].astype("int")

test_df["Age"]  = test_df["Age"].astype("int")
# convert the embarked columns into category number using pandas get_dummies method. 

embark_dummies_train = pd.get_dummies(train_df["Embarked"])

embark_dummies_test  = pd.get_dummies(test_df["Embarked"])



# join the dummied columns

train_df = train_df.join(embark_dummies_train)

test_df  = test_df.join(embark_dummies_test)



# drop the original embarked column

train_df.drop(["Embarked"], axis=1, inplace=True)

test_df.drop(["Embarked"], axis=1, inplace=True)



# Another implement of dealing with embarked column

# train_df.loc[train_df["Embarked"] == "S", "Embarked"] = 0

# train_df.loc[train_df["Embarked"] == "C", "Embarked"] = 1

# train_df.loc[train_df["Embarked"] == "Q", "Embarked"] = 2



# test_df.loc[test_df["Embarked"] == "S", "Embarked"] = 0

# test_df.loc[test_df["Embarked"] == "C", "Embarked"] = 1

# test_df.loc[test_df["Embarked"] == "Q", "Embarked"] = 2



# replace all the occurances of male with the number 0 and female with the number 1

train_df.loc[train_df["Sex"] == "male", "Sex"] = 0

train_df.loc[train_df["Sex"] == "female", "Sex"] = 1



test_df.loc[test_df["Sex"] == "male", "Sex"] = 0

test_df.loc[test_df["Sex"] == "female", "Sex"] = 1



# create dummies values for Pclass column

pclass_dummies_train  = pd.get_dummies(train_df['Pclass'])

pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']



pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']



train_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



train_df = train_df.join(pclass_dummies_train)

test_df  = test_df.join(pclass_dummies_test)
# create a feature named "FamilySize" to combine column "SibSp" and column "Parch"

train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"]

test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"]



# drop the column "SibSp" and "Parch"

train_df.drop(["SibSp", "Parch"], axis=1, inplace=True)

test_df.drop(["SibSp", "Parch"], axis=1, inplace=True)
# generate training and test dataset

X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
# Logistic Regression

lg = LogisticRegression()

lg.fit(X_train, Y_train)

Y_pred = lg.predict(X_test)

lg.score(X_train, Y_train)
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt

%matplotlib inline



predictors = [col for col in train_df.columns if col != "Survived"]



# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(train_df[predictors], train_df["Survived"])



# Get the raw p-values for each feature, and transform from p-values into scores

scores = -np.log10(selector.pvalues_)



# Plot the scores.

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()
from sklearn import cross_validation



# Pick only the six best features.

predictors = ["Fare", "Sex", "C", "S", "Class_1", "Class_3"]



rf = RandomForestClassifier(n_estimators=50, min_samples_split=4, min_samples_leaf=2)

kf = cross_validation.KFold(train_df.shape[0], n_folds=3)

scores = cross_validation.cross_val_score(rf, train_df[predictors], train_df["Survived"], cv=kf)



print(scores.mean())
rf_1 = RandomForestClassifier(n_estimators=100)

rf_1.fit(train_df[predictors], train_df["Survived"])

Y_pred_1 = rf_1.predict(test_df[predictors])

rf_1.score(train_df[predictors], train_df["Survived"])
rf_2 = RandomForestClassifier(n_estimators=100)

rf_2.fit(X_train, Y_train)

Y_pred_2 = rf_2.predict(X_test)

rf_2.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred_2

    })

submission.to_csv('titanic_survival_prediction.csv', index=False)