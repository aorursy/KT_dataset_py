import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train
test
df = pd.concat([train, test], sort=False, axis="rows")
df
df.isnull().sum()
df.dtypes
df["Salutation"] = df["Name"].str.extract("([A-Za-z]+)\.", expand=False)                                  # pandas の文字列を正規表現で分割



df["Salutation"].replace(["Mme", "Ms"], "Mrs", inplace=True)                                              # Mrs に変換

df["Salutation"].replace("Mlle", "Miss", inplace=True)                                                    # Miss に変換

df["Salutation"].replace(["Capt", "Col", "Dr", "Major", "Rev"], "Officer", inplace=True)                  # Officer カテゴリにまとめる

df["Salutation"].replace(["Countess", "Don", "Dona", "Jonkheer", "Lady", "Sir"], "Royalty", inplace=True) # Royalty カテゴリにまとめる
df.loc[df["Name"].str.contains("Mr. ") == True, "Name"] = 0

df.loc[df["Name"].str.contains("Miss. ") == True, "Name"] = 1

df.loc[df["Name"].str.contains("Mrs. ") == True, "Name"] = 2

df.loc[df["Name"].str.contains("Master. ") == True, "Name"] = 3

df.loc[df["Name"].str.contains("Dr. ") == True, "Name"] = 3

df.loc[df["Name"].str.contains("Rev. ") == True, "Name"] = 4

df.loc[df["Name"].str.contains("Col. ") == True, "Name"] = 5

df.loc[df["Name"].str.contains("Major. ") == True, "Name"] = 6

df.loc[df["Name"].str.contains("Jonkheer. ") == True, "Name"] = 7

df.loc[df["Name"].str.contains("Mme. ") == True, "Name"] = 8

df.loc[df["Name"].str.contains("Capt. ") == True, "Name"] = 9

df.loc[df["Name"].str.contains("Ms. ") == True, "Name"] = 10

df.loc[df["Name"].str.contains("Mlle. ") == True, "Name"] = 11

df.loc[df["Name"].str.contains("Don. ") == True, "Name"] = 12

df.loc[df["Name"].str.contains("Countess. ") == True, "Name"] = 13

df.loc[df["Name"].str.contains("Sir. ") == True, "Name"] = 14

df.loc[df["Name"].str.contains("Dona. ") == True, "Name"] = 15



df["Name"].value_counts()
df["Sex"] = df["Sex"].replace({"male":0, "female":1})



df["Sex"].value_counts()
df["Cabin"] = df["Cabin"].fillna(0)

df.loc[df["Cabin"].str.contains("C") == True, "Cabin"] = 1

df.loc[df["Cabin"].str.contains("B") == True, "Cabin"] = 2

df.loc[df["Cabin"].str.contains("D") == True, "Cabin"] = 3

df.loc[df["Cabin"].str.contains("E") == True, "Cabin"] = 4

df.loc[df["Cabin"].str.contains("A") == True, "Cabin"] = 5

df.loc[df["Cabin"].str.contains("F") == True, "Cabin"] = 6

df.loc[df["Cabin"].str.contains("G") == True, "Cabin"] = 7

df.loc[df["Cabin"].str.contains("T") == True, "Cabin"] = 8



df["Cabin"].value_counts()
df["Embarked"] = df["Embarked"].fillna("S")



df["Embarked"].value_counts()
df.loc[df["Embarked"] == "S", "Embarked"] = 0

df.loc[df["Embarked"] == "C", "Embarked"] = 1

df.loc[df["Embarked"] == "Q", "Embarked"] = 2



df["Embarked"].value_counts()
df_age = df.loc[:, ["Age", "Sex", "SibSp", "Parch", "Salutation"]]

df_age = pd.get_dummies(df_age, columns=["Sex", "Salutation"])



df_age_notnull = df_age[df_age["Age"].notnull()] # 学習データ

df_age_null = df_age[df_age["Age"].isnull()]     # 欠損値補完が必要な学習データ



X = df_age_notnull.iloc[:, 1:]                   # 説明変数

y = df_age_notnull.iloc[:, 0]                    # 目的変数



pipeline = Pipeline([("scl", StandardScaler()),  # 標準化を Pipeline により置き換え

                    ("est", RandomForestRegressor(random_state=0))])

pipeline.fit(X, y)

age_predicted = pipeline.predict(df_age_null.iloc[:, 1:])



df.loc[(df["Age"].isnull()), "Age"] = age_predicted
df["Fare"] = df["Fare"].fillna(0)
train = df.iloc[:891]

train
X_test = df.iloc[891:]

X_test = X_test.drop(["Survived", "Ticket", "Salutation"], axis="columns")

X_test
X_train = train.drop(["Survived", "Ticket", "Salutation"], axis="columns")

y_train = train["Survived"].astype(int)



print(X_train.head())

print(y_train.head())
model = RandomForestClassifier(n_estimators=200, random_state=71)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_true = pd.read_csv("../input/titanic/gender_submission.csv", index_col="PassengerId")

y_true
print("正答率： {}".format(accuracy_score(y_true=y_true, y_pred=y_pred)))
submission = pd.DataFrame({"PassengerId":X_test["PassengerId"], "Survived":y_pred})

submission.to_csv("submission.csv", index=False)