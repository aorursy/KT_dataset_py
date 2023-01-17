import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

import re

print(os.listdir("../input"))
df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

feature_list = []

df.head()
df.drop("PassengerId", 1, inplace=True)

df.head()
df.info()
plt.figure(figsize=(6, 6))

plt.pie(df["Pclass"].value_counts(), labels=df["Pclass"].value_counts().index, autopct='%1.1f%%')

plt.title("Pclass Percentages")

plt.show()
print(df.groupby(by='Pclass').mean()["Survived"])
df["Pclass"].corr(df["Survived"])
feature_list.append('Pclass')
df["Age"].isna().sum()
print("{:.2f}%".format((df["Age"].isna().sum() / len(df.index)) * 100))
age_description = df["Age"].describe()

age_description
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

plt.hist(x=df["Age"].dropna())

plt.xlabel("Age")

plt.ylabel("Count")

plt.title("Age Histogram")



plt.subplot(1, 2, 2)

sns.boxplot(x=df["Age"])

plt.title("Age Boxplot")

plt.show()
df["Age"].corr(df["Survived"])
temp_age = df["Age"].fillna(value=df["Age"].mean())
print("NaN count:", temp_age.isna().sum())



plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)

plt.hist(x=temp_age.dropna())

plt.xlabel("Age")

plt.ylabel("Count")

plt.title("Age Histogram")



plt.subplot(1, 2, 2)

sns.boxplot(x=temp_age)

plt.title("Age Boxplot")

plt.show()
temp_age.corr(df["Survived"])
df["Cat_Age"] = pd.cut(temp_age, bins=[0, 18, 60, 100], labels=[0, 1, 2]) # 0:child, 1:adult, 2:old

df.drop("Age", axis=1, inplace=True)
sns.barplot(x=df["Cat_Age"], y=df["Survived"])

plt.xticks(np.arange(3), ("Child", "Adult", "Old"))

plt.show()
df["Cat_Age"].corr(df["Survived"])
feature_list.append('Cat_Age')

feature_list
plt.figure(figsize=(6, 6))

plt.pie(df["Sex"].value_counts(), labels=df["Sex"].value_counts().index, autopct='%1.1f%%')

plt.title("Sex Percentages")

plt.show()
df["Sex"] = df["Sex"].apply(lambda x: 0 if x == "male" else 1) # 0: male, 1: female
df["Sex"].corr(df["Survived"])
feature_list.append('Sex')

feature_list
df["Alone"] = df.apply(lambda row: 1 if row["SibSp"] + row["Parch"] == 0 else 0, axis=1) # 1: Alone, 0: Not alone
df["Alone"].corr(df["Survived"])
feature_list.append('Alone')

feature_list
df["Ticket"].isna().sum()
df["Ticket"].nunique()
df.drop("Cabin", 1, inplace=True)

test_df.drop("Cabin", 1, inplace=True)

df.head()
df["Fare"] = pd.cut(df["Fare"], 3, labels=[0, 1, 2]) # 0: Low, 1: Medium, 2: High



test_df["Fare"].fillna(test_df["Fare"].mean(), inplace=True)

test_df["Fare"] = pd.cut(test_df["Fare"], 3, labels=["Low", "Medium", "High"])

df.head()
df["Fare"].corr(df["Survived"])
feature_list.append("Fare")

feature_list
df.head()
import sklearn.preprocessing
embarked_encoder = sklearn.preprocessing.LabelEncoder()



df["Embarked"].fillna("S", inplace=True)

df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])

df.head()
df["Embarked"].corr(df["Survived"])
feature_list.append("Embarked")

feature_list
import sklearn.model_selection

import sklearn.ensemble

import sklearn.metrics
X, y = df[feature_list], df["Survived"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
model_rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)

model_gb = sklearn.ensemble.GradientBoostingClassifier()
model_rf.fit(X_train, y_train)

model_gb.fit(X_train, y_train)
yhat_rf = model_rf.predict(X_test)

yhat_gb = model_gb.predict(X_test)
accuracy_rf = sklearn.metrics.accuracy_score(y_test, yhat_rf)

recall_rf = sklearn.metrics.recall_score(y_test, yhat_rf)

precision_rf = sklearn.metrics.precision_score(y_test, yhat_rf)



print("Random Forest Accuracy:", accuracy_rf)

print("Random Forest Recall:", recall_rf)

print("Random Forest Precision:", precision_rf)

print()



accuracy_gb = sklearn.metrics.accuracy_score(y_test, yhat_gb)

recall_gb = sklearn.metrics.recall_score(y_test, yhat_gb)

precision_gb = sklearn.metrics.precision_score(y_test, yhat_gb)



print("Gradient Boosting Accuracy:", accuracy_gb)

print("Gradient Boosting Recall:", recall_gb)

print("Gradient Boosting Precision:", precision_gb)

print()