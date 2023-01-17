import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier


%pip install ppscore
import ppscore as pps
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
print("length of train_data : {}".format(len(train_data)))
print("length of test_data : {}".format(len(test_data)))

# check if it is balanced
train_data.Survived.value_counts()
display(train_data.head(20))
print("NULL VALUES IN TRAINING DATA")
print(train_data.isna().sum())
print("NULL VALUES IN TEST DATA")
print(test_data.isna().sum())

# check, is there any correlation between first letter of cabin and surviving
# null values are replaced with the letter "N"
for df in [train_data, test_data]:
    df["Cabin"].loc[df["Cabin"].notnull()] = df["Cabin"].loc[df["Cabin"].notnull()].apply(lambda x: x[0])
    df["Cabin"].fillna("N", inplace = True)

sns.barplot(x = "Cabin", y = "Survived", data= train_data.sort_values("Cabin"), orient = "v")

train_data.Cabin.value_counts()
for df in [train_data, test_data]:
    df["Cabin"].loc[df["Cabin"] != "N"] = 1
    df["Cabin"].loc[df["Cabin"] == "N"] = 0
sns.barplot(x = "Cabin", y = "Survived", data= train_data, orient = "v")

train_data.Cabin.value_counts()

for df in [train_data, test_data]:
    df.replace("male", 0, inplace = True)
    df.replace("female", 1, inplace = True)
for df in [train_data, test_data]:
    df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    
titles = np.unique(train_data["Title"])

for title in titles: 
    count = (train_data["Title"] == title).sum()
    temp = train_data[train_data["Title"] == title]["Survived"]
    surviving_perc = temp.sum() / len(temp)
    print("{} occured {} times -- surviving percentage is : {}".format(title, count, surviving_perc))
for df in [train_data, test_data]:
    df.replace(["Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Major","Rev"], "Others", inplace = True)
    df.replace(["Miss", "Mlle", "Mme", "Mrs", "Ms"], "Mrs", inplace = True)
    df.replace(["Lady", "Sir", "the Countess"], "Royal", inplace = True)

sns.barplot(x = "Title", y = "Survived", data= train_data, orient = "v")

train_data.Title.value_counts()

train_data["Family Number"] = train_data["SibSp"] + train_data["Parch"]
test_data["Family Number"] = test_data["SibSp"] + train_data["Parch"]

test_data["Alone"] = np.nan
train_data["Alone"] = np.nan

for df in [test_data, train_data]:
    df["Alone"].loc[df["Family Number"] == 0 ] = 1
    df["Alone"].loc[df["Family Number"] != 0 ] = 0
    
sns.barplot(x = "Alone", y = "Survived", data= train_data, orient = "v")

combine = pd.concat([train_data, test_data])
tickets = np.unique(combine["Ticket"])
combine["GroupSize"] = 0
# combine["Same Ticket Surv Perc"] = 0

for ticket in tickets: 
    group_size = len(combine.loc[combine["Ticket"] == ticket,"Survived"])
    combine.loc[combine["Ticket"] == ticket, "GroupSize"] = group_size

combine["Family Size"] = np.nan
combine.loc[(combine["Family Number"] == 0), "Family Size"] = "Small"
combine.loc[(combine["Family Number"] > 0) & (combine["Family Number"] < 4), "Family Size" ] = "Medium"
combine.loc[(combine["Family Number"] >= 4), "Family Size"] = "Big"

combine["Embarked"].value_counts()
combine["Embarked"].fillna("S", inplace = True)
combine['Fare'] = combine['Fare'].fillna(combine.groupby('Pclass')['Fare'].transform('mean'))
train_data = combine.loc[combine["Survived"].notnull()]
fig, axs = plt.subplots(3,3, figsize = (15,10))

sns.barplot(x="Pclass", y="Survived", data=train_data, orient="v", ax=axs[0][0])
sns.barplot(x="Sex", y="Survived", data=train_data, orient="v", ax=axs[0][1])
sns.barplot(x="Family Size", y="Survived", data=train_data, orient="v", ax=axs[0][2])
sns.barplot(x="Cabin", y="Survived", data=train_data, orient="v", ax=axs[1][0])
sns.barplot(x="Embarked", y="Survived", data=train_data, orient="v",ax=axs[1][1])
sns.barplot(x="Title", y="Survived", data=train_data, orient="v",ax=axs[1][2])
sns.barplot(x="Parch", y="Survived", data=train_data, ax = axs[2][0])
sns.barplot(x="SibSp", y="Survived", data=train_data, ax = axs[2][1])
sns.barplot(x="Alone", y="Survived", data=train_data, ax = axs[2][2])

plt.figure(figsize=(10,6))
sns.catplot(x="Survived", y="Fare", data=train_data)


plt.figure(figsize=(10,6))
sns.catplot(x="Survived", y="Age", data=train_data)

encoded_columns = ["Title", "Embarked", "Family Size"]

encoded_features = []
for column in encoded_columns:
    encoded_arr = OneHotEncoder().fit_transform(combine[column].values.reshape(-1, 1)).toarray()
    n = combine[column].nunique()
    cols = ['{}_{}'.format(column, n) for n in range(1, n + 1)]
    encoded_df = pd.DataFrame(encoded_arr, columns=cols)
    encoded_df.index = combine.index
    encoded_features.append(encoded_df)
combine = pd.concat([combine, *encoded_features[:3]], axis=1)

drop_columns = encoded_columns + ["PassengerId", "Name","Ticket"]
combine.drop(columns = drop_columns , inplace = True)

survived = combine["Survived"]
combine.drop(columns = ["Survived"], inplace = True)
imputer = KNNImputer()
imputer.fit(combine)
combine[:] = imputer.transform(combine)
combine["AgeGroup"] = "old"
combine.loc[combine["Age"] <= 15, "AgeGroup"] = "children"
combine.loc[(combine["Age"] > 15) & (combine["Age"] <= 25), "AgeGroup"] = "young"
combine.loc[(combine["Age"] > 25) & (combine["Age"] <= 35), "AgeGroup"] = "adult"
combine.loc[(combine["Age"] > 35) & (combine["Age"] <= 55), "AgeGroup"] = "mature"

combine["AgeGroup"].value_counts()
combine["Survived"] = survived
train_data = combine.loc[combine["Survived"].notnull()]
test_data = combine.loc[combine["Survived"].isna()]

sns.barplot(x = "AgeGroup", y = "Survived", data= train_data, orient = "v")

encoded_arr = OneHotEncoder().fit_transform(combine["AgeGroup"].values.reshape(-1, 1)).toarray()
encoded_features = []
n = combine["AgeGroup"].nunique()
cols = ['{}_{}'.format("AgeGroup", n) for n in range(1, n + 1)]
encoded_df = pd.DataFrame(encoded_arr, columns=cols)
encoded_df.index = combine.index
encoded_features.append(encoded_df)
combine = pd.concat([combine, *encoded_features[:1]], axis=1)

combine.drop(columns = ["AgeGroup"], inplace = True)

scaler = MinMaxScaler()
scaler.fit(combine)
combine[:] = scaler.transform(combine)

combine["Survived"] = survived
train_data = combine.loc[combine["Survived"].notnull()]
test_data = combine.loc[combine["Survived"].isna()]

figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
Var_Corr = train_data.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')

matrix_df = pps.matrix(train_data)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
matrix_df = matrix_df.apply(lambda x: round(x, 2)) # Rounding matrix_df's values to 0,XX

sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.75, annot=True)
combine.drop(columns = ["Age","Parch", "SibSp", "GroupSize", "Family Number", "Title_1","Title_4", "Title_5", "Embarked_2","AgeGroup_1", "AgeGroup_3", "AgeGroup_4", "AgeGroup_5"], inplace = True)

train_data = combine.loc[combine["Survived"].notnull()]
test_data = combine.loc[combine["Survived"].isna()]

X = train_data.drop(columns = ["Survived"])
y = train_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42)
#construct pipeline

feature_engineering = [
    None,
    PCA(n_components = round(X.shape[1]*0.9)),
    PCA(n_components = round(X.shape[1]*0.8)),
    PCA(n_components = round(X.shape[1]*0.7)),
    PCA(n_components = round(X.shape[1]*0.6)),
    PCA(n_components = round(X.shape[1]*0.5)),
    PCA(n_components = round(X.shape[1]*0.3)),
    PCA(n_components = round(X.shape[1]*0.1)),
    PolynomialFeatures(degree = 1),
    PolynomialFeatures(degree = 2),
    PolynomialFeatures(degree = 3)
]

params = [
    {
        "feature_eng": feature_engineering,
        "clf": [GaussianNB()]
    },
    {
        "feature_eng": feature_engineering,
        "clf": [SGDClassifier()],
        "clf__loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        "clf__penalty": ["l1", "l2", "elasticnet"]
    },   
    {
        "feature_eng": feature_engineering,
        "clf": [DecisionTreeClassifier()],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [3,4,5,6,7,10,15]        
    },   
    {
        "feature_eng": feature_engineering,
        "clf": [RandomForestClassifier()],
        "clf__max_depth": [3,4,5,6,7,10,15]
    }, 
    {
        "feature_eng": feature_engineering,
        "clf": [SVC()],
        "clf__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"]
    }, 
    {
        "feature_eng": feature_engineering,
        "clf": [KNeighborsClassifier()],
        "clf__n_neighbors": [3,5,10,15,30,40]
    }, 
    {
        "feature_eng": feature_engineering,
        "clf": [GradientBoostingClassifier()],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__n_estimators": [70,100,120,150]
    }, 
    {
        "feature_eng": feature_engineering,
        "clf": [XGBClassifier()],
        "clf__n_estimators": [5,10,30,50,70,100,120]
    }, 
]
pipe = Pipeline(steps=[('feature_eng', None),('clf', DecisionTreeClassifier())])
clf = RandomizedSearchCV(pipe, params, n_iter=200, scoring="accuracy", refit='accuracy',n_jobs=-1, cv=5, random_state=42)
clf.fit(X_train, y_train)

pd.DataFrame(clf.cv_results_).sort_values(by = ["mean_test_score"], ascending = False).iloc[:10]
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

ids = pd.read_csv("/kaggle/input/titanic/test.csv")["PassengerId"]
test_data.drop(columns = ["Survived"], inplace = True)
pred = clf.predict(test_data)
df = pd.DataFrame({"PassengerId": ids, "Survived": pred.astype(int)})

df.to_csv('submissions.csv')