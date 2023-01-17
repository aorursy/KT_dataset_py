#Importing the data analysis libraries

import numpy as np # linear algebra

import pandas as pd # data processing



#Importing the visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

#Ensuring that we don't see any warnings while running the cells

import warnings

warnings.filterwarnings('ignore') 



#Importing the counter

from collections import Counter



#Importing sci-kit learn libraries that we will need for this project

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

#Reading the data from the given files and creating a training and test dataset

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.sample(10)
train.describe(include="all")
def detect_outliers(dataframe, n, features):

    

    outliers_indices = []

    

    for feature in features:

        

        #determining the upper and lower quartiles

        Quart1 = dataframe[feature].quantile(0.25)

        Quart3 = dataframe[feature].quantile(0.75)

        

        #determining the upper and lower outlier thresholds to remove the outliers

        upper_outlier_threshold = Quart3 + (Quart3 - Quart1) * 1.5

        lower_outlier_threshold = Quart1 - (Quart3 - Quart1) * 1.5

        

        #finding the outliers and saving their indices in the form of a list, according to the given threshold

        feature_outliers_list = dataframe[(dataframe[feature] > upper_outlier_threshold) | (dataframe[feature] < lower_outlier_threshold)].index

        

        #appending the outliers for each feature to the main outliers_indices list

        outliers_indices.extend(feature_outliers_list)

        

    #Selecting features that have more than 2 outliers

    return list(a for a, b in Counter(outliers_indices).items() if b > n)
outliers = detect_outliers(train, 2, ["Age", "SibSp", "Fare", "Parch"])

train.loc[outliers]
train = train.drop(outliers, axis = 0).reset_index(drop = True)
print(pd.isnull(train).sum())
df =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

df.describe(include = "all")
sns.heatmap(df[["Survived","Age","Pclass", "SibSp", "Parch", "Fare"]].corr(), cmap = 'coolwarm', annot = True)
#My function function to visualize and count the values in each category of each feature

def bar_plot(variable):

    

    #This code is used to solve problem when there are no survivors for a category, which causes an error in the display code

    feature_categories = df[variable].sort_values().unique()

    for category in feature_categories:

        temp_series = df["Survived"][df[variable] == category].value_counts(normalize = True)

        if temp_series.shape == (1,):

            temp_series = temp_series.append(pd.Series([0], index=[1]))

        elif temp_series.shape == (0,):

            continue

        print("Fraction of {} = {} who survived:".format(variable, category), temp_series[1])

    #visualize

    sns.barplot(x = df[variable],y = df["Survived"],  data = df).set_title('Fraction Survived With Respect To {}'.format(variable))
bar_plot("Pclass")
bar_plot("Sex")
bar_plot("SibSp")
bar_plot("Parch")
#Filling missing value

#Since Embarked only has 2 missing values, I will use the mode from the Series to determine the missing values

df["Embarked"] = df["Embarked"].fillna(df['Embarked'].mode()[0])

bar_plot("Embarked")
sns.factorplot("Pclass", col="Embarked",  data=train, size=6, kind="count")
df["Cabin"] = df["Cabin"].notnull().astype(int)



pclass1 = df["Cabin"][df["Pclass"] == 1].value_counts()[1]

print("recorded Cabins with pclass1 = 1: {}".format(pclass1))



sns.barplot(x="Pclass", y="Cabin", data=df)
# Exploring Age vs Sex, Parch , Pclass and SibSP

g = sns.factorplot(y="Age",x="Sex",data=df, kind="box")

g = sns.factorplot(y="Age",x="Pclass", data=df, kind="box")

g = sns.factorplot(y="Age",x="Parch", data=df, kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=df, kind="box")

#Using a heatmap to determine the correlation between the remaining features

sns.heatmap(df[["Age","Pclass","SibSp", "Parch"]].corr(), cmap = 'coolwarm', annot = True)
#small function that will remove the missing values in the age column

def fill_age_missing_values(df):

    Age_Nan_Indices = list(df[df["Age"].isnull()].index)



    #for loop that iterates over all the missing age indices

    for index in Age_Nan_Indices:

        #temporary variables to hold SibSp, Parch and Pclass values pertaining to the current index

        temp_Pclass = df.iloc[index]["Pclass"]

        temp_SibSp = df.iloc[index]["SibSp"]

        temp_Parch = df.iloc[index]["Parch"]

        age_median = df["Age"][((df["Pclass"] == temp_Pclass) & (df["SibSp"] == temp_SibSp) & (df["Parch"] == temp_Parch))].median()

        if df.iloc[index]["Age"]:

            df["Age"].iloc[index] = age_median

        if np.isnan(age_median):

            df["Age"].iloc[index] = df["Age"].median()

    return df
#Using the function to remove missing values in both train and test set

df = fill_age_missing_values(df)

df.describe(include="all")
df["Age"].isnull().sum()
df.head()
#Creating a new column("Title") using list comprehension

df["Title"] = pd.Series([name.split(",")[1].split(".")[0].strip() for name in df["Name"]])

df.head()
pd.crosstab(df['Title'], df['Sex'])
# Convert to categorical values Title 

df["Title"] = df["Title"].replace(['Lady', 'the Countess', 'Countess', 'Don', 'Jonkheer', 'Dona', 'Sir'], 'Royals')

df["Title"] = df["Title"].replace(['Col', 'Dr', 'Major', 'Capt'], 'Professionals')

df["Title"] = df["Title"].replace(["Ms", "Mme", "Mlle", "Mrs"], 'Miss')

df["Title"] = df["Title"].replace(['Master', 'Rev'], 'Mas/Rev')

df["Title"] = df["Title"].map({"Mas/Rev": 0, "Miss": 1, "Mr": 2, "Royals": 3, "Professionals": 4})
pd.crosstab(df['Title'], df['Sex'])
sns.factorplot(x="Title",y="Survived",data=df, kind="bar").set_xticklabels(["Master","Miss","Mr","Royals","Professionals"]).set_ylabels("survival probability")
df["Ftotal"] = 1 + df["SibSp"] + df["Parch"]
sns.factorplot(x="Ftotal",y="Survived",data=df, kind="bar").set_ylabels("survival probability")
df["Age"] = df["Age"].astype(int)

df.loc[(df['Age'] <= 2), 'Age Group'] = 'Baby' 

df.loc[((df["Age"] > 2) & (df['Age'] <= 10)), 'Age Group'] = 'Child' 

df.loc[((df["Age"] > 10) & (df['Age'] <= 19)), 'Age Group'] = 'Young Adult'

df.loc[((df["Age"] > 19) & (df['Age'] <= 60)), 'Age Group'] = 'Adult'

df.loc[(df["Age"] > 60), 'Age Group'] = 'Senior'

df["Age Group"] = df["Age Group"].map({"Baby": 0, "Child": 1, "Young Adult": 2, "Adult": 3, "Senior": 4})
df.sample(5)
df.describe(include = "all")
sns.distplot(df["Fare"], color="m", label="Skewness : %.2f"%(df["Fare"].skew())).legend(loc="best")
# Apply log to Fare to reduce skewness distribution

df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

sns.distplot(df["Fare"], color="g", label="Skewness : %.2f"%(df["Fare"].skew())).legend(loc="best")
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df.sample(5)
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

df.sample(5)
passenger_ID = pd.Series(df["PassengerId"], name = "PassengerId")

df = df.drop(["Name", "PassengerId", "SibSp", "Parch", "Age", "Ticket"], axis=1)

df.sample(5)
df = pd.get_dummies(df, columns = ["Title"])

df = pd.get_dummies(df, columns = ["Embarked"])

df = pd.get_dummies(df, columns = ["Pclass"])

df = pd.get_dummies(df, columns = ["Age Group"])
train = df[:train.shape[0]]

test = df[train.shape[0]:].drop(["Survived"], axis = 1)
#StratifiedKFold aims to ensure each class is (approximately) equally represented across each test fold

k_fold = StratifiedKFold(n_splits=5)



X_train = train.drop(labels="Survived", axis=1)

y_train = train["Survived"]



# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)



# Creating objects of each classifier

LG_classifier = LogisticRegression(random_state=0)

SVC_classifier = SVC(kernel="rbf", random_state=0)

KNN_classifier = KNeighborsClassifier()

NB_classifier = GaussianNB()

DT_classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)

RF_classifier = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=0)



#putting the classifiers in a list so I can iterate over there results easily

titanic_classifiers = [LG_classifier, SVC_classifier, KNN_classifier, NB_classifier, DT_classifier, RF_classifier]



#This dictionary is just to grad the name of each classifier

classifier_dict = {

    0: "Logistic Regression",

    1: "Support Vector Classfication",

    2: "K Nearest Neighbor Classification",

    3: "Naive bayes Classifier",

    4: "Decision Trees Classifier",

    5: "Random Forest Classifier",

}



titanic_results = pd.DataFrame({'Model': [],'Mean Accuracy': [], "Standard Deviation": []})



#Iterating over each classifier and getting the result

for i, classifier in enumerate(titanic_classifiers):

    classifier_scores = cross_val_score(classifier, X_train, y_train, cv=k_fold, n_jobs=2, scoring="accuracy")

    titanic_results = titanic_results.append(pd.DataFrame({"Model":[classifier_dict[i]], 

                                                           "Mean Accuracy": [classifier_scores.mean()],

                                                           "Standard Deviation": [classifier_scores.std()]}))
print (titanic_results.to_string(index=False))
from sklearn.model_selection import GridSearchCV



RF_classifier = RandomForestClassifier()





## Search grid for optimal parameters

RF_paramgrid = {"max_depth": [None],

                  "max_features": [1, 3, 10],

                  "min_samples_split": [2, 3, 10],

                  "min_samples_leaf": [1, 3, 10],

                  "bootstrap": [False],

                  "n_estimators" :[100,200,300],

                  "criterion": ["entropy"]}





RF_classifiergrid = GridSearchCV(RF_classifier, param_grid = RF_paramgrid, cv=k_fold, scoring="accuracy", n_jobs= -1, verbose=1)



RF_classifiergrid.fit(X_train,y_train)



RFC_optimum = RF_classifiergrid.best_estimator_



# Best Accuracy Score

RF_classifiergrid.best_score_
IDtest = passenger_ID[train.shape[0]:].reset_index(drop = True)
from sklearn.ensemble import RandomForestClassifier

X_train = train.drop(labels="Survived", axis=1)

y_train = train["Survived"]



# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(test)



RFC_optimum.fit(X_train, y_train)



test_predictions = pd.Series(RFC_optimum.predict(X_test).astype(int), name="Survived")

titanic_results = pd.concat([IDtest, test_predictions], axis = 1)

titanic_results.to_csv('submission.csv', index=False)