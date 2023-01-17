# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import libraries

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#We will use different classifiers and then try Voting Classifier to see if it helps in increasing score.

#Classfiers used in _v1
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

#Model_selection

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

#set data visualization stying

sns.set(style='white', context = 'notebook', palette='deep')
#Load data

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
IDtest = test_data["PassengerId"]
#Outlier Detection

from collections import Counter

def get_outliers(df,n,features):
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        #1st Quartile
        Q1 = np.percentile(df[col],25)
        #3rd Quartile
        Q3 = np.percentile(df[col],75)
        #Inter-Quartile Range
        IQR = Q3 - Q1
        
        #Outliers Range        
        outliers_boundary = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        list_outlier_cols = df[(df[col] < Q1 - outliers_boundary) | (df[col] > Q3 + outliers_boundary)].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(list_outlier_cols)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    more_than_two_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return more_than_two_outliers   

#Function takes 3 parameters - DataFrame,number of outliers you want to check in an observations, feature columns 
Outliers_to_drop = get_outliers(train_data, 2, ["Age","SibSp","Parch","Fare"])
        
train_data.loc[Outliers_to_drop]
#Drop outliers

train_data = train_data.drop(Outliers_to_drop, axis = 0)
#Concate datasets
full_dataset = pd.concat(objs=[train_data,test_data], axis = 0).reset_index(drop=True)
full_dataset.info()
full_dataset.isnull().sum()
#In Version V1, we didn't look into SibSp and Parch features, so let's start with them.

train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Let's do the similar analysis for Parch

train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#As we can see large family have less survival rate, I am going to make a new feature which we can call family size.
#Family size = SibSp + Parch + Individual

full_dataset["FamilySize"] = full_dataset["SibSp"] + full_dataset["Parch"] + 1

full_dataset.head()
#Also I am going to drop Cabin Variable as Cabin has more than 70% null

full_dataset.drop(["Cabin"],axis =1, inplace=True)
full_dataset.head()
#Embarked - Since only two values are missing and that is in training data, we can fill it with highest occuring value as we did in V1.
full_dataset["Embarked"].fillna('S', inplace=True)
#Fare can be filled with median value as well.

fare_median = full_dataset["Fare"].median()
fare_median

full_dataset["Fare"].fillna(fare_median, inplace=True)
#Let's do one hot encoding like V1.

from sklearn.preprocessing import LabelEncoder

le_pClass = LabelEncoder()
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
full_dataset['PClass_encoded'] = le_pClass.fit_transform(full_dataset.Pclass)
full_dataset['Sex_encoded'] = le_sex.fit_transform(full_dataset.Sex)
full_dataset['Embarked_encoded'] = le_embarked.fit_transform(full_dataset.Embarked)
full_dataset.head()
#One hot encoding for categorical columns (PClass, Sex, Embarked)

from sklearn.preprocessing import OneHotEncoder

pClass_ohe = OneHotEncoder()
sex_ohe = OneHotEncoder()
embarked_ohe = OneHotEncoder()

Xp =pClass_ohe.fit_transform(full_dataset.PClass_encoded.values.reshape(-1,1)).toarray()
Xs =sex_ohe.fit_transform(full_dataset.Sex_encoded.values.reshape(-1,1)).toarray()
Xe =embarked_ohe.fit_transform(full_dataset.Embarked_encoded.values.reshape(-1,1)).toarray()
#Add back to original dataframe

train_dataOneHot = pd.DataFrame(Xp, columns = ["PClass_"+str(int(i)) for i in range(Xp.shape[1])])
full_dataset = pd.concat([full_dataset, train_dataOneHot], axis=1)

train_dataOneHot = pd.DataFrame(Xs, columns = ["Sex_"+str(int(i)) for i in range(Xs.shape[1])])
full_dataset = pd.concat([full_dataset, train_dataOneHot], axis=1)

train_dataOneHot = pd.DataFrame(Xe, columns = ["Embarked_"+str(int(i)) for i in range(Xe.shape[1])])
full_dataset = pd.concat([full_dataset, train_dataOneHot], axis=1)
full_dataset.head()
#First, let us take age.
#Let us see how other features are correlated with age and if we can impute age as per other features.

g = sns.catplot(y="Age",x="Sex",data=full_dataset,kind="box")
g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=full_dataset,kind="box")
g = sns.catplot(y="Age",x="Parch", data=full_dataset,kind="box")
g = sns.catplot(y="Age",x="SibSp", data=full_dataset,kind="box")
#Convert Sex feature into 0 and 1 and then check correlation matrix.

full_dataset["Sex"] = full_dataset["Sex"].map({"male": 0, "female":1})

g = sns.heatmap(full_dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
#Get indexes of rows with NaN as age.
#We are getting indexes of all the columns and then getting back all the indexes where Age is null

index_NaN_age = list(full_dataset["Age"][full_dataset["Age"].isnull()].index)


for i in index_NaN_age:
    age_med = full_dataset["Age"].median()
    age_pred = full_dataset["Age"][((full_dataset['SibSp'] == full_dataset.iloc[i]["SibSp"]) 
                                    & (full_dataset['Parch'] == full_dataset.iloc[i]["Parch"]) 
                                    & (full_dataset['Pclass'] == full_dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        full_dataset['Age'].iloc[i] = age_pred
    else :
        full_dataset['Age'].iloc[i] = age_med
#Name, Get title from the name.

full_dataset_title = [i.split(",")[1].split(".")[0].strip() for i in full_dataset["Name"]]
full_dataset["Title"] = pd.Series(full_dataset_title)
full_dataset["Title"].head()
#Histogram for Titles

g = sns.countplot(x="Title", data = full_dataset)
g = plt.setp(g.get_xticklabels(), rotation = 45)

#Replace with Rare
full_dataset["Title"] = full_dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 
                                             'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
#Replace feminine titles with Ms.
full_dataset["Title"] = full_dataset["Title"].replace(['Miss', 'Ms','Mme','Mlle', 'Mrs'], 'Ms')

#Map titles
full_dataset["Title"] = full_dataset["Title"].map({"Master":0, "Ms":1 ,"Mr":2, "Rare":3})
full_dataset["Title"] = full_dataset["Title"].astype(int)

#Histogram Again
g = sns.countplot(x="Title", data = full_dataset)
g = g.set_xticklabels(["Master","Ms","Mr","Rare"])
#Let us see if survival chnaces depends on titles.

g = sns.catplot(x="Title", y= "Survived", data = full_dataset, kind ='bar')
g = g.set_xticklabels(["Master","Ms","Mr","Rare"])
#We can now remove Name column

full_dataset.drop(["Name"], axis=1, inplace=True)
full_dataset.head()
#Drop extra columns

full_dataset.drop(["PassengerId","Embarked","Pclass","Sex", "Ticket","Parch", "SibSp", 
                   "PClass_encoded","Sex_encoded","Embarked_encoded"]
                , axis =1, inplace=True)

full_dataset.head()
train_len = len(train_data)
train_len
#Let's divide data into Train and test now.

train_data = full_dataset[:train_len]
test_data = full_dataset[train_len:]

#drop survived column from test_data

test_data.drop(["Survived"], axis =1, inplace=True)

print(train_data.shape)
print(test_data.shape)
      
#Separating features and target variable from training data

train_data["Survived"] = train_data["Survived"].astype(int)
train_data["Fare"] = train_data["Fare"].astype(float)


y_train = train_data["Survived"]
X_train = train_data.drop(labels=["Survived"], axis =1)

#10 fold cross validation

kfold = StratifiedKFold(n_splits=10)
#Modeling Steps.

random_state = 42
classifiers = []
classifiers.append(SVC(random_state = random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                                      random_state=random_state,learning_rate =0.1))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))



cv_results = []

for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y_train, scoring="accuracy", cv = kfold, n_jobs=-1))
    
cv_means = []
cv_std = []

for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValErrors": cv_std, 
                       "Algorithm": ["SVC", "RandomForestClassifier", "AdaBoostClassifier",
                                     "Decision Tree Classifier", "Extra Trees" ,"Gradient Boosting", 
                                     "K Nearest Neighbors", "Logistic regression"]})

cv_res = cv_res.sort_values(by = "CrossValMeans", ascending=False)
cv_res
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
#Predictions on test data
gbc_clf = GradientBoostingClassifier(random_state=random_state)
gbc_clf.fit(X_train, y_train)

Y_pred = gbc_clf.predict(test_data)
submission = pd.DataFrame({
        "PassengerId": IDtest,
        "Survived": Y_pred
    })

submission.to_csv('Titanic_Prediction_v3.csv', index=False)
