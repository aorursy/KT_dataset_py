import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

cmap = sns.diverging_palette(250, 10, as_cmap=True)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
datasets = [train, test]
data = pd.concat([train, test])
train.head()
test.head()
train.info()
print('_'*40)
test.info()
train.isnull().sum()
test.isnull().sum()
train.dtypes
train.describe()
test.describe()
plt.subplots(figsize=(12,9))
sns.heatmap(train.drop(["PassengerId"], axis = 1).corr(), annot = True, cmap = cmap)
train["Embarked"].value_counts()
for dataset in datasets:
    dataset["Embarked"].fillna("S", inplace = True)
train.info()
print('_'*40)
test.info()
train[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending = False)
train[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Pclass")
train[["Embarked", "Survived"]].groupby(["Embarked"], as_index = False).mean().sort_values(by = "Embarked")
pd.crosstab([train["Embarked"], train["Pclass"]], [train["Sex"], train["Survived"]], margins = True).style.background_gradient(cmap = cmap)
for dataset in datasets:
    dataset["Embarked"] = dataset["Embarked"].map({"C": 0, "Q": 1, "S": 2})
train.head()
for dataset in datasets:
    dataset["FamilySize"] = dataset["SibSp"]+dataset["Parch"]+1
train.head()
pd.crosstab(train["FamilySize"], train["Survived"], margins = True).style.background_gradient(cmap = cmap)
data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                          'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        #A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      data.loc[data['Family_Survival']!=0.5].shape[0])
for _, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(data[data['Family_Survival']!=0.5].shape[0]))

 # Family_Survival in TRAIN_DF and TEST_DF:
train['Family_Survival'] = data['Family_Survival'][:891]
test['Family_Survival'] = data['Family_Survival'][891:]
for dataset in datasets:
    dataset["FamilySize"] = np.where((dataset["FamilySize"]) == 1 , 'Solo',
                           np.where((dataset["FamilySize"]) <= 4,'Medium', 'Big'))
    
    dataset["FamilySize"] = dataset["FamilySize"].map({"Solo":0, "Medium":1, "Big":2})
train.head()
for dataset in datasets:
    dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1})
train.head()
for dataset in datasets:
    dataset["Title"] = dataset["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
train.head()
pd.crosstab(train["Title"], train["Sex"], margins = True).sort_values(by = "All", ascending = False)
for dataset in datasets:
    dataset["Title"] = dataset["Title"].replace(["Dr", "Rev", "Major", "Col", "Mlle", "Don", "Jonkheer", "Lady", "Mme", "Countess", "Ms", "Sir", "Capt"], "Other")
    dataset["Title"] = dataset["Title"].map({"Mr":0, "Miss":1, "Mrs":2 , "Master": 3, "Other" :4})
for dataset in datasets:
    dataset["Title"].fillna(0, inplace = True)
for dataset in datasets:
        df = train.groupby(['Title', 'Pclass'])['Age']
        dataset['Age'] = df.transform(lambda x: x.fillna(x.mean()))
#for dataset in datasets:
#    dataset["Age"].fillna(dataset["Age"].mean(), inplace = True)
#data['AgeBin'] = pd.qcut(data['Age'], 4)
#
#from sklearn.preprocessing import LabelEncoder
#label = LabelEncoder()
#data['AgeBin_Code'] = label.fit_transform(data['AgeBin'])
#
#train['AgeBin_Code'] = data['AgeBin_Code'][:891]
#test['AgeBin_Code'] = data['AgeBin_Code'][891:]
#
#train.drop(['Age'], 1, inplace=True)
#test.drop(['Age'], 1, inplace=True)
for dataset in datasets:
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
for dataset in datasets:
    dataset["Fare"].fillna(dataset["Fare"].mean(), inplace = True)
for dataset in datasets:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train.head()
for dataset in datasets:
    dataset.drop(["SibSp"], axis = 1, inplace = True)
    dataset.drop(["Parch"], axis = 1, inplace = True)
    dataset.drop(["Cabin"], axis = 1, inplace = True)
    dataset.drop(["Name"], axis = 1, inplace = True)
    dataset.drop(["PassengerId"], axis = 1, inplace = True)
    dataset.drop(["Ticket"], axis = 1, inplace = True)
    dataset.drop(["FamilySize"], axis = 1, inplace = True)
    
train.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(["Survived"], axis = 1), train["Survived"])
# Fitting Random Forest Classification to the Training set
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_features='auto', 
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

# Creating the Grid Search Parameter list
parameters = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1, 5, 10],
             "min_samples_split" : [12, 16, 20, 24],
             "n_estimators": [100, 400, 700]}

# Setting up the gridSearch to find the optimal parameters
gridSearch = GridSearchCV(estimator=classifier,
                  param_grid=parameters,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

# Getting the optimal grid search parameters
gridSearch = gridSearch.fit(X_train, y_train)

# Printing the out of bag score and the best parameters values
print(gridSearch.best_score_)
print(gridSearch.best_params_)

# building the random forrest classifier
classifier = RandomForestClassifier(criterion='entropy', 
                             n_estimators=100,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
classifier.fit(X_train, y_train)
print("%.5f" % classifier.oob_score_)

# Creating the list of important features
pd.concat((pd.DataFrame(X_train.columns, columns = ['variable']), 
           pd.DataFrame(classifier.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:8]

prediction = classifier.predict(test)

temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = prediction
temp.to_csv("../working/submission.csv", index = False)