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
# import packages:
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.describe()
test.describe()
# left only important features (drop id's)
train_df = train.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test.drop(['Name','Ticket'], axis=1)
# Embarked get dummies

train_df["Embarked"] = train_df["Embarked"].fillna("S")

#sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0], ax=axis2)

embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

embark_dummies_train  = pd.get_dummies(train_df['Embarked'].values)
embark_dummies_test  = pd.get_dummies(test_df['Embarked'].values)

cols = embark_dummies_train.columns
train_df[cols] = embark_dummies_train
test_df[cols]  = embark_dummies_test

train_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace = True)
# Sex get dummies
def get_person(x):
    return 'child' if x.Age < 16 else x.Sex
    
train_df['Person'] = train_df.apply(get_person,axis=1)
test_df['Person']    = test_df.apply(get_person,axis=1)

# No need to use Sex column since we created Person column
train_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace = True)
             
person_dummies_titanic = pd.get_dummies(train_df['Person'].values)
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'].values)
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

cols = person_dummies_titanic.columns
train_df[cols] = person_dummies_titanic
test_df[cols]    = person_dummies_test

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=train_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = train_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

train_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)
# Pclass get dummies
pclass_dummies_titanic  = pd.get_dummies(train_df['Pclass'].values)
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'].values)
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

cols = pclass_dummies_titanic.columns
train_df[cols] = pclass_dummies_titanic
test_df[cols] = pclass_dummies_test
#Cabin: drop this feature because too much nans
train_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)
# impute nan to mean values
train_df['Age'] = train_df['Age'].fillna(train_df["Age"].mean())
test_df['Age'] = test_df['Age'].fillna(train_df["Age"].mean())
# some simle ensebble models + knn
def get_ensemble_models():
    rf = RandomForestClassifier(n_estimators=101,min_samples_leaf=3, min_samples_split=2)
    bagg = BaggingClassifier(n_estimators=101,random_state=42)
    extra = ExtraTreesClassifier(n_estimators=101,random_state=42)
    ada = AdaBoostClassifier(n_estimators=101,random_state=42)
    grad = GradientBoostingClassifier(n_estimators=101,random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=3)
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    #xgb = xgboost.XGBClassifier(n_estimators=101, random_state = 42)
    classifier_list = [rf,bagg,extra,ada,grad, knn]
    classifier_results = np.array([np.array([]),np.array([]),np.array([]),np.array([]),np.array([]), np.array([]),np.array([])])
    classifier_targets = np.array([np.array([]),np.array([]),np.array([]),np.array([]),np.array([]), np.array([]),np.array([])])
    classifier_name_list = np.array(['Random Forests','Bagging','Extra Trees','AdaBoost','Gradient Boost', 'knn'])
    return classifier_list, classifier_name_list, classifier_results, classifier_targets
def append_evaluation_metrics(trained_model, trained_model_name, X_test,y_test):
    cr = [np.append(classifier_results[i], trained_model.predict(X_test)) if np.where(classifier_name_list == trained_model_name)[0][0] == i else classifier_results[i] for i in range(6)]
    ct = [np.append(classifier_targets[i], y_test) if np.where(classifier_name_list == trained_model_name)[0][0] == i else classifier_targets[i] for i in range(6)]
    return cr,ct
def print_evaluation_metrics(trained_models,trained_model_names,classifier_results, classifier_targets):
    for trained_model,trained_model_name,classifier_result, classifier_target in zip(trained_models, trained_model_names,classifier_results, classifier_targets):
        print('--------- For Model : ', trained_model_name, ' ---------------\n')
        print(metrics.classification_report(classifier_target,classifier_result))
        print("Accuracy Score : ",metrics.accuracy_score(classifier_target,classifier_result))
        print("Roc_Auc Score : ",metrics.roc_auc_score(classifier_target,classifier_result))
        print("---------------------------------------\n")
#create validate sample
X_train,X_valid,y_train,y_valid = train_test_split(train_df.drop(["Survived"], axis = 1).values, train_df['Survived'].values, stratify =train_df['Survived'].values, test_size = 0.25)
%%time
classifier_list, classifier_name_list, classifier_results, classifier_targets = get_ensemble_models()
for classifier,classifier_name,classifier_result, classifier_target in tqdm(zip(classifier_list,classifier_name_list, classifier_results,classifier_targets)):
    classifier.fit(X_train, y_train)
    classifier_results, classifier_targets = append_evaluation_metrics(classifier, classifier_name, X_valid, y_valid)
print_evaluation_metrics(classifier_list,classifier_name_list,classifier_results, classifier_targets)
#best score on validation set got EXtraTree model
# train it on full data and score test sample:

extra = ExtraTreesClassifier(n_estimators=101,random_state=42).fit(train_df.drop('Survived',axis = 1).values,train_df['Survived'].values)
#score test sample
prediction = extra.predict(test_df.dropna(axis=1).values)
# save submisson to csv
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv('titanic.csv', index=False)