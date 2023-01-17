# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection



# reading raw data into a dataframe



df = pd.read_csv("../input/titanic/train.csv")



print (df.shape)

print (df.isnull().sum())
# Removed Name, Ticket and Cabin columns from df



df.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)



print (df.shape)
# combining Spuse/siblings/parents/children as family



#df['Family_members'] = df['SibSp'] + df['Parch']



#print (df.isnull().sum())

#df.head()
fig, axes = plt.subplots(3, 4, figsize=(20,15))



sns.countplot(x = "Survived", hue = "Pclass", data=df, ax = axes[0,0])



sns.boxplot(x = "Pclass", y = "Age", data=df, ax = axes[0,2])



sns.boxplot(x = "Pclass", y = "Fare", data=df, ax = axes[0,3])



#sns.countplot(x = "Survived", hue = "Family_members", data=df, ax=axes[0,1])



sns.boxplot(x = "Survived", y = "Fare", data=df, ax = axes[1,0])



sns.boxplot(x = "Survived", y = "Age", data=df, ax = axes[1,1])



sns.countplot(x = "Survived", hue = "Sex", data=df, ax = axes[1,3])



sns.countplot(x = "Survived", hue = "Embarked", data=df, ax = axes[1,2])



sns.countplot(x = "Pclass", hue = "Embarked", data=df, ax = axes[2,0])



sns.boxplot(x = "Sex", y = "Age", data=df, ax = axes[2,1])



axes[2,1].set(ylim=(25,30))



sns.boxplot(x = "Pclass", y = "Age", hue = "Sex", data=df, ax = axes[2,2])



axes[2,2].set(ylim=(10,50))
# plot of Pclass vs Age for male/female shows the median age wrt to Pclass and Sex. Use this to fill up missing age values for given Pclass and Sex



for i in range(len(df)):

    if np.isnan(df.loc[i, "Age"]) == True and df.loc[i, "Pclass"] == 1 and df.loc[i, "Sex"] == 'male':

        df.loc[i, "Age"] = 40

    

    elif np.isnan(df.loc[i, "Age"]) == True and df.loc[i, "Pclass"] == 1 and df.loc[i, "Sex"] == 'female':

        df.loc[i, "Age"] = 35

    

    elif np.isnan(df.loc[i, "Age"]) == True and df.loc[i, "Pclass"] == 2 and df.loc[i, "Sex"] == 'male':

        df.loc[i, "Age"] = 30



    elif np.isnan(df.loc[i, "Age"]) == True and df.loc[i, "Pclass"] == 2 and df.loc[i, "Sex"] == 'female':

        df.loc[i, "Age"] = 28    

    

    elif np.isnan(df.loc[i, "Age"]) == True and df.loc[i, "Pclass"] == 3 and df.loc[i, "Sex"] == 'male':

        df.loc[i, "Age"] = 25



    elif np.isnan(df.loc[i, "Age"]) == True and df.loc[i, "Pclass"] == 3 and df.loc[i, "Sex"] == 'female':

        df.loc[i, "Age"] = 22
# Drop rows from traning set for which Embarkation place is not available



df.dropna(axis=0, how='any', inplace = True)

df.head(5)

print (df.shape)

print (df.isnull().sum())



# using LabelEncoder functionality to encode categorical variables



from sklearn.preprocessing import LabelEncoder

df["Sex_code"] = LabelEncoder().fit_transform(df["Sex"])

df["Embarked_code"] = LabelEncoder().fit_transform(df["Embarked"])



df.head(10)
#X_train = df[[ "Pclass", "Age", "Family_members","Fare","Sex_code", "Embarked_code"]]

X_train = df[[ "Pclass", "Age", "SibSp","Parch", "Sex_code"]]

y_train = df[["Survived"]]

z_train = df[["PassengerId"]]

X_train.head(5)



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import StandardScaler



#X_train[['Age','Fare']] = MinMaxScaler().fit_transform(X_train[['Age','Fare']])

#X_train[['Age','Fare']] = StandardScaler().fit_transform(X_train[['Age','Fare']])

#X_train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



models = []

models.append(('LogisticReg', LogisticRegression(solver='liblinear',multi_class='auto',max_iter=1000)))

models.append(('LR_newton-cg', LogisticRegression(solver='newton-cg', multi_class='auto', max_iter=1000)))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('QDA', QuadraticDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier(n_neighbors=10, weights='distance')))

models.append(('Decision Tree', DecisionTreeClassifier()))

models.append(('GaussianNB', GaussianNB()))

models.append(('SVM_rbf', SVC(gamma='auto')))

models.append(('Random Forest', RandomForestClassifier(n_estimators=100)))

models.append(('Neural Network', MLPClassifier(activation = 'tanh', solver='adam', alpha=1e-5, random_state=1, max_iter = 10000)))



scoring = 'accuracy'

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, shuffle=True)

    cv_results = model_selection.cross_val_score(model, X_train, y_train.values.ravel(), cv=kfold, scoring = scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
df_test = pd.read_csv("../input/titanic/test.csv")

print (df_test.shape)

print (df_test.isnull().sum())
# Removed Name, Ticket and Cabin columns from df_test



df_test.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)



print (df_test.shape)



# combining Spuse/siblings/parents/children as family



#df_test['Family_members'] = df_test['SibSp'] + df_test['Parch']



print (df_test.shape)



# plot of Pclass vs Age for male/female shows the median age wrt to Pclass and Sex. Use this to fill up missing age values for given Pclass and Sex



for i in range(len(df_test)):

    if np.isnan(df_test.loc[i, "Age"]) == True and df_test.loc[i, "Pclass"] == 1 and df_test.loc[i, "Sex"] == 'male':

        df_test.loc[i, "Age"] = 40

    

    elif np.isnan(df_test.loc[i, "Age"]) == True and df_test.loc[i, "Pclass"] == 1 and df_test.loc[i, "Sex"] == 'female':

        df_test.loc[i, "Age"] = 35

    

    elif np.isnan(df_test.loc[i, "Age"]) == True and df_test.loc[i, "Pclass"] == 2 and df_test.loc[i, "Sex"] == 'male':

        df_test.loc[i, "Age"] = 30



    elif np.isnan(df_test.loc[i, "Age"]) == True and df_test.loc[i, "Pclass"] == 2 and df_test.loc[i, "Sex"] == 'female':

        df_test.loc[i, "Age"] = 28    

    

    elif np.isnan(df_test.loc[i, "Age"]) == True and df_test.loc[i, "Pclass"] == 3 and df_test.loc[i, "Sex"] == 'male':

        df_test.loc[i, "Age"] = 25



    elif np.isnan(df_test.loc[i, "Age"]) == True and df_test.loc[i, "Pclass"] == 3 and df_test.loc[i, "Sex"] == 'female':

        df_test.loc[i, "Age"] = 22



print (df_test.shape)

print (df_test.isnull().sum())



# Fare data is missing from one row. This is for Pclass 3. so replacing it with median fare for class 3



df_test["Fare"].fillna(8, inplace = True)



# using LabelEncoder functionality to encode categorical variables



from sklearn.preprocessing import LabelEncoder

df_test["Sex_code"] = LabelEncoder().fit_transform(df_test["Sex"])

df_test["Embarked_code"] = LabelEncoder().fit_transform(df_test["Embarked"])



print (df_test.shape)

print (df_test.isnull().sum())

df_test.head()
#X_test = df_test[[ "Pclass", "Age", "Family_members","Fare","Sex_code", "Embarked_code"]]

X_test = df_test[[ "Pclass", "Age", "SibSp","Parch","Sex_code"]]

#y_test = df_test['Survived']

#z_test = df_test["PassengerId"]



print (X_test.isnull().sum())

X_test.head()
## Make predictions on validation dataset with Random Forest mainly to get features importance

ran_for = RandomForestClassifier(n_estimators=100)

ran_for.fit(X_train, y_train.values.ravel())

y_predict_rf = ran_for.predict(X_test)

#print('Accuracy_score: %.3f' % accuracy_score(y_test, y_predict_rf))

#print('Confusion_matrix:\n %s' % confusion_matrix(y_test, y_predict_rf))

#print('Classification_report:\n %s' % classification_report(y_test, y_predict_rf))

#print(ran_for.feature_importances_)
fig3 = plt.figure()

features = list(X_train.head(0))

importances = ran_for.feature_importances_

indices = np.argsort(importances)

plt.title('Feature Importances', fontsize=24)

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=18)

plt.xlabel('Relative Importance', fontsize=20)

plt.xticks(fontsize=18)

plt.show()
## Make predictions on validation dataset with Neural network

NN = MLPClassifier(activation = 'tanh', solver='adam', alpha=1e-5, random_state=1, max_iter = 10000)

NN.fit(X_train, y_train.values.ravel())

y_predict = NN.predict(X_test)
y_predict_df = pd.DataFrame({'Survived': y_predict})



predictions = pd.merge(df_test["PassengerId"], y_predict_df, left_index = True, right_index = True)



predictions.head(20)
predictions.to_csv("predictions_with_NN.csv", index=False)