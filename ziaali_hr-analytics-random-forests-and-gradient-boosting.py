# Import Desired libraries.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
attrition=data
print(data.columns)
print(data.shape)
# Differentiate numerical features (minus the target) and categorical features
categorical_features = data.select_dtypes(include=['object']).columns
categorical_features

numerical_features = data.select_dtypes(exclude = ["object"]).columns
print(categorical_features.shape)
print(categorical_features)
print(numerical_features)
print(data.isnull().values.any())
data.describe() # this creates a kind of summary of the datset withh various statistical features.
sns.countplot("Attrition",data=data)
plt.show()
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat,annot=True,fmt= '.1f',vmax=.8, square=True)
x, y, hue = "Attrition", "prop", "Gender"
f, axes = plt.subplots(1,2)
sns.countplot(x=x, hue=hue, data=data, ax=axes[0])
prop_df = (data[x]
           .groupby(data[hue])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())
sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=axes[1])
x, y, hue = "Attrition", "prop", "Department"
f, axes = plt.subplots(1,2,figsize=(10,5))
sns.countplot(x=x, hue=hue, data=data, ax=axes[0])
prop_df = (data[x]
           .groupby(data[hue])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())
sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=axes[1])
sns.pairplot(data=data,x_vars=['MonthlyIncome'], y_vars=['Age'],height=5, hue='Attrition',palette="prism")
sns.set()
cols=['Age','DailyRate','Education','JobLevel','DistanceFromHome','EnvironmentSatisfaction','Attrition']
sns.pairplot(data[cols],hue='Attrition',height=2.5,palette="hls")
#for c in data.columns:
    #print("---- %s ---" % c)
    #print(data[c].value_counts())
data1=data
di={"Yes": 1, "No": 0}
data1["Attrition"].replace(di,inplace=True)

attrition=data
data1.shape
target=data.iloc[:,1]
print(target.head(5))
print(target.dtypes)
target=pd.DataFrame(target)
print(target.dtypes)
print(data1.columns)
data1.head(5)
data1.drop(["Attrition","Over18","StandardHours","EmployeeCount","EmployeeNumber"],axis=1,inplace=True)

categorical=data1.select_dtypes(include=['object']).columns
data1.shape
print(data1.columns)
print(categorical)
Prediction=data1##copy paste
print(data1.columns)
dummie=pd.get_dummies(data=data1, columns=['OverTime','BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole','MaritalStatus'])
dummie=pd.DataFrame(dummie)
new_data=pd.concat([data1, dummie], axis=1)
# print(new_data.columns)
print(target.head(5))
new_data.drop(['OverTime','BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole','MaritalStatus'],axis=1,inplace=True)
# Since we have already created dummy variables so we can drop the columns with categorical features.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(new_data,target,test_size=0.33,random_state=7)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
#  importing Libraries for our model 
# Importing Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=1000)
forest.fit(x_train,y_train.values.ravel())
predicted= forest.predict(x_test)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(x_train, y_train)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(x_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(x_test, y_test)))
    print()
# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set

gb = GradientBoostingClassifier(n_estimators=20,learning_rate = 0.5,random_state = 7)
gb.fit(x_train, y_train)
predictions = gb.predict(x_test)

print("Confusion Matrix for Gradient boosting:")
print(confusion_matrix(y_test, predictions))
print()
print("Classification Report for Gradient Boosting")
print(classification_report(y_test, predictions))
print("Accuracy score (validation): {0:.3f}".format(forest.score(x_test, y_test)))
print("Confusion Matrix for Random Forests:")
print(confusion_matrix(y_test, predicted))
print()
print("Classification Report for Random Forests")
print(classification_report(y_test, predicted))
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=7, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
forest_sm = RandomForestClassifier(n_estimators=500, random_state=7)
forest_sm.fit(x_train_res, y_train_res.ravel())
prediction2 = forest_sm.predict(x_test)
print("Accuracy score (validation): {0:.3f}".format(forest_sm.score(x_test, y_test)))
print("Confusion Matrix for Random Forests:")
print(confusion_matrix(y_test, prediction2))
print()
print("Classification Report for Random Forests")
print(classification_report(y_test, prediction2))
gb_sm = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 7)
gb_sm.fit(x_train_res, y_train_res.ravel())
prediction3 = gb_sm.predict(x_test)

print("Confusion Matrix for Gradient boosting:")
print(confusion_matrix(y_test, prediction3))
print()
print("Classification Report for Gradient Boosting")
print(classification_report(y_test, prediction3))
print("Accuracy score (validation): {0:.3f}".format(gb_sm.score(x_test, y_test)))






