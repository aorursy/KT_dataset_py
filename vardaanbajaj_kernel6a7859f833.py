# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
test_df.head()
print("No. of training examples: " + str(train_df.shape[0]))
print("No. of testing examples: " + str(test_df.shape[0]))
# check for missing values
train_df.isna().sum()
# calculating percent missing values for Age, Cabin, Embarked
print("Percentage missing age values: " + str((train_df['Age'].isna().sum()/train_df.shape[0])*100))
print("Percentage missing cabin values: " + str((train_df['Cabin'].isna().sum()/train_df.shape[0])*100))
print("Percentage missing embarked values: " + str((train_df['Embarked'].isna().sum()/train_df.shape[0])*100))
# Distribution of ages in data
age_distribution = dict(train_df["Age"].value_counts())
lists = sorted(age_distribution.items()) # sorted by key
x, y = zip(*lists)
plt.plot(x, y)
plt.show()
train_data = train_df.copy()
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True) # Since age distribution is a bit skewed towards the left side, it's better we use median of ages to fill the missing age values
train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True) # Filling with most frequent occuring value
train_data.drop('Cabin', axis=1, inplace=True) # Dropping Cabin column since 77% cabin values are missing
train_data.isna().sum()
train_data.head()
# creating one hot encodings for Pclass, Embarked, Sex
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training
final_train.head()
#Applying same changes to test set
test_df.isna().sum()
test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()

sc = StandardScaler()
final_train[["Age", "Fare"]] = sc.fit_transform(final_train[["Age", "Fare"]])
final_test[["Age", "Fare"]] = sc.fit_transform(final_test[["Age", "Fare"]])

final_train.head()
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
cols = ["Age", "SibSp", "Parch", "Fare", "Pclass_1", "Pclass_2", "Pclass_3", "Embarked_C", "Embarked_Q", "Embarked_S", "Sex_male"]
X = final_train[cols]
y = final_train['Survived']
model = LogisticRegression()

# selecting top 8 features
rfe = RFE(model, n_features_to_select = 8)
rfe = rfe.fit(X, y)
print('Top 8 most important features: ' + str(list(X.columns[rfe.support_])))
selected_features = ['Age', 'SibSp', 'Pclass_1', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Sex_male']

X = final_train[selected_features]
y = final_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

print('Train/Test split results:')
print("Logistic Regression accuracy is: " + str(accuracy_score(y_test, y_pred)))
print("Logistic Regression log_loss is: " + str(log_loss(y_test, y_pred_proba)))
final_test['Survived'] = logreg.predict(final_test[selected_features])
final_test['PassengerId'] = test_df['PassengerId']

results = final_test[['PassengerId','Survived']]
results.to_csv("submission.csv", index=False)
