import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv("../input/train.csv")

df.head()
## Find the rows and columns in the Data Frame

print(df.shape)

## To find the Columns which have atleast one NaN Value.

df.loc[:,df.isna().any()]

## we got 3 Column - Age, Cabin and Embarked

df.loc[:,df.isna().any()].columns.tolist()
## Missing Values are Imputed with MEAN, in new Column 'Age_New'

# print(df['Age'].mode()[0])

df['Age_New'] = df['Age'].fillna(df['Age'].mean())

# df['Age_New'] = df['Age'].fillna(df['Age'].mode()[0])

# df['Age_New'] = df['Age'].fillna(df['Age'].max())

print(df['Age_New'].describe())
## Cabin is Category Type Feature.

# df[df['Cabin'].isnull()].count() ## 687 Cabin's are NaN - which is very high

# check how Cabin effects the Survied?? 

# df[df['Cabin'].isnull()== False]['Survived'].value_counts() ## impact is huge to Survived where cabin is NaN

# to populate the NaN values, check the Fare vs Cabin

# df[(df['Cabin'].isnull()== False) & (df['Survived']==1)].loc[:,('Fare','Embarked')].sort_values(['Fare'])



##Create a new Column Cabin_New - Making Nan as 1 and rest a 0.

df['Cabin_New'] = df['Cabin'].fillna(1)

df['Cabin_New'] = df['Cabin_New'].apply(lambda x: 0 if x != 1 else x)

df['Cabin_New'].value_counts()
# print(df[df['Embarked'].isnull()])

df[(df['Fare'] > 80.0) &(df['Pclass'] == 1)].sort_values(['Fare'])

#replaing NaN by S

df['Embarked_New'] = df['Embarked'].fillna('S')

df['Embarked_New'].value_counts()

df.columns
##One Hot Encoding for features - Pclass, Sex, Embarked

df1 = pd.get_dummies(df,columns=['Pclass','Sex','Embarked_New'],drop_first=True)

df1.head()
# del df1['Pclass_3','Sex_female','Embarked_New_Q'] # No need to run this as drop_first=True, has worked well.
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
df1.columns

#
#Train on Features - 'SibSp', 'Parch','Age_New', 'Cabin_New', 'Pclass_2','Pclass_3', 'Sex_male', 

#'Embarked_New_Q', 'Embarked_New_S'

##Note: .values -- changes the ouput to numpy array - which is needed by Models

X = df1[['SibSp', 'Parch','Age_New', 'Cabin_New', 'Pclass_2','Pclass_3', 

              'Sex_male','Embarked_New_Q', 'Embarked_New_S']].values

print(type(X))
y = df1[['Survived']].values

print(type(y))
#X_train and y_train

#X_test and y_test

#Split is 75% and 25%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# print(X_train,y_train)
print(type(X_train))

plt.hist(X_train[:,2], bins=20);
##### For Feature Scaling Import the Class from SkLearn

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
##Fit Transform

X_train_sc = sc.fit_transform(X_train)

X_train_sc
sc.mean_
sc.var_
##Transform the X_Test, they same X_train was Fit_Transform

X_test_sc = sc.transform(X_test)
import seaborn as sns
# let's look at how transformed age looks like compared to the original variable

sns.jointplot(X_train[:,2], X_train_sc[:,2], kind='kde')
#Never Scale Y - as Y is 0 or 1.

# y_train_sc = sc.fit_transform(y_train)

# y_test_sc = sc.transform(y_test)

# print(y_test_sc)
# Fitting Logistic Regression to the Training set, on scaled data set

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)

lr.fit(X_train_sc, y_train)

# models.append('Logistic Regression')
# lr.classes_

# lr.get_params

# lr.multi_class

lr.intercept_
print('Coeff: ',lr.coef_)

print('Intercept: ' ,lr.intercept_)

print('Iterations: ',lr.n_iter_)
##Predict the Model use the Scaled Data Frame X_test_sc

lr.predict(X_test_sc)
probs = lr.predict_proba(X_test_sc)

probs
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 

                             recall_score, f1_score)
print('Confusion Matrix for LR: \n',confusion_matrix(y_test, lr.predict(X_test_sc)))

print('Accuracy for LR: \n',accuracy_score(y_test, lr.predict(X_test_sc)))

# acc.append(accuracy_score(y_test, lr.predict(X_test_sc)))

print('Precision for LR: \n',precision_score(y_test, lr.predict(X_test_sc)))

# precision.append(precision_score(y_test, lr.predict(X_test_sc)))

print('Recall for LR: \n',recall_score(y_test, lr.predict(X_test_sc)))

# recall.append(recall_score(y_test, lr.predict(X_test_sc)))

print('f1_score for LR: \n',f1_score(y_test, lr.predict(X_test_sc)))

# f1.append(f1_score(y_test, lr.predict(X_test_sc)))
X_test_sc.shape
#df[(df['Name'].str.contains('Miss')) & (df['Survived'] == 0)].count()

# .groupby(['Survived']).count() ## to match the string in a column