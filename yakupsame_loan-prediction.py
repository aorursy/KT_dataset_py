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
from sklearn.preprocessing import OneHotEncoder
train = pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/train_csv.csv')

test = pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/test.csv.csv')
train.head()
test.head()
test_id =test['Loan_ID']

train.shape
test.shape
train.describe()
train['Education'].value_counts()
train['Loan_Amount_Term'].value_counts()
train['Property_Area'].value_counts()
train['Gender'].value_counts()
train.isnull().sum()
train.head()
train['test'] = 0

test['test'] = 1
train.head()
test.head()
df = train.append(test, sort = False)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=[20,20])

plt.subplot(411)

sns.countplot(x = 'Education', hue = 'Loan_Status', data = df)

plt.subplot(412)

sns.countplot(x = 'Married', hue = 'Loan_Status', data = df)

plt.subplot(413)

sns.countplot(x = 'Self_Employed', hue = 'Loan_Status', data = df)

plt.subplot(414)

sns.countplot(x = 'Property_Area', hue = 'Loan_Status', data = df)
df.shape
df.tail(10)
df['Loan_Status'].value_counts()
df.isnull().sum()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(df.iloc[ :,[8,9,10]])

df.iloc[ :,[8,9,10]] = imputer.transform(df.iloc[ :,[8,9,10]])
imputer2 = SimpleImputer(missing_values = np.nan , strategy = 'most_frequent')

imputer2 = imputer2.fit(df.iloc[ : ,[1,2,3,5]])

df.iloc[ : ,[1,2,3,5]] = imputer2.transform(df.iloc[ : ,[1,2,3,5]])
df.isnull().sum()
df_cat = pd.get_dummies(data=df, columns=['Gender', 'Married','Education','Self_Employed','Property_Area'])
df_cat['Loan_Status'] = df_cat['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)
df_cat.head()
df_cat.drop(['Loan_ID'],axis =1,inplace=True)


df_cat.hist(figsize = (20,20))


correlations = df_cat.corr()

f , ax =plt.subplots(figsize=(20,20))

sns.heatmap(correlations,annot=True,cmap="coolwarm")
df_cat.head()
df_cat.describe()
df_cat['Dependents'] = df_cat['Dependents'].replace(['3+'],'3')
plt.figure(figsize=(15,10))

plt.boxplot(df_cat['ApplicantIncome'])

plt.show()
plt.figure(figsize=(15,10))

plt.boxplot(df_cat['CoapplicantIncome'])

plt.show()
plt.figure(figsize=(15,10))

plt.boxplot(df_cat['LoanAmount'])

plt.show()
plt.scatter(x = df_cat['ApplicantIncome'],y= df_cat['CoapplicantIncome'])

plt.show()
import matplotlib.pyplot as plt

plt.scatter(x = train['Loan_Status'],y= train['Loan_Amount_Term'])

plt.show()
from scipy import stats

app_income = stats.zscore(df_cat['ApplicantIncome'])

coapp_income = stats.zscore(df_cat['CoapplicantIncome'])

df_cat.head()
scaled_columns = ['ApplicantIncome','CoapplicantIncome','LoanAmount']

scaled_columns1=['Loan_Amount_Term','Dependents']

from sklearn.preprocessing import RobustScaler , StandardScaler

scaler = RobustScaler()

df_cat[scaled_columns] = scaler.fit_transform(df_cat[scaled_columns])

scaler2=StandardScaler()

df_cat[scaled_columns1] = scaler2.fit_transform(df_cat[scaled_columns1])
df_cat.head(10)



test = df_cat[df_cat['test'] == 1]

train = df_cat[df_cat['test'] == 0]
test.head()
train.head()
train.drop(['test'],axis=1,inplace=True)

test.drop(['test'],axis=1,inplace=True)
train.head()
test.head()
test.drop(['Loan_Status'],axis=1,inplace=True)

test.head()
train['Loan_Status'].value_counts()
y = train['Loan_Status']

train.drop(['Loan_Status'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.3,random_state=123)
x_train
x_test
y_train
y_test
from sklearn.metrics import accuracy_score,confusion_matrix , classification_report
from xgboost import XGBClassifier

classifier_xg = XGBClassifier()

classifier_xg.fit(x_train , y_train)



y_pred_xg = classifier_xg.predict(x_test)

accuracy_score(y_test,y_pred_xg)

from sklearn.linear_model import LogisticRegression

classifier_log = LogisticRegression(random_state = 123)

classifier_log.fit(x_train,y_train)



y_pred_log = classifier_log.predict(x_test)

accuracy_score(y_test,y_pred_log)

from sklearn.ensemble import RandomForestClassifier

classifier_forest = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',random_state=123)

classifier_forest.fit(x_train,y_train)



y_pred_forest = classifier_forest.predict(x_test)

accuracy_score(y_test,y_pred_forest)
from sklearn.svm import SVC

classifier_svm = SVC(kernel='linear')

classifier_svm.fit(x_train,y_train)



y_pred_svm = classifier_svm.predict(x_test)

accuracy_score(y_test,y_pred_svm)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier_svm,X = x_train,y = y_train,cv=10)

print("Accuracy: {:.2f} % ".format(accuracies.mean()*100))

print("Standard Deviation {:.2f} % : ".format(accuracies.std()*100))
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [0.25,0.5,0.75,1],'kernel' :['linear']},

              {'C': [0.25,0.5,0.75,1],'kernel' :['rbf'] , 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_search = GridSearchCV(estimator = classifier_svm,

                          param_grid = parameters,

                          scoring = 'accuracy',

                          cv = 10,

                          n_jobs = -1)

grid_search.fit(x_train,y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} % ".format(best_accuracy*100))

print("Best Parameters : ",best_parameters)
classifier = SVC(C = 0.25,kernel='linear')

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

accuracy_score(y_test,y_pred)

print(classification_report(y_test,y_pred))
y_test = classifier.predict(test)
submission = submission = pd.DataFrame({

        "Loan_Id": test_id,

        "Loan_Status": y_test

    })

submission.head(10)