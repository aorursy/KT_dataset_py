import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('fivethirtyeight')

%matplotlib inline
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.info()
df.head()
df.isnull().sum()
df.nunique().sort_values()
df = df.drop(columns=['EmployeeCount','EmployeeNumber', 'Over18','StandardHours'])
df.describe()
plt.subplots(figsize=(30,30))

sns.heatmap(df.corr(), annot=True, linewidths=0.8);
# drop highly correlated column

df = df.drop(columns=['JobLevel','MonthlyIncome', 'TotalWorkingYears',

                      'YearsInCurrentRole', 'YearsWithCurrManager', 

                      'YearsSinceLastPromotion'])
df['Attrition'] = df['Attrition'].replace('Yes', 0)

df['Attrition'] = df['Attrition'].replace('No', 1)

df['Attrition'] = df['Attrition'].astype('int64')
df.Attrition.value_counts()

# imbalance

# target column
print(df.Attrition.value_counts())



print(df.BusinessTravel.value_counts())

print(df.Department.value_counts())

print(df.EducationField.value_counts())

print(df.Gender.value_counts())

print(df.JobRole.value_counts())

print(df.MaritalStatus.value_counts())

print(df.OverTime.value_counts())
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

colnames_to_encode = ['Attrition','BusinessTravel', 'Department', 'EducationField', 

                      'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

for c in colnames_to_encode:

    df[c] = label_encoder.fit_transform(df[c])

print(df.Attrition.value_counts())

print(df.BusinessTravel.value_counts())

print(df.Department.value_counts())

print(df.EducationField.value_counts())

print(df.Gender.value_counts())

print(df.JobRole.value_counts())

print(df.MaritalStatus.value_counts())

print(df.OverTime.value_counts())

target = df.Attrition

df = df.drop(labels=['Attrition'], axis=1)

df.insert(24,'Attrition',target)

df.head()
X = df.iloc[:, :-1].values   

y = df.iloc[:, 24].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 10, ratio=1.0)

X_train_sm,  y_train_sm = sm.fit_sample(X_train, y_train)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



model = LogisticRegression(solver='lbfgs', random_state=9)

rfe = RFE(model, 5)

fit = rfe.fit(X_train_sm, y_train_sm)

print("Num Features: %s" % (fit.n_features_))

print("Selected Features: %s" % (fit.support_))

print("Feature Ranking: %s" % (fit.ranking_))
df.info()
df = df.drop(columns=['BusinessTravel','DailyRate', 'Department',

                      'DistanceFromHome', 'Education', 'EducationField', 

                      'NumCompaniesWorked', 'Gender', 'HourlyRate',

                      'JobRole', 'JobSatisfaction','MonthlyRate', 'PercentSalaryHike',

                      'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',

                      'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany'])
model.fit(X_train_sm, y_train_sm)

y_pred = model.predict(X_test)
from sklearn import metrics

metrics.confusion_matrix(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
y_hats = model.predict_proba(X_test)
y_hats2 = model.predict(X)



df['y_hats'] = y_hats2



df.to_csv('data1.csv')