import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/framinghamnew/framingham.csv", na_values='?',header=None)

df.columns= ['Gender (Male)','Age','Education','Current Smoker','Cigs/Day','BP Meds','Prevalent Stroke','Prevalent Hypertension','Diabetes', 'Total Cholesterol', 'Systolic BP',

            'Diastolic BP', 'BMI', 'Heart Rate', 'Glucose','Ten Year CHD']

df.drop(df.index[0], inplace = True)

df.head(10)
df.isnull().sum()
df.shape
df.info()
df = df.apply(pd.to_numeric, errors='coerce')
df.info()
df.describe()
count=0

for i in df.isnull().sum(axis=1):

    if i>0:

        count=count+1

print('Total number of rows with missing values is ', count)

print('Since the number of rows with missing values are only',round((count/len(df.index))*100), '% of the entire dataset given, we decide to drop the rows with the missing or NaN values.')
df = df.dropna()
df.isnull().sum()
df.shape
df.mean()
df.boxplot(column='Cigs/Day')
df.loc[df['Cigs/Day'] > 59, 'Cigs/Day'] = 9.02
df.boxplot(column='Cigs/Day')
df.boxplot(column='Education')
df.boxplot(column='Age')
df["Age"].describe()
df['Total Cholesterol'].describe()
df.boxplot(column='Total Cholesterol')
df.loc[df['Total Cholesterol'] > 480, 'Total Cholesterol'] = 237
df.boxplot(column='Total Cholesterol')
df['Total Cholesterol'].describe()
df['Systolic BP'].describe()
df.boxplot(column='Systolic BP')
df.loc[df['Systolic BP'] > 270, 'Systolic BP'] = df['Systolic BP'].median()
df.boxplot(column='Systolic BP')
df['Systolic BP'].describe()
df['Diastolic BP'].describe()
df.boxplot(column='Diastolic BP')
df['BMI'].describe()
df.boxplot(column='BMI')
df.loc[df['BMI'] > 49, 'BMI'] = df['BMI'].median()
df.boxplot(column='BMI')
df['BMI'].describe()
df['Heart Rate'].describe()
df.boxplot(column='Heart Rate')
df['Glucose'].describe()
df.boxplot(column='Glucose')
def draw_histograms(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')

        ax.set_title(feature+" Distribution",color='DarkRed')

        

    fig.tight_layout()  

    plt.show()

draw_histograms(df,df.columns,6,3)
sn.countplot(x='Ten Year CHD',data=df)

df["Ten Year CHD"].value_counts()

sn.pairplot(data=df)

df = df.drop("Education", axis=1)
df.head(10)
X = df.drop(['Ten Year CHD'], axis = 1)

y = df['Ten Year CHD']

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



model_RF = RandomForestClassifier(max_depth = 25, n_estimators = 100).fit(X_train, y_train)

model_SVM = svm.SVC().fit(X_train, y_train)

model_LR = LogisticRegression().fit(X_train, y_train)



print("The score of Test Data Set for Random Forest Classifier is ", model_RF.score(X_test, y_test))

print("The score of Train Data Set for Random Forest Classifier is",model_RF.score(X_train, y_train))





print("The score of Test Data Set for Support Vector Machine is   ",model_SVM.score(X_test, y_test))

print("The score of Train Data Set for Support Vector Machine is  ",model_SVM.score(X_train, y_train))





print("The score of Test Data Set for Linear Regression is        ",model_LR.score(X_test, y_test))

print("The score of Train Data Set for Linear Regression is       ",model_LR.score(X_train, y_train))
model_RF = RandomForestClassifier(max_depth = 25, n_estimators = 100).fit(X_train, y_train)

model_SVM = svm.SVC().fit(X_train, y_train)

model_LR = LogisticRegression().fit(X_train, y_train)



a = model_RF.predict(X_train)

b = model_SVM.predict(X_train)

c = model_LR.predict(X_train)



d = []

for i in range(len(a)):

    d.append([a[i], b[i], c[i]])

d = np.asarray(d)



model = LogisticRegression().fit(d, y_train)

model.score(d, y_train)
a = model_RF.predict(X_test)

b = model_SVM.predict(X_test)

c = model_LR.predict(X_test)



d = []

for i in range(len(a)):

    d.append([a[i], b[i], c[i]])

d = np.asarray(d)



print(model_RF.score(X_test, y_test))

print(model_SVM.score(X_test, y_test))

print(model_LR.score(X_test, y_test))

model.score(d, y_test)