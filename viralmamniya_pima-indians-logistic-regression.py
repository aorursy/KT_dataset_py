# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pdata = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

pdata.head()
from pandas_profiling import ProfileReport

#EDA using Pandas Profiling

dataset = pdata

profile = ProfileReport(dataset, title='Pandas Profiling Report')

profile.to_widgets()
profile.to_notebook_iframe()
profile.to_file("your_report.html")
pdata.shape # Check number of columns and rows in data frame
pdata.isnull().values.any() # If there are any null values in data set
sns.pairplot(pdata,diag_kind='kde')
# Correlation 

corr=pdata.corr()

f,ax=plt.subplots(1,1,figsize=(12,8))

sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
pdata.describe()
pdata.info()
df=pdata.loc[(pdata.BMI>10) & (pdata.BloodPressure>20) & (pdata.Glucose>25)]

df.head()
df.shape
df.loc[(df.SkinThickness<5)& (df.Outcome==0), 'SkinThickness']=int(df[(df.Outcome==0)]['SkinThickness'].median())

df.loc[(df.SkinThickness<5)& (df.Outcome==1), 'SkinThickness']=int(df[(df.Outcome==1)]['SkinThickness'].median())

df.head()
df.loc[(df.Insulin==0)& (df.Outcome==0), 'Insulin']=int(df[(df.Outcome==0)]['Insulin'].median())

df.loc[(df.Insulin==0)& (df.Outcome==1), 'Insulin']=int(df[(df.Outcome==1)]['Insulin'].median())

df.head()
n_true = len(df.loc[df['Outcome'] == True])

n_false = len(df.loc[df['Outcome'] == False])

print("Number of true cases: {0} ({1:2.2f}%)".format(n_true, (n_true / (n_true + n_false)) * 100 ))

print("Number of false cases: {0} ({1:2.2f}%)".format(n_false, (n_false / (n_true + n_false)) * 100))
from sklearn.model_selection import train_test_split



X = df.drop('Outcome',axis=1)     # Predictor feature columns (8 X m)

Y = df['Outcome']   # Predicted class (1=True, 0=False) (1 X m)



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 1 is just any random seed number



x_train.head()
from sklearn import metrics



from sklearn.linear_model import LogisticRegression



# Fit the model on train

model = LogisticRegression(solver="liblinear")

model.fit(x_train, y_train)

#predict on test

y_predict = model.predict(x_test)





coef_df = pd.DataFrame(model.coef_)

coef_df['intercept'] = model.intercept_

print(coef_df)
model_score = model.score(x_test, y_test)

print(model_score)