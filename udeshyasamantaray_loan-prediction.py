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
import sklearn as sk
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
#data frame
tens_df=pd.read_csv("E:\\DATASET\\loan_data_set.csv")
tens_df
tens_df.info
tens_df.describe()
#create a plot and distribution analysis
tens_df['ApplicantIncome'].hist(bins=70,grid=False)
tens_df.boxplot(column='ApplicantIncome',grid=False,by='Education')
tens_df.boxplot(column='ApplicantIncome')
tens_df['LoanAmount'].hist(bins=100,grid=False)
tens_df.boxplot(column='LoanAmount')
#categorical value
temp=tens_df['Credit_History'].value_counts(ascending=True)
temp.plot(kind = 'bar')
#data munging
tens_df.apply(lambda x: sum(x.isnull()),axis=0)

tens_df['LoanAmount'].fillna(tens_df['LoanAmount'].mean(),inplace=True)

tens_df['Self_Employed'].fillna('No',inplace=True)

tens_df['Gender'].fillna(tens_df['Gender'].mode()[0], inplace=True)
tens_df['Married'].fillna(tens_df['Married'].mode()[0], inplace=True)
tens_df['Dependents'].fillna(tens_df['Dependents'].mode()[0], inplace=True)
tens_df['Loan_Amount_Term'].fillna(tens_df['Loan_Amount_Term'].mode()[0], inplace=True)
tens_df['Credit_History'].fillna(tens_df['Credit_History'].mode()[0], inplace=True)

tens_df
tens_df.apply(lambda x: sum(x.isnull()),axis=0)
#Label Encoder

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    tens_df[i] = le.fit_transform(tens_df[i])
tens_df 
#Training Model

X = tens_df[['Credit_History','ApplicantIncome','LoanAmount','Self_Employed']]
y = tens_df['Loan_Status']

#From Decision Tree

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X,y)
predictions = model.predict(X)

model.score(X,y)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(predictions,y)
cm
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('predict')
plt.ylabel('truth')
from sklearn.metrics import accuracy_score
print(accuracy_score(predictions,y))